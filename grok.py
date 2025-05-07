import pandas as pd
import numpy as np
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, StudentsT
from scipy.stats import multivariate_t
import matplotlib.pyplot as plt
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math

# 加载数据
data = pd.read_csv("history-index-data.csv")

print(data)

# 步骤 1：计算日均变化率（对数收益率）
log_returns = pd.DataFrame(
    {
        "SH50": np.log(data["SH50"] / data["SH50"].shift(1)).dropna() * 100,
        "HS300": np.log(data["HS300"] / data["HS300"].shift(1)).dropna() * 100,
        "ZZ500": np.log(data["ZZ500"] / data["ZZ500"].shift(1)).dropna() * 100,
    }
).dropna()

# 打印日均变化率摘要
print("\n日均变化率摘要：")
print(f"{'指数':<10} | {'平均日收益率 (%)':<25} | {'日收益率标准差 (%)':<25}")
print("-" * 60)


for col in log_returns.columns:
    mean_rate = log_returns[col].mean()
    std_rate = log_returns[col].std()
    print(f"{col:<10} | {mean_rate:<25.6f} | {std_rate:<25.6f}")


# 步骤 2：拟合DCC-GARCH(1,1)模型
def fit_dcc_garch(log_returns):
    print("\n正在拟合DCC-GARCH(1,1)模型...")

    # 为每个序列拟合单变量GARCH(1,1)，残差服从t分布
    models = {}
    std_residuals = {}
    cond_vol = {}

    for col in log_returns.columns:
        # 定义GARCH(1,1)模型，残差服从t分布
        am = arch_model(
            log_returns[col], mean="Constant", vol="GARCH", p=1, q=1, dist="t"
        )
        res = am.fit(disp="off")
        models[col] = res
        std_residuals[col] = res.std_resid  # 标准化残差
        cond_vol[col] = res.conditional_volatility  # 条件波动率

        # 打印GARCH参数
        print(f"\n{col} 的GARCH(1,1)参数：")
        print(f"mu: {res.params['mu']:.6f}")
        print(f"omega: {res.params['omega']:.6f}")
        print(f"alpha[1]: {res.params['alpha[1]']:.6f}")
        print(f"beta[1]: {res.params['beta[1]']:.6f}")
        print(f"nu (t分布自由度): {res.params['nu']:.6f}")

    # 计算动态相关性（简化的DCC估计）
    # 计算标准化残差的协方差
    std_resid_df = pd.DataFrame(std_residuals)
    Q_bar = std_resid_df.cov()  # 标准化残差的无条件协方差

    # 初始化DCC参数（示例值）
    a, b = 0.05, 0.9  # 典型的DCC参数
    T = len(log_returns)
    Q_t = np.zeros((T, 3, 3))
    R_t = np.zeros((T, 3, 3))
    Q_t[0] = Q_bar

    for t in range(1, T):
        z_t = std_resid_df.iloc[t - 1].values
        Q_t[t] = (1 - a - b) * Q_bar + a * np.outer(z_t, z_t) + b * Q_t[t - 1]
        Q_t_inv_sqrt = np.diag(1 / np.sqrt(np.diag(Q_t[t])))
        R_t[t] = Q_t_inv_sqrt @ Q_t[t] @ Q_t_inv_sqrt

    # 绘制动态相关性
    plt.figure(figsize=(10, 6))
    plt.plot(log_returns.index, R_t[:, 0, 1], label="SH50-HS300 correlation")
    plt.plot(log_returns.index, R_t[:, 0, 2], label="SH50-ZZ500 correlation")
    plt.plot(log_returns.index, R_t[:, 1, 2], label="HS300-ZZ500 correlation")
    plt.title("DCC-GARCH dynamic Correlation")
    plt.xlabel("date")
    plt.ylabel("Correlation coefficient")
    plt.legend()
    filename = "output-grok-dcc-garch-plot.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as high-resolution PNG
    plt.close()

    return models, cond_vol, R_t, Q_bar, a, b


def monte_carlo_simulation_chunk(
    chunk_id, n_simulations_chunk, n_days, indices, params, Q_bar, a, b
):
    print(f"\n正在运行蒙特卡洛模拟块 {chunk_id + 1}...")

    n_indices = len(indices)
    sim_returns_chunk = np.zeros((n_simulations_chunk, n_days, n_indices))

    for sim in range(n_simulations_chunk):
        Q_t = Q_bar
        R_t = np.eye(n_indices)
        vars_t = [params[col]["last_var"] for col in indices]
        sim_path = np.zeros((n_days, n_indices))

        for t in range(n_days):
            # Update conditional variances
            for i, col in enumerate(indices):
                epsilon_t = sim_path[t - 1, i] if t > 0 else 0
                vars_t[i] = (
                    params[col]["omega"]
                    + params[col]["alpha"] * epsilon_t**2
                    + params[col]["beta"] * vars_t[i]
                )

            # Simulate multivariate t-distributed residuals
            sigma_t = np.sqrt(vars_t)
            cov_t = np.diag(sigma_t) @ R_t @ np.diag(sigma_t)
            df = min(
                [params[col]["nu"] for col in indices]
            )  # Use the smallest degree of freedom
            z_t = multivariate_t.rvs(shape=cov_t, df=df, size=1)

            # Generate log returns
            for i, col in enumerate(indices):
                sim_path[t, i] = params[col]["mu"] + z_t[i]

            # Update DCC correlation
            Q_t = (1 - a - b) * Q_bar + a * np.outer(z_t, z_t) + b * Q_t
            Q_t_inv_sqrt = np.diag(1 / np.sqrt(np.diag(Q_t)))
            R_t = Q_t_inv_sqrt @ Q_t @ Q_t_inv_sqrt

        sim_returns_chunk[sim] = sim_path

    print(f"块 {chunk_id + 1} 完成。")

    return sim_returns_chunk


# 步骤 3：蒙特卡洛模拟
def monte_carlo_dcc_garch(
    log_returns,
    models,
    cond_vol,
    Q_bar,
    a,
    b,
    n_simulations=10000,
    n_days=52,
    n_processes=12,
):
    print("\n正在运行蒙特卡洛模拟...")

    start_time = pd.Timestamp.now()
    print("start time:", start_time)

    # Initialize parameters
    indices = log_returns.columns
    params = {
        col: {
            "mu": models[col].params["mu"],
            "omega": models[col].params["omega"],
            "alpha": models[col].params["alpha[1]"],
            "beta": models[col].params["beta[1]"],
            "nu": models[col].params["nu"],
            "last_var": cond_vol[col].tolist()[-1] ** 2,
        }
        for col in indices
    }

    # Divide simulations into chunks
    chunk_size = math.ceil(n_simulations / n_processes)
    chunks = [
        (i, min(chunk_size, n_simulations - i * chunk_size)) for i in range(n_processes)
    ]

    # Run simulations in parallel using ProcessPoolExecutor
    sim_returns = np.zeros((n_simulations, n_days, len(indices)))
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [
            executor.submit(
                monte_carlo_simulation_chunk,
                chunk_id,
                n_simulations_chunk,
                n_days,
                indices,
                params,
                Q_bar,
                a,
                b,
            )
            for chunk_id, n_simulations_chunk in chunks
        ]

        # Collect results
        start_idx = 0
        for future in futures:
            chunk_result = future.result()
            sim_returns[start_idx : start_idx + chunk_result.shape[0]] = chunk_result
            start_idx += chunk_result.shape[0]

    print("end time:", pd.Timestamp.now())
    print("total time:", pd.Timestamp.now() - start_time)

    # Calculate statistics
    print("\n模拟结果统计量：")
    for i, col in enumerate(indices):
        mean_sim = sim_returns[:, :, i].mean()
        std_sim = sim_returns[:, :, i].std()
        print(
            f"{col}: 模拟平均对数收益率 = {mean_sim:.6f} %, 模拟标准差 = {std_sim:.6f} %"
        )

    return sim_returns


# 执行分析
models, cond_vol, R_t, Q_bar, a, b = fit_dcc_garch(log_returns)
sim_returns = monte_carlo_dcc_garch(log_returns, models, cond_vol, Q_bar, a, b)

# print("\n模拟结果：")
# print(sim_returns.shape[0], "个模拟结果")
# print("每个模拟结果", sim_returns.shape[1], "天")
# print("每个模拟结果", sim_returns.shape[2], "个指数")
# 将模拟的对数收益率转换为实际收益率


# 根据收益率计算指数
def calculate_index(sim_returns, initial_prices):
    print(f"\nInitial prices: {initial_prices}")
    index_values = np.zeros_like(sim_returns)
    index_values[:, 0, :] = initial_prices  # Set the initial prices for all simulations

    # Calculate index values for each day based on simulated log returns
    for t in range(1, sim_returns.shape[1]):
        index_values[:, t, :] = index_values[:, t - 1, :] * np.exp(
            sim_returns[:, t, :] / 100
        )

    return index_values


# 从data中取用最后一个交易日的价格作为初始价格
initial_prices = data.iloc[-1, 1:].values
# 计算模拟的指数值
simulated_index_values = calculate_index(sim_returns, initial_prices)

# 将模拟的指数值每一列转换为dict, 并保存为JSON文件
simulated_index_dict = {
    col: simulated_index_values[:, :, i].tolist()
    for i, col in enumerate(log_returns.columns)
}


# print("\n模拟的指数值：")
# print(simulated_index_dict)

# 将模拟的指数值绘制成图表
for key, index_group in simulated_index_dict.items():
    # print("key:", key)
    # print("index_group", index_group)
    plt.figure(figsize=(10, 6))
    plt.axhline(index_group[0][0], color="black", lw=0.5, ls="--")
    for sim in index_group:
        # thiner lines
        plt.plot(sim, lw=0.3)  # Plot each simulation with transparency
    plt.title(f"{col} simulated index values")
    plt.xlabel("day")
    plt.ylabel("index value")
    plt.tight_layout()
    filename = "output-grok-monte-carlo-simulation-" + key + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as high-resolution PNG
    plt.close()
