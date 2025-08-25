import pandas as pd
import numpy as np
from arch import arch_model
from scipy.linalg import cholesky
import openpyxl
import io
import matplotlib.pyplot as plt

# 产品参数
tau = 0.249
mu = 0.03  # 无风险利率（风险中性漂移率）
knockout_level = 1.12
strike_level = 1.02
participation = 1.2
knockout_rate = 0.023
fixed_rate = 0.0

# Step 1: Load and preprocess data
with open("databank.xlsx", "rb") as file:
    data_str = file.read()
df = pd.read_excel(io.BytesIO(data_str), engine='openpyxl')
df['Idxtrd01'] = pd.to_datetime(df['Idxtrd01'])
df.set_index('Idxtrd01', inplace=True)

# Calculate daily log returns
log_returns = np.log(df / df.shift(1)).dropna()

# GARCH(1,1) model for volatility forecasting
def garch_forecast(returns, horizon=57, window=252):
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
    forecasts = []
    for i in range(len(returns) - window, len(returns)):
        train_data = returns.iloc[i-window:i]
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=horizon)
        forecasts.append(np.sqrt(forecast.variance.values[-1, :]))
    return np.mean(forecasts, axis=0)  # Average forecast over rolling windows

vol_hs300 = garch_forecast(log_returns['HS300'])
vol_sz50 = garch_forecast(log_returns['SZ50'])
vol_zz500 = garch_forecast(log_returns['ZZ500'])

# Save GARCH forecast results to Excel
forecast_data = {
    'HS300': vol_hs300,
    'SZ50': vol_sz50,
    'ZZ500': vol_zz500
}
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_excel('garch_forecast_results_2.xlsx', index=False)

# Step 2: Compute correlation matrix
corr_matrix = log_returns.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Step 3: Cholesky decomposition
L = cholesky(corr_matrix, lower=True)

# 定义模拟函数，便于敏感性和收敛分析
def run_simulation(n_simulations, multiplier=1.0):
    n_days = 57
    initial_prices = df.iloc[-1][['HS300', 'SZ50', 'ZZ500']].values
    simulated_paths = {'HS300': np.zeros((n_simulations, n_days + 1)),
                       'SZ50': np.zeros((n_simulations, n_days + 1)),
                       'ZZ500': np.zeros((n_simulations, n_days + 1))}
    simulated_paths['HS300'][:, 0] = initial_prices[0]
    simulated_paths['SZ50'][:, 0] = initial_prices[1]
    simulated_paths['ZZ500'][:, 0] = initial_prices[2]

    # 调整波动率
    adj_vol_hs300 = vol_hs300 * multiplier
    adj_vol_sz50 = vol_sz50 * multiplier
    adj_vol_zz500 = vol_zz500 * multiplier

    dt = tau / n_days  # 年化步长
    drift = mu * dt  # 风险中性漂移 (假设无股息)

    for t in range(1, n_days + 1):
        Z = np.random.normal(0, 1, (n_simulations, 3))
        correlated_Z = Z @ L
        log_ret = np.zeros((n_simulations, 3))
        log_ret[:, 0] = drift + correlated_Z[:, 0] * adj_vol_hs300[t-1]
        log_ret[:, 1] = drift + correlated_Z[:, 1] * adj_vol_sz50[t-1]
        log_ret[:, 2] = drift + correlated_Z[:, 2] * adj_vol_zz500[t-1]
        daily_ret = np.exp(log_ret)
        log_ret = np.where(daily_ret > 1.1, np.log(1.1), log_ret)
        log_ret = np.where(daily_ret < 0.9, np.log(0.9), log_ret)
        simulated_paths['HS300'][:, t] = simulated_paths['HS300'][:, t-1] * np.exp(log_ret[:, 0])
        simulated_paths['SZ50'][:, t] = simulated_paths['SZ50'][:, t-1] * np.exp(log_ret[:, 1])
        simulated_paths['ZZ500'][:, t] = simulated_paths['ZZ500'][:, t-1] * np.exp(log_ret[:, 2])

    # Step 5: Identify knock-out events
    knock_out = np.zeros(n_simulations, dtype=bool)
    for t in range(n_days + 1):
        ratios = np.vstack([
            simulated_paths['HS300'][:, t] / initial_prices[0],
            simulated_paths['SZ50'][:, t] / initial_prices[1],
            simulated_paths['ZZ500'][:, t] / initial_prices[2]
        ])
        knock_out |= np.max(ratios, axis=0) >= knockout_level
    knock_out_count = np.sum(knock_out)
    knock_out_rate = knock_out_count / n_simulations * 100

    # Step 6: Compute terminal yields for non-knock-out paths
    non_knock_out = ~knock_out
    terminal_yields = {
        'HS300': simulated_paths['HS300'][non_knock_out, -1] / initial_prices[0],
        'SZ50': simulated_paths['SZ50'][non_knock_out, -1] / initial_prices[1],
        'ZZ500': simulated_paths['ZZ500'][non_knock_out, -1] / initial_prices[2]
    }
    basket_performance = np.max([
        terminal_yields['HS300'],
        terminal_yields['SZ50'],
        terminal_yields['ZZ500']
    ], axis=0)

    # Step 7: Calculate final yield for non-knock-out paths
    final_rate = fixed_rate + participation * np.maximum(0, basket_performance - strike_level)

    # Step 8: Calculate payoffs
    payoffs = np.zeros(n_simulations)
    payoffs[knock_out] = 1 + knockout_rate * tau
    payoffs[non_knock_out] = 1 + final_rate * tau

    theory_price = np.mean(payoffs)
    theory_std = np.std(payoffs)
    theory_se = theory_std / np.sqrt(n_simulations)
    avg_annual_yield = (theory_price - 1 ) / tau

    ko_se = np.sqrt(knock_out_rate / 100 * (1 - knock_out_rate / 100) / n_simulations)

    return {
        'theory_price': theory_price,
        'ko_prob': knock_out_rate,
        'avg_yield': avg_annual_yield,
        'theory_se': theory_se,
        'ko_se': ko_se,
        'payoffs': payoffs  # 用于进一步分析，如果需要
    }

# 5.3.1 波动率敏感性分析
adjustments = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]
sensitivity_results = []
for adj in adjustments:
    mult = 1 + adj
    res = run_simulation(n_simulations=100000, multiplier=mult)
    sensitivity_results.append((adj * 100, res['theory_price'], res['ko_prob'], res['avg_yield'] * 100))
    print(f"Vol adj {adj*100}%: Price {res['theory_price']:.4f}, KO {res['ko_prob']:.4f}%, Yield {res['avg_yield']*100:.4f}%")

# 5.3.2 收敛性分析
n_list = [10000, 50000, 100000, 200000, 500000, 1000000]
convergence_results = []
ko_means = []
price_means = []
ko_ses = []
price_ses = []
for n in n_list:
    res = run_simulation(n_simulations=n, multiplier=1.0)
    convergence_results.append((n, res['ko_prob'], res['ko_se'], res['theory_price'], res['theory_se']))
    ko_means.append(res['ko_prob'])
    price_means.append(res['theory_price'])
    ko_ses.append(res['ko_se'])
    price_ses.append(res['theory_se'])
    print(f"n={n}: KO {res['ko_prob']:.1f}% (SE {res['ko_se']:.4f}), Price {res['theory_price']:.4f} (SE {res['theory_se']:.5f})")

# 绘制收敛图
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(n_list, price_means, 'b-', label='Average Payoff')
ax1.fill_between(n_list, np.array(price_means) - 1.96 * np.array(price_ses), np.array(price_means) + 1.96 * np.array(price_ses), color='b', alpha=0.2)
ax1.set_xlabel('Number of Simulations')
ax1.set_ylabel('Average Payoff', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(n_list, ko_means, 'r-', label='Knockout Probability (%)')
ax2.fill_between(n_list, np.array(ko_means) - 1.96 * np.array(ko_ses) * 100, np.array(ko_means) + 1.96 * np.array(ko_ses) * 100, color='r', alpha=0.2)  # SE scaled to %
ax2.set_ylabel('Knockout Probability (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.legend(loc='upper right')
plt.title('Convergence Analysis')
plt.savefig('convergence_plot_2.png')
plt.close()

# (原始代码中的绘图和保存部分可保留或移除，根据需要)