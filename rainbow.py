import pandas as pd
import numpy as np
from arch import arch_model
from scipy.linalg import cholesky
import openpyxl
import io
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

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

# Optimize GARCH forecasting with multithreading
def garch_forecast_multithreaded(returns, horizon=57, window=252):
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)

    def forecast_chunk(start, end):
        chunk_forecasts = []
        for i in range(start, end):
            train_data = returns.iloc[i-window:i]
            model_fit = model.fit(disp='off')
            forecast = model_fit.forecast(horizon=horizon)
            chunk_forecasts.append(np.sqrt(forecast.variance.values[-1, :]))
        return chunk_forecasts

    forecasts = []
    chunk_size = (len(returns) - window) // 8  # Divide into 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(forecast_chunk, i * chunk_size, (i + 1) * chunk_size)
            for i in range(8)
        ]
        for future in futures:
            forecasts.extend(future.result())

    return np.mean(forecasts, axis=0)  # Average forecast over rolling windows

vol_hs300 = garch_forecast_multithreaded(log_returns['HS300'])
vol_sz50 = garch_forecast_multithreaded(log_returns['SZ50'])
vol_zz500 = garch_forecast_multithreaded(log_returns['ZZ500'])

# Save GARCH forecast results to Excel
forecast_data = {
    'HS300': vol_hs300,
    'SZ50': vol_sz50,
    'ZZ500': vol_zz500
}
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_excel('garch_forecast_results.xlsx', index=False)

# Step 2: Compute correlation matrix
corr_matrix = log_returns.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Step 3: Cholesky decomposition
L = cholesky(corr_matrix, lower=True)

# Define Monte Carlo simulation function with multithreading

# Ensure the function returns all expected values
def run_monte_carlo_multithreaded(n_simulations, n_days, initial_prices, vol_hs300, vol_sz50, vol_zz500, L, scale_factor=1.0, plot_paths=False):
    vol_hs300_scaled = vol_hs300 * scale_factor
    vol_sz50_scaled = vol_sz50 * scale_factor
    vol_zz500_scaled = vol_zz500 * scale_factor

    simulated_paths = {
        'HS300': np.zeros((n_simulations, n_days + 1)),
        'SZ50': np.zeros((n_simulations, n_days + 1)),
        'ZZ500': np.zeros((n_simulations, n_days + 1))
    }
    simulated_paths['HS300'][:, 0] = initial_prices[0]
    simulated_paths['SZ50'][:, 0] = initial_prices[1]
    simulated_paths['ZZ500'][:, 0] = initial_prices[2]

    def simulate_chunk(start, end):
        for t in range(1, n_days + 1):
            Z = np.random.normal(0, 1, (end - start, 3))
            correlated_Z = Z @ L.T
            log_ret = np.zeros((end - start, 3))
            log_ret[:, 0] = correlated_Z[:, 0] * vol_hs300_scaled[t-1]
            log_ret[:, 1] = correlated_Z[:, 1] * vol_sz50_scaled[t-1]
            log_ret[:, 2] = correlated_Z[:, 2] * vol_zz500_scaled[t-1]
            daily_ret = np.exp(log_ret)
            log_ret = np.where(daily_ret > 1.1, np.log(1.1), log_ret)
            log_ret = np.where(daily_ret < 0.9, np.log(0.9), log_ret)
            simulated_paths['HS300'][start:end, t] = simulated_paths['HS300'][start:end, t-1] * np.exp(log_ret[:, 0])
            simulated_paths['SZ50'][start:end, t] = simulated_paths['SZ50'][start:end, t-1] * np.exp(log_ret[:, 1])
            simulated_paths['ZZ500'][start:end, t] = simulated_paths['ZZ500'][start:end, t-1] * np.exp(log_ret[:, 2])

    chunk_size = n_simulations // 8  # Divide into 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(simulate_chunk, i * chunk_size, (i + 1) * chunk_size)
            for i in range(8)
        ]
        for future in futures:
            future.result()

    # Plot paths if requested
    if plot_paths:
        fig, axes = plt.subplots(3, 1, figsize=(12,36))
        num_paths_to_plot = min(100000, n_simulations)

        # Plot HS300
        for i in range(num_paths_to_plot):
            axes[0].plot(simulated_paths['HS300'][i], color='blue', alpha=0.05)
        axes[0].axhline(y=initial_prices[0], color='black', linestyle='--', label='Initial Price')
        axes[0].axhline(y=initial_prices[0]*1.12, color='red', linestyle='--', label='Knock-out Threshold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Simulated Paths - HS300 (Sample of 100000 paths)')
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('Index Level')
        axes[0].legend()

        # Plot SZ50
        for i in range(num_paths_to_plot):
            axes[1].plot(simulated_paths['SZ50'][i], color='green', alpha=0.05)
        axes[1].axhline(y=initial_prices[1], color='black', linestyle='--', label='Initial Price')
        axes[1].axhline(y=initial_prices[1]*1.12, color='red', linestyle='--', label='Knock-out Threshold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Simulated Paths - SZ50 (Sample of 100000 paths)')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Index Level')
        axes[1].legend()

        # Plot ZZ500
        for i in range(num_paths_to_plot):
            axes[2].plot(simulated_paths['ZZ500'][i], color='orange', alpha=0.05)
        axes[2].axhline(y=initial_prices[2], color='black', linestyle='--', label='Initial Price')
        axes[2].axhline(y=initial_prices[2]*1.12, color='red', linestyle='--', label='Knock-out Threshold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Simulated Paths - ZZ500 (Sample of 100000 paths)')
        axes[2].set_xlabel('Days')
        axes[2].set_ylabel('Index Level')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig('simulated_paths.png', dpi=300)
        plt.close()
        print("Simulated paths visualization saved as 'simulated_paths.png'.")

    # Knock-out events
    knock_out = np.zeros(n_simulations, dtype=bool)
    for t in range(1, n_days + 1):  # Start from 1 as day 0 is initial
        ratios = np.vstack([
            simulated_paths['HS300'][:, t] / initial_prices[0],
            simulated_paths['SZ50'][:, t] / initial_prices[1],
            simulated_paths['ZZ500'][:, t] / initial_prices[2]
        ])
        knock_out |= np.max(ratios, axis=0) >= 1.12
    knock_out_rate = np.sum(knock_out) / n_simulations

    # Non-knock-out yields
    non_knock_out = ~knock_out
    terminal_yields = {
        'HS300': simulated_paths['HS300'][non_knock_out, -1] / initial_prices[0],
        'SZ50': simulated_paths['SZ50'][non_knock_out, -1] / initial_prices[1],
        'ZZ500': simulated_paths['ZZ500'][non_knock_out, -1] / initial_prices[2]
    }
    basket_performance = np.maximum.reduce([
        terminal_yields['HS300'],
        terminal_yields['SZ50'],
        terminal_yields['ZZ500']
    ])
    final_yield = 1.2 * np.maximum(0, basket_performance - 1.02)
    mean_final_yield = np.mean(final_yield) if len(final_yield) > 0 else 0

    # Overall average yield (including knock-out paths)
    avg_yield = (knock_out_rate * 0.023) + ((1 - knock_out_rate) * mean_final_yield)

    return knock_out_rate, mean_final_yield, avg_yield, final_yield

# Replace original functions with optimized versions
vol_hs300 = garch_forecast_multithreaded(log_returns['HS300'])
vol_sz50 = garch_forecast_multithreaded(log_returns['SZ50'])
vol_zz500 = garch_forecast_multithreaded(log_returns['ZZ500'])

# Save GARCH forecast results to Excel
forecast_data = {
    'HS300': vol_hs300,
    'SZ50': vol_sz50,
    'ZZ500': vol_zz500
}
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_excel('garch_forecast_results.xlsx', index=False)

# Step 2: Compute correlation matrix
corr_matrix = log_returns.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Step 3: Cholesky decomposition
L = cholesky(corr_matrix, lower=True)

# Parameters
n_days = 57
initial_prices = df.iloc[-1][['HS300', 'SZ50', 'ZZ500']].values
base_n_sim = 100000

# Store simulation results in a dictionary for reuse
simulation_results = {}

def get_or_run_simulation(n_sim, n_days, initial_prices, vol_hs300, vol_sz50, vol_zz500, L, scale_factor=1.0, plot_paths=False):
    """Run simulation if not cached, otherwise return cached results"""
    key = (n_sim, scale_factor)
    if key not in simulation_results:
        results = run_monte_carlo_multithreaded(n_sim, n_days, initial_prices, 
                                              vol_hs300, vol_sz50, vol_zz500, L, 
                                              scale_factor=scale_factor, 
                                              plot_paths=plot_paths)
        simulation_results[key] = results
    return simulation_results[key]

# Step 4: Base simulation
print("Running base simulation...")
base_results = get_or_run_simulation(base_n_sim, n_days, initial_prices, 
                                   vol_hs300, vol_sz50, vol_zz500, L, 
                                   plot_paths=True)
base_knock_out_rate, base_mean_final_yield, base_avg_yield, base_final_yield = base_results

print(f"Base Knock-out rate: {base_knock_out_rate*100:.2f}%")
print(f"Base Average final yield (non-knock-out): {base_mean_final_yield*100:.2f}%")
print(f"Base Overall average yield: {base_avg_yield*100:.2f}%")

# Step 8: Result Analysis - Vega Sensitivity
print("Calculating Vega sensitivity...")
scale_factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
vega_results = []
base_sigma = np.mean([np.mean(vol_hs300), np.mean(vol_sz50), np.mean(vol_zz500)])
for k in scale_factors:
    _, _, avg_yield, _ = get_or_run_simulation(base_n_sim, n_days, initial_prices, 
                                             vol_hs300, vol_sz50, vol_zz500, L, k)
    vega = (avg_yield - base_avg_yield) / ((k - 1) * base_sigma) if k != 1 else 0
    vega_results.append({'Scale': k, 'Avg Yield': avg_yield, 'Vega': vega})

vega_df = pd.DataFrame(vega_results)
vega_df.to_excel('vega_sensitivity.xlsx', index=False)
print("Vega Sensitivity Results:")
print(vega_df)

# Plot Vega
plt.figure()
plt.plot(vega_df['Scale'], vega_df['Vega'], marker='o')
plt.title('Basket Vega Sensitivity')
plt.xlabel('Volatility Scale Factor')
plt.ylabel('Vega')
plt.savefig('vega_plot.png')
plt.close()

# Step 9: Result Analysis - Convergence
print("Analyzing convergence...")
sim_counts = [10000, 50000, 100000, 200000, 500000, 1000000]
conv_results = []

for n in sim_counts:
    # Run simulation if needed
    knock_out_rate, mean_final_yield, avg_yield, final_yield = get_or_run_simulation(
        n, n_days, initial_prices, vol_hs300, vol_sz50, vol_zz500, L
    )
    
    # Calculate standard errors
    std_err_ko = np.sqrt(knock_out_rate * (1 - knock_out_rate) / n)
    
    # For yield std error, use batching of existing results
    batch_size = max(n // 100, 1)
    batch_yields = []
    for i in range(0, len(final_yield), batch_size):
        batch_end = min(i + batch_size, len(final_yield))
        batch_mean = np.mean(final_yield[i:batch_end]) if len(final_yield[i:batch_end]) > 0 else 0
        batch_yields.append(batch_mean)
    
    std_err_yield = np.std(batch_yields) / np.sqrt(len(batch_yields)) if batch_yields else 0
    conv_results.append({
        'N': n, 
        'KO Rate': knock_out_rate, 
        'Avg Yield': avg_yield, 
        'Std Err KO': std_err_ko, 
        'Std Err Yield': std_err_yield
    })

conv_df = pd.DataFrame(conv_results)
conv_df.to_excel('convergence_analysis.xlsx', index=False)
print("Convergence Results:")
print(conv_df)

# Format knock-out rate and average yield to 4 decimal places
conv_df['KO Rate'] = conv_df['KO Rate'].round(4)
conv_df['Avg Yield'] = conv_df['Avg Yield'].round(4)

# Plot Convergence
fig, ax1 = plt.subplots()
ax1.plot(conv_df['N'], conv_df['KO Rate'], 'b-', label='KO Rate')
ax1.errorbar(conv_df['N'], conv_df['KO Rate'], yerr=conv_df['Std Err KO'], fmt='b-o')
ax1.set_xlabel('Simulations')
ax1.set_ylabel('KO Rate')
ax2 = ax1.twinx()
ax2.plot(conv_df['N'], conv_df['Avg Yield'], 'r-', label='Avg Yield')
ax2.errorbar(conv_df['N'], conv_df['Avg Yield'], yerr=conv_df['Std Err Yield'], fmt='r-o')
ax2.set_ylabel('Avg Yield')
fig.legend()
plt.title('Convergence Analysis (LLN Assessment)')
plt.savefig('convergence_plot.png')
plt.close()

# Step 10: Result Analysis - Risk Metrics (based on base non-knock-out final yields)
if len(base_final_yield) > 0:
    sorted_yields = np.sort(base_final_yield)
    var_95 = -np.percentile(sorted_yields, 5)  # Max loss (negative for loss)
    es_95 = -np.mean(sorted_yields[sorted_yields <= np.percentile(sorted_yields, 5)])
    print(f"VaR (95%): {var_95*100:.2f}%")
    print(f"ES (95%): {es_95*100:.2f}%")
else:
    print("No non-knock-out paths for risk analysis.")