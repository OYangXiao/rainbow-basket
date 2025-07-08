import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import openpyxl
import io

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

# Step 2: Compute correlation matrix
corr_matrix = log_returns.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Step 3: Cholesky decomposition
L = cholesky(corr_matrix, lower=True)

# Step 4: Monte Carlo simulation
n_simulations = 10000
n_days = 57
initial_prices = df.iloc[-1][['HS300', 'SZ50', 'ZZ500']].values
simulated_paths = { 'HS300': np.zeros((n_simulations, n_days + 1)),
                    'SZ50': np.zeros((n_simulations, n_days + 1)),
                    'ZZ500': np.zeros((n_simulations, n_days + 1)) }
simulated_paths['HS300'][:, 0] = initial_prices[0]
simulated_paths['SZ50'][:, 0] = initial_prices[1]
simulated_paths['ZZ500'][:, 0] = initial_prices[2]

for t in range(1, n_days + 1):
    # Generate uncorrelated standard normal random variables
    Z = np.random.normal(0, 1, (n_simulations, 3))
    # Apply Cholesky decomposition for correlation
    correlated_Z = Z @ L
    # Generate log returns with GARCH volatilities
    log_ret = np.zeros((n_simulations, 3))
    log_ret[:, 0] = correlated_Z[:, 0] * vol_hs300[t-1]
    log_ret[:, 1] = correlated_Z[:, 1] * vol_sz50[t-1]
    log_ret[:, 2] = correlated_Z[:, 2] * vol_zz500[t-1]
    # Ensure daily returns satisfy 0.9 < exp(log_ret) < 1.1
    daily_ret = np.exp(log_ret)
    log_ret = np.where(daily_ret > 1.1, np.log(1.1), log_ret)
    log_ret = np.where(daily_ret < 0.9, np.log(0.9), log_ret)
    # Update prices using GBM
    simulated_paths['HS300'][:, t] = simulated_paths['HS300'][:, t-1] * np.exp(log_ret[:, 0])
    simulated_paths['SZ50'][:, t] = simulated_paths['SZ50'][:, t-1] * np.exp(log_ret[:, 1])
    simulated_paths['ZZ500'][:, t] = simulated_paths['ZZ500'][:, t-1] * np.exp(log_ret[:, 2])

# Save simulation results to Excel
excel_data = {}
for index in ['HS300', 'SZ50', 'ZZ500']:
    for t in range(n_days + 1):
        excel_data[f'{index}_Day_{t}'] = simulated_paths[index][:, t]
sim_df = pd.DataFrame(excel_data)
sim_df.to_excel('monte_carlo_simulations.xlsx', index=False)

# Plot sample paths (first 100 for clarity)
plt.figure(figsize=(15, 10))
for index, color in zip(['HS300', 'SZ50', 'ZZ500'], ['blue', 'green', 'red']):
    for i in range(100):
        plt.plot(range(n_days + 1), simulated_paths[index][i], color=color, alpha=0.1)
    plt.plot([], label=index, color=color)
plt.title('Monte Carlo Simulated Price Paths')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.savefig('price_paths.png')
plt.close()

# Step 5: Identify knock-out events
knock_out = np.zeros(n_simulations, dtype=bool)
for t in range(n_days + 1):
    ratios = np.vstack([
        simulated_paths['HS300'][:, t] / initial_prices[0],
        simulated_paths['SZ50'][:, t] / initial_prices[1],
        simulated_paths['ZZ500'][:, t] / initial_prices[2]
    ])
    knock_out |= np.max(ratios, axis=0) >= 1.12
knock_out_count = np.sum(knock_out)
print(f"Number of knock-out events: {knock_out_count}")

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
final_yield = 1.2 * np.maximum(0, basket_performance - 1.02)
mean_final_yield = np.mean(final_yield)
print(f"Mean final yield for non-knock-out paths: {mean_final_yield:.4f}")