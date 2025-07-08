import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
import uuid
import os
from concurrent.futures import ProcessPoolExecutor
import time
from arch import arch_model

# Reading the data
data_start = time.time()
with open("databank.xlsx", "rb") as file:
    data_str = file.read()
df = pd.read_excel(io.BytesIO(data_str), engine='openpyxl')
df['Idxtrd01'] = pd.to_datetime(df['Idxtrd01'])
data_end = time.time()
print(f"[Time] Data loading: {data_end - data_start:.2f}s")

# Step 1: Calculating daily log returns
returns_start = time.time()
df['HS300_log_return'] = np.log(df['HS300'] / df['HS300'].shift(1))
df['SZ50_log_return'] = np.log(df['SZ50'] / df['SZ50'].shift(1))
df['ZZ500_log_return'] = np.log(df['ZZ500'] / df['ZZ500'].shift(1))

returns = df[['HS300_log_return', 'SZ50_log_return', 'ZZ500_log_return']].dropna()
returns_end = time.time()
print(f"[Time] Log returns calculation: {returns_end - returns_start:.2f}s")

# Step 2: Calculating correlation matrix
corr_start = time.time()
corr_matrix = returns.corr().to_numpy()
print("Correlation Matrix:")
print(corr_matrix)
corr_end = time.time()
print(f"[Time] Correlation matrix: {corr_end - corr_start:.2f}s")

# Step 3: Cholesky decomposition
cholesky_start = time.time()
L = np.linalg.cholesky(corr_matrix)
cholesky_end = time.time()
print(f"[Time] Cholesky decomposition: {cholesky_end - cholesky_start:.2f}s")

# Step 4: Monte Carlo simulation parameters
S0 = df[df['Idxtrd01'] == '2024-12-25'][['HS300', 'SZ50', 'ZZ500']].values.flatten()
# sigma = np.array([0.204654934, 0.18675667, 0.268128998]) / np.sqrt(242)  # Daily volatility
mu = 0.08  # Assuming zero drift
T = 157  # Number of days
N = 100000  # Number of simulations
dt = 1 / 242  # Time step in years
np.random.seed(42)

# Step 1.5: Use GARCH(1,1) to forecast volatility for each index

def garch_forecast(returns, horizon=157, window=252):
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
    forecasts = []
    for i in range(len(returns) - window, len(returns)):
        train_data = returns.iloc[i-window:i]
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=horizon)
        forecasts.append(np.sqrt(forecast.variance.values[-1, :]))
    return np.mean(forecasts, axis=0)  # Average forecast over rolling windows

sigma_hs300 = garch_forecast(returns['HS300_log_return'], horizon=T)
sigma_sz50 = garch_forecast(returns['SZ50_log_return'], horizon=T)
sigma_zz500 = garch_forecast(returns['ZZ500_log_return'], horizon=T)

sigma = np.array([sigma_hs300.mean(), sigma_sz50.mean(), sigma_zz500.mean()])

def simulate_paths_chunk(start, end, S0, L, sigma, mu, dt, T):
    chunk_N = end - start
    S_chunk = np.zeros((T + 1, chunk_N, 3))
    S_chunk[0, :, :] = S0
    for t in range(1, T + 1):
        Z = np.random.normal(0, 1, (chunk_N, 3))
        Z_correlated = Z @ L.T
        dW = Z_correlated * np.sqrt(dt)
        dS = mu * dt + sigma * dW
        r = np.exp(dS)
        r = np.clip(r, 0.9, 1.1)
        S_chunk[t, :, :] = S_chunk[t-1, :, :] * r
    return S_chunk

# Initializing arrays
S = np.zeros((T + 1, N, 3))
S[0, :, :] = S0
knock_out = np.zeros(N, dtype=bool)
basket_returns = np.zeros(N)

# Running Monte Carlo simulation (parallel)
monte_start = time.time()
num_workers = os.cpu_count() or 4
print(f"Using {num_workers} workers for parallel processing.")
chunk_size = N // num_workers
chunks = [(i*chunk_size, (i+1)*chunk_size if i < num_workers-1 else N) for i in range(num_workers)]

results = []
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(simulate_paths_chunk, start, end, S0, L, sigma, mu, dt, T) for start, end in chunks]
    for future in futures:
        results.append(future.result())
S = np.concatenate(results, axis=1)
monte_end = time.time()
print(f"[Time] Monte Carlo simulation: {monte_end - monte_start:.2f}s")

excel_start = time.time()
# Saving simulation results to Excel
excel_data = {}
for i, index in enumerate(['HS300', 'SZ50', 'ZZ500']):
    excel_data[index] = pd.DataFrame(S[:, :, i], columns=[f'Path_{j+1}' for j in range(N)])
    excel_data[index].index = [f'Day_{t}' for t in range(T + 1)]
excel_filename = f'simulated_paths_{uuid.uuid4().hex}.xlsx'
# with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#     for index, data in excel_data.items():
#         data.to_excel(writer, sheet_name=index)
excel_end = time.time()
print(f"[Time] Excel data preparation: {excel_end - excel_start:.2f}s")

# Step 5: Checking for knock-out (vectorized)
knockout_start = time.time()
# S.shape: (T+1, N, 3), S0.shape: (3,)
knockout_matrix = (S / S0 >= 1.12)  # shape: (T+1, N, 3)
knock_out = np.any(knockout_matrix, axis=(0, 2))  # shape: (N,)
knock_out_count = np.sum(knock_out)
print(f"Knock-out events: {knock_out_count} out of {N}")
knockout_end = time.time()
print(f"[Time] Knock-out check: {knockout_end - knockout_start:.2f}s")

# Step 6: Calculating basket returns for non-knock-out paths
basket_start = time.time()
for n in range(N):
    if not knock_out[n]:
        final_prices = S[-1, n, :]
        returns = final_prices / S0
        basket_returns[n] = np.max(returns)
basket_end = time.time()
print(f"[Time] Basket return calculation: {basket_end - basket_start:.2f}s")

# Step 7: Calculating final payoff
payoff_start = time.time()
payoffs = np.zeros(N)
for n in range(N):
    if not knock_out[n]:
        payoffs[n] = 1.2 * max(0, basket_returns[n] - 1.02)

mean_payoff = np.mean(payoffs[~knock_out]) if np.sum(~knock_out) > 0 else 0
print(f"Mean payoff for non-knock-out paths: {mean_payoff:.4f}")
payoff_end = time.time()
print(f"[Time] Payoff calculation: {payoff_end - payoff_start:.2f}s")

# Plotting sample paths
plot_start = time.time()
plt.figure(figsize=(15, 10))
for i, index in enumerate(['HS300', 'SZ50', 'ZZ500']):
    plt.subplot(3, 1, i+1)
    for n in range(min(100, N)):
        plt.plot(S[:, n, i], alpha=0.1)
    plt.title(f'{index} Simulated Paths')
    plt.xlabel('Days')
    plt.ylabel('Price')
plt.tight_layout()

# Saving plot to buffer
plt.savefig("simulated_paths.png", format='png')
plt.close()
plot_end = time.time()
print(f"[Time] Plotting and saving: {plot_end - plot_start:.2f}s")

# Generating markdown report
report = f"""
# Analysis Report for HS300, SZ50, and ZZ500 Indices

## Correlation Matrix
```
{corr_matrix}
```

## Monte Carlo Simulation Results
- **Number of Simulations**: {N}
- **Time Horizon**: {T} days
- **Trading Days per Year**: 242
- **Knock-out Events**: {knock_out_count} out of {N} ({knock_out_count/N*100:.2f}%)
- **Mean Payoff for Non-Knock-out Paths**: {mean_payoff:.4f}
- **Simulated Data**: Saved to {excel_filename}

"""
print(report)

# Providing download link for Excel file
print(f"Download simulated paths: {excel_filename}")