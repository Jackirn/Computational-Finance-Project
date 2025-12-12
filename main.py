import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Reproducibility (optional but recommended)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# Paths
csv_path_prices = "csv/asset_prices.csv"
csv_path_caps   = "csv/capitalization_weights.csv"
csv_path_prices_os = "csv/asset_prices_out_of_sample.csv"

os.makedirs("Plots", exist_ok=True)


# Loading Data
df_prices_is = pd.read_csv(csv_path_prices, index_col=0, parse_dates=True).sort_index()
df_prices_os = pd.read_csv(csv_path_prices_os, index_col=0, parse_dates=True).sort_index()

# Concatenate IS + OOS
df_prices = pd.concat([df_prices_is, df_prices_os], axis=0).sort_index()

cap = pd.read_csv(csv_path_caps).set_index("Asset")

# Market weights aligned to columns
market_w_array = cap["MarketWeight"].loc[df_prices.columns].values
market_weights = torch.tensor(market_w_array, dtype=torch.float32)  # shape: (n_assets,)

prices = df_prices.values
T, n_assets = prices.shape

# Daily returns
df_returns = df_prices.pct_change().dropna()
returns = df_returns.values
ret_dates = df_returns.index


# Parameters
window_len  = 600
delta_days  = 252
batch_size  = 64
epochs      = 100
lr          = 1e-3
hidden_dim  = 64


# Build features X and targets Y (252d forward returns)
X_list = []
Y_list = []
valid_dates = []

for i in range(window_len - 1, len(df_returns)):
    anchor_date = ret_dates[i]
    t_price = df_prices.index.get_loc(anchor_date)
    t_future = t_price + delta_days

    window_ret = returns[i - window_len + 1: i + 1, :]  # (window_len, n_assets)
    x_t = window_ret.reshape(-1)  # flatten

    # target (252d forward return)
    if t_future < T:
        P_t = prices[t_price, :]
        P_future = prices[t_future, :]
        y_t = (P_future / P_t) - 1.0
    else:
        y_t = np.zeros(n_assets)  # dummy for tail (pure OOS end)

    X_list.append(x_t)
    Y_list.append(y_t)
    valid_dates.append(anchor_date)

X = np.vstack(X_list)
Y = np.vstack(Y_list)
valid_dates = pd.to_datetime(valid_dates)

# Temporal train-test split
split_date = pd.Timestamp("2023-01-01")
train_mask = valid_dates < split_date
test_mask  = valid_dates >= split_date

X_train = X[train_mask]
Y_train = Y[train_mask]

# remove samples with dummy all-zero targets (no future available)
keep = np.where(np.abs(Y_train).sum(axis=1) > 0)[0]
X_train = X_train[keep]
Y_train = Y_train[keep]

X_test = X[test_mask]
test_dates = valid_dates[test_mask]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# PyTorch Dataset / Loader
class PortfolioDataset(Dataset):
    def __init__(self, X_, Y_):
        self.X = torch.tensor(X_, dtype=torch.float32)
        self.Y = torch.tensor(Y_, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(
    PortfolioDataset(X_train, Y_train),
    batch_size=batch_size,
    shuffle=True
)

# Model
class PortfolioNet(nn.Module):
    def __init__(self, input_dim, n_assets, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.net(x))

def risk_adjusted_loss(weights, future_returns, lambda_conc=1e-3, lambda_bench=1e-3):
    """
    weights: (B, n_assets) long-only, sum=1
    future_returns: (B, n_assets) 252d forward returns
    """
    port_ret = (weights * future_returns).sum(dim=1)
    bench_ret = (market_weights * future_returns).sum(dim=1)
    active_ret = port_ret - bench_ret

    info_ratio = active_ret.mean() / (active_ret.std() + 1e-6)

    loss = -info_ratio
    loss += lambda_conc * torch.mean(torch.sum(weights ** 2, dim=1))
    loss += lambda_bench * torch.mean(torch.sum((weights - market_weights) ** 2, dim=1))
    return loss

model = PortfolioNet(X.shape[1], n_assets, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
print("Inizio Training...")
model.train()
for ep in range(epochs):
    epoch_loss = []
    for bx, by in train_loader:
        optimizer.zero_grad()
        w = model(bx)
        loss = risk_adjusted_loss(w, by)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    if (ep + 1) % 10 == 0:
        print(f"Epoch {ep+1}/{epochs} | Loss: {np.mean(epoch_loss):.4f}")

# Backtest: compute daily returns using next-day returns + weights predicted on each test date
model.eval()
test_weights = []
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_x = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
        w = model(batch_x).numpy()
        test_weights.append(w)

test_weights = np.vstack(test_weights)  # shape (len(X_test), n_assets)

strategy_returns = []
benchmark_returns = []
dates_backtest = []

for i in range(len(test_dates) - 1):
    date_t = test_dates[i]

    # locate the same date in df_returns
    try:
        idx_in_ret = df_returns.index.get_loc(date_t)
    except KeyError:
        continue

    if idx_in_ret + 1 >= len(df_returns):
        continue

    next_day_ret = returns[idx_in_ret + 1]
    next_date = df_returns.index[idx_in_ret + 1]

    w_t = test_weights[i]
    r_p = float(np.dot(w_t, next_day_ret))
    r_b = float(np.dot(market_w_array, next_day_ret))

    strategy_returns.append(r_p)
    benchmark_returns.append(r_b)
    dates_backtest.append(next_date)

strategy_returns = np.array(strategy_returns)
benchmark_returns = np.array(benchmark_returns)
dates_backtest = pd.to_datetime(dates_backtest)

def get_drawdown(cum_ret):
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    return drawdown, drawdown.min()

def perf_metrics(r):
    ann_factor = 252
    r = np.asarray(r)
    mean_ret = r.mean() * ann_factor
    vol_ret = r.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / vol_ret if vol_ret > 0 else 0
    cum = (1 + r).cumprod()
    dd, max_dd = get_drawdown(cum)
    calmar = mean_ret / abs(max_dd) if max_dd < 0 else 0
    return mean_ret, vol_ret, sharpe, max_dd, calmar, cum, dd

# 2023-2024 report 
mask_2324 = (dates_backtest >= pd.Timestamp("2023-01-01")) & (dates_backtest <= pd.Timestamp("2024-12-31"))
ret_2324 = strategy_returns[mask_2324]
bench_2324 = benchmark_returns[mask_2324]
dates_2324 = dates_backtest[mask_2324]

if len(ret_2324) > 0:
    m, v, sh, maxdd, cal, cum, dd = perf_metrics(ret_2324)
    print("\n=== Performance Report (2023-2024) ===")
    print(f"Annualized Return:     {m:.2%}")
    print(f"Annualized Volatility: {v:.2%}")
    print(f"Sharpe Ratio:          {sh:.2f}")
    print(f"Max Drawdown:          {maxdd:.2%}")
    print(f"Calmar Ratio:          {cal:.2f}")

    # Plot 2023-2024
    _, _, _, _, _, cum_bench_2324, _ = perf_metrics(bench_2324)
    dd_2324, _ = get_drawdown(cum)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(dates_2324, cum, label="Neural Net Strategy", linewidth=1.5)
    plt.plot(dates_2324, cum_bench_2324, label="Market Benchmark", linestyle="--", linewidth=1.5)
    plt.title("Cumulative Returns (Out-of-Sample 2023-2024)")
    plt.ylabel("Growth Factor")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.fill_between(dates_2324, dd_2324, 0, alpha=0.3, label="Strategy Drawdown")
    plt.plot(dates_2324, dd_2324, linewidth=1)
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

# 2025 report (required by project)
mask_2025 = (dates_backtest >= pd.Timestamp("2025-01-01")) & (dates_backtest <= pd.Timestamp("2025-12-31"))
ret_2025 = strategy_returns[mask_2025]
bench_2025 = benchmark_returns[mask_2025]
dates_2025 = dates_backtest[mask_2025]

if len(ret_2025) == 0:
    print("\n[Warning] Nessun dato di backtest nel 2025 (controlla le date del CSV OOS).")
else:
    mean_ret_2025, vol_ret_2025, sharpe_2025, max_dd_2025, calmar_2025, cum_strat_2025, dd_2025 = perf_metrics(ret_2025)
    _, _, _, _, _, cum_bench_2025, _ = perf_metrics(bench_2025)

    print(f"\n=== Performance Report (2025 only) ===")
    print(f"Annualized Return:     {mean_ret_2025:.2%}")
    print(f"Annualized Volatility: {vol_ret_2025:.2%}")
    print(f"Sharpe Ratio:          {sharpe_2025:.2f}")
    print(f"Max Drawdown:          {max_dd_2025:.2%}")
    print(f"Calmar Ratio:          {calmar_2025:.2f}")

    # Plot equity curve 2025 
    plt.figure(figsize=(8, 5))
    plt.plot(dates_2025, cum_strat_2025, label="NN Strategy 2025")
    plt.plot(dates_2025, cum_bench_2025, label="Market Benchmark 2025", linestyle="--")
    plt.title("Cumulative Returns - Out-of-Sample 2025")
    plt.ylabel("Growth Factor")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot drawdown 2025 
    plt.figure(figsize=(8, 4))
    plt.fill_between(dates_2025, dd_2025, 0, alpha=0.3, label="Strategy Drawdown 2025")
    plt.plot(dates_2025, dd_2025, linewidth=1)
    plt.title("Drawdown Profile - 2025")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()