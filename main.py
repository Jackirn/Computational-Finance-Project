import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Loading Data

csv_path_prices = "csv/asset_prices.csv"          
csv_path_caps   = "csv/capitalization_weights.csv" 

df_prices = pd.read_csv(csv_path_prices, index_col=0, parse_dates=True)
df_prices = df_prices.sort_index()

cap = pd.read_csv(csv_path_caps)
cap = cap.set_index("Asset")

# Market Weights Tensor
market_w_array = cap["MarketWeight"].loc[df_prices.columns].values
market_weights = torch.tensor(market_w_array, dtype=torch.float32)

prices = df_prices.values
dates  = df_prices.index
T, n_assets = prices.shape

# Daily Returns Calculation
df_returns = df_prices.pct_change().dropna()
returns = df_returns.values
ret_dates = df_returns.index


# Parameters and Preprocessing

window_len  = 600
delta_days  = 252
batch_size  = 64
epochs      = 100  # Epochs for training
lr          = 1e-3
hidden_dim  = 64

X_list = []
Rdelta_list = []
Valid_Dates = []

# Rolling Window Feature and Target Creation
for i in range(window_len - 1, len(df_returns)):
    anchor_date = ret_dates[i]
    t_price = df_prices.index.get_loc(anchor_date)
    
    t_future = t_price + delta_days
    
    window_ret = returns[i - window_len + 1 : i + 1, :]
    x_t = window_ret.reshape(-1)
    
    X_list.append(x_t)
    Valid_Dates.append(anchor_date)
    
    # if future date within bounds, calculate target returns
    if t_future < T:
        P_t = prices[t_price, :]
        P_future = prices[t_future, :]
        R_delta = (P_future / P_t) - 1.0
    else:
        # dummy target for out-of-sample
        R_delta = np.zeros(n_assets) 
        
    Rdelta_list.append(R_delta)

X = np.vstack(X_list)
Y = np.vstack(Rdelta_list)
Valid_Dates = pd.to_datetime(Valid_Dates)


# temporal train-test split

split_date = pd.Timestamp("2023-01-01")

train_mask = Valid_Dates < split_date
test_mask  = Valid_Dates >= split_date

X_train = X[train_mask]
Y_train = Y[train_mask]

# filtering samples with all-zero targets
valid_y_idx = np.where(np.abs(Y_train).sum(axis=1) > 0)[0]
X_train = X_train[valid_y_idx]
Y_train = Y_train[valid_y_idx]

X_test = X[test_mask]
test_dates = Valid_Dates[test_mask]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Model and Training

class PortfolioDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

train_loader = DataLoader(PortfolioDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

class PortfolioNet(nn.Module):
    def __init__(self, input_dim, n_assets, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.net(x))

def risk_adjusted_loss(weights, future_returns, lambda_conc=1e-3, lambda_bench=1e-3):
    # loss based on Information Ratio
    port_ret = (weights * future_returns).sum(dim=1)
    bench_ret = (market_weights * future_returns).sum(dim=1)
    active_ret = port_ret - bench_ret
    
    # Information Ratio batch-wise
    info_ratio = active_ret.mean() / (active_ret.std() + 1e-6)
    
    loss = -info_ratio
    loss += lambda_conc * torch.mean(torch.sum(weights ** 2, dim=1))
    loss += lambda_bench * torch.mean(torch.sum((weights - market_weights) ** 2, dim=1))
    return loss

model = PortfolioNet(X.shape[1], n_assets, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Inizio Training...")
model.train()
for ep in range(epochs):
    epoch_loss = []
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = risk_adjusted_loss(model(bx), by)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    if (ep+1) % 10 == 0:
        print(f"Epoch {ep+1}/{epochs} | Loss: {np.mean(epoch_loss):.4f}")

# Backtest and Evaluation

model.eval()
test_weights = []
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_x = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
        w = model(batch_x)
        test_weights.append(w.numpy())

test_weights = np.vstack(test_weights)

strategy_returns = []
benchmark_returns = []
dates_backtest = []

# Daily Returns Calculation for Backtest
for i in range(len(test_dates) - 1):
    date_t = test_dates[i]
    try:
        idx_in_ret = df_returns.index.get_loc(date_t)
        if idx_in_ret + 1 < len(df_returns):
            next_day_ret = returns[idx_in_ret + 1] 
            next_date = df_returns.index[idx_in_ret + 1]
            
            w_t = test_weights[i]
            
            r_p = np.dot(w_t, next_day_ret)
            r_b = np.dot(market_w_array, next_day_ret)
            
            strategy_returns.append(r_p)
            benchmark_returns.append(r_b)
            dates_backtest.append(next_date)
    except KeyError:
        continue

strategy_returns = np.array(strategy_returns)
benchmark_returns = np.array(benchmark_returns)
dates_backtest = pd.to_datetime(dates_backtest)

# Cumulative Returns
cum_ret_strat = (1 + strategy_returns).cumprod()
cum_ret_bench = (1 + benchmark_returns).cumprod()

# Drawdown Function
def get_drawdown(cum_ret):
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    return drawdown, drawdown.min()

dd_strat, max_dd_strat = get_drawdown(cum_ret_strat)
dd_bench, max_dd_bench = get_drawdown(cum_ret_bench)

# Metrics 
ann_factor = 252
mean_ret = strategy_returns.mean() * ann_factor
vol_ret = strategy_returns.std() * np.sqrt(ann_factor)
sharpe = mean_ret / vol_ret if vol_ret > 0 else 0
calmar = mean_ret / abs(max_dd_strat) if max_dd_strat < 0 else 0

print(f"\n=== Performance Report (2023-2024) ===")
print(f"Annualized Return:     {mean_ret:.2%}")
print(f"Annualized Volatility: {vol_ret:.2%}")
print(f"Sharpe Ratio:          {sharpe:.2f}")
print(f"Max Drawdown:          {max_dd_strat:.2%}")
print(f"Calmar Ratio:          {calmar:.2f}")

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(dates_backtest, cum_ret_strat, label='Neural Net Strategy', color='blue')
plt.plot(dates_backtest, cum_ret_bench, label='Market Benchmark', color='gray', linestyle='--')
plt.title('Cumulative Returns (Out-of-Sample 2023-2024)')
plt.ylabel('Growth Factor')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.fill_between(dates_backtest, dd_strat, 0, color='red', alpha=0.3, label='Strategy Drawdown')
plt.plot(dates_backtest, dd_strat, color='red', linewidth=1)
plt.title('Drawdown Profile')
plt.ylabel('Drawdown %')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Plots/performance_plot.png')
plt.show()