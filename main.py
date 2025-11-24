import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================
# 1. Hyperparametri principali
# ==============================
csv_path_prices = "csv/asset_prices.csv"          # prezzi asset
csv_path_caps   = "csv/capitalization_weights.csv"  # pesi di capitalizzazione

window_len  = 600          # L: giorni di storia usati come input
delta_days  = 252          # Δ: ~1 anno di borsa
batch_size  = 64
epochs      = 300
lr          = 1e-3
hidden_dim  = 64           # dimensione hidden layer rete

lambda_conc_train  = 1e-3  # penalità concentrazione nella loss NN
lambda_bench_train = 1e-3  # penalità distanza da market-cap nella loss NN

# ==============================
# 2. Caricamento dati e preprocessing
# ==============================
# CSV prezzi del tipo:
# ,Asset1,Asset2,...,AssetN
# 2001-10-09,100.0,100.0,...
# 2001-10-10,103.58,101.68,...

df_prices = pd.read_csv(csv_path_prices, index_col=0, parse_dates=True)
df_prices = df_prices.sort_index()   # per sicurezza

# CSV capitalizzazione:
# Asset,MacroGroup,MarketWeight
cap = pd.read_csv(csv_path_caps)
cap = cap.set_index("Asset")

# Allineiamo i pesi di market cap all'ordine delle colonne di df_prices
market_w_array = cap["MarketWeight"].loc[df_prices.columns].values  # shape (n_assets,)

# Prezzi come numpy
prices = df_prices.values                  # shape (T, n_assets)
dates  = df_prices.index
T, n_assets = prices.shape

# Rendimenti giornalieri: li usiamo come feature (storia)
df_returns = df_prices.pct_change().dropna()
returns = df_returns.values               # shape (T-1, n_assets)
ret_dates = df_returns.index

# ==============================
# 3. Costruzione dataset (X_t, R_delta_t)
#    X_t: finestra di history sui rendimenti (L giorni)
#    R_delta_t: rendimento cumulato a 1 anno per ogni asset
# ==============================

X_list = []
Rdelta_list = []

# Scorriamo gli indici di df_returns dove possiamo:
# - guardare indietro "window_len" giorni di returns
# - guardare avanti "delta_days" giorni di prezzi
for i in range(window_len - 1, len(df_returns)):
    anchor_date = ret_dates[i]  # questa è la data t1 (fine della finestra di storia)

    # posizione di anchor_date in df_prices
    t_price = df_prices.index.get_loc(anchor_date)

    # giorno futuro t1 + Δ
    t_future = t_price + delta_days
    if t_future >= T:
        # non ho abbastanza dati futuri, mi fermo
        break

    # 3.1. finestra di rendimenti: [i - window_len + 1, ..., i]
    window_ret = returns[i - window_len + 1 : i + 1, :]   # shape (window_len, n_assets)

    # feature = finestra appiattita
    x_t = window_ret.reshape(-1)                          # shape (window_len * n_assets,)

    # 3.2. rendimento a orizzonte Δ (t -> t+Δ) per ogni asset
    P_t      = prices[t_price, :]                         # prezzi alla data t
    P_future = prices[t_future, :]                        # prezzi alla data t+Δ
    R_delta  = (P_future / P_t) - 1.0                     # shape (n_assets,)

    X_list.append(x_t)
    Rdelta_list.append(R_delta)

X = np.vstack(X_list)             # shape (N, window_len * n_assets)
R_delta = np.vstack(Rdelta_list)  # shape (N, n_assets)
N, input_dim = X.shape

print("Numero di campioni:", N)
print("Input_dim:", input_dim, "n_assets:", n_assets)

# ==============================
# 4. Train / Test split per tempo
# ==============================
split_idx = int(0.8 * N)

X_train = X[:split_idx]
Y_train = R_delta[:split_idx]

X_test  = X[split_idx:]
Y_test  = R_delta[split_idx:]

# ==============================
# 5. Dataset & DataLoader PyTorch
# ==============================
class PortfolioDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = PortfolioDataset(X_train, Y_train)
test_dataset  = PortfolioDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==============================
# 6. Modello neurale in PyTorch
#    Input: feature X_t
#    Output: pesi di portafoglio w_t (softmax)
# ==============================
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
        logits = self.net(x)           # (batch_size, n_assets)
        w = self.softmax(logits)       # pesi long-only, somma=1
        return w

model = PortfolioNet(input_dim=input_dim, n_assets=n_assets, hidden_dim=hidden_dim)

# ==============================
# 7. Funzioni di loss e optimizer
# ==============================
market_weights = torch.tensor(market_w_array, dtype=torch.float32)

def risk_adjusted_loss(weights, future_returns,
                       lambda_conc=0.0, lambda_bench=0.0, eps=1e-6):
    """
    weights: (batch, n_assets)  -> w_t
    future_returns: (batch, n_assets) -> R^{(Δ)}_t
    Valuta una proxy di information ratio batch-wise vs market-cap.
    """

    # Ritorno del tuo portafoglio
    port_ret = (weights * future_returns).sum(dim=1)           # (batch,)

    # Ritorno del benchmark market-cap weighted
    bench_ret = (market_weights * future_returns).sum(dim=1)   # (batch,)

    active_ret = port_ret - bench_ret                          # (batch,)

    mean_active = active_ret.mean()
    std_active  = active_ret.std(unbiased=False) + eps

    info_ratio = mean_active / std_active

    # Loss base = - information ratio
    loss = -info_ratio

    # Penalità di concentrazione: sum w_i^2
    if lambda_conc > 0:
        conc_penalty = torch.mean(torch.sum(weights ** 2, dim=1))
        loss = loss + lambda_conc * conc_penalty

    # Penalità di distanza dal benchmark: ||w - m||^2
    if lambda_bench > 0:
        diff = weights - market_weights
        bench_penalty = torch.mean(torch.sum(diff ** 2, dim=1))
        loss = loss + lambda_bench * bench_penalty

    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ==============================
# 8. Training loop
# ==============================
for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []

    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        w_batch = model(X_batch)                # pesi del portafoglio
        loss = risk_adjusted_loss(
            w_batch,
            Y_batch,
            lambda_conc=lambda_conc_train,
            lambda_bench=lambda_bench_train,
        )
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Valutazione grezza su test set con la stessa loss
    model.eval()
    test_losses = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            w_batch = model(X_batch)
            loss = risk_adjusted_loss(
                w_batch,
                Y_batch,
                lambda_conc=lambda_conc_train,
                lambda_bench=lambda_bench_train,
            )
            test_losses.append(loss.item())

    print(f"Epoch {epoch:03d} | "
          f"Train loss: {np.mean(train_losses):.6f} | "
          f"Test loss: {np.mean(test_losses):.6f}")

# ==============================
# 9. Esempio: pesi e rendimento sull'ultimo sample di test
# ==============================
model.eval()
with torch.no_grad():
    X_last = torch.tensor(X_test[-1:], dtype=torch.float32)   # shape (1, input_dim)
    w_last_tensor = model(X_last)[0]                          # (n_assets,)

    # rendimento a 1 anno corrispondente (vettore R_delta per l'ultimo sample)
    R_last_tensor = torch.tensor(Y_test[-1], dtype=torch.float32)  # (n_assets,)

    # rendimento di portafoglio a 1 anno: w_t^T R_delta_t
    port_ret_last = torch.dot(w_last_tensor, R_last_tensor).item()

    w_last = w_last_tensor.numpy()

print("\nPesi del portafoglio per l'ultimo sample di test:")
for name, w_i in zip(df_prices.columns, w_last):
    print(f"{name}: {w_i:.4f}")

print(f"\nRendimento del portafoglio nell'ultimo anno (ultimo sample di test): {port_ret_last:.4%}")
print(f"Fattore di crescita partendo da 1 unità di capitale: {1.0 + port_ret_last:.4f}")

# ==============================
# 10. Portafoglio min-var (unconstrained + long-only fix)
# ==============================
Y_train_np = Y_train
Y_test_np  = Y_test

n_assets = Y_train_np.shape[1]

Sigma = np.cov(Y_train_np, rowvar=False)
Sigma += 1e-6 * np.eye(n_assets)
ones = np.ones(n_assets)

# soluzione unconstrained (può shortare e levare)
Sigma_inv_ones = np.linalg.solve(Sigma, ones)
w_mv = Sigma_inv_ones / (ones @ Sigma_inv_ones)

print("\nPesi min-var unconstrained (possono shortare):")
for name, w in zip(df_prices.columns, w_mv):
    print(f"{name}: {w:.4f}")

# Fix long-only: clip a 0 e rinormalizza
w_mv_long = np.clip(w_mv, 0.0, None)
if w_mv_long.sum() > 0:
    w_mv_long /= w_mv_long.sum()
else:
    w_mv_long = np.ones_like(w_mv_long) / len(w_mv_long)

print("\nPesi min-var LONG-ONLY (rinormalizzati):")
for name, w in zip(df_prices.columns, w_mv_long):
    print(f"{name}: {w:.4f}")

# Ritorni del portafoglio min-var long-only su TRAIN e TEST
train_port_ret_mv  = Y_train_np @ w_mv_long
test_port_ret_mv   = Y_test_np  @ w_mv_long

print("\nSTATISTICHE MIN-VAR LONG-ONLY (ORIZZONTE 1 ANNO):")
print(f"Train - rendimento medio 1y: {train_port_ret_mv.mean():.4%}")
print(f"Train - deviazione standard 1y: {train_port_ret_mv.std():.4%}")
print(f"Test  - rendimento medio 1y: {test_port_ret_mv.mean():.4%}")
print(f"Test  - deviazione standard 1y: {test_port_ret_mv.std():.4%}")

# Log-growth per evitare overflow
train_log_growth_mv = np.sum(np.log1p(train_port_ret_mv))
test_log_growth_mv  = np.sum(np.log1p(test_port_ret_mv))

print(f"\nLog fattore di crescita train (min-var long-only): {train_log_growth_mv:.4f}")
print(f"Log fattore di crescita test  (min-var long-only): {test_log_growth_mv:.4f}")

# ==============================
# 11. Confronto finale: NN vs market-cap vs equal-weight vs min-var long-only
# ==============================
# Ritorni 1y della strategia NN su TUTTO il train/test
model.eval()
with torch.no_grad():
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

    W_train_t = model(X_train_t)   # (N_train, n_assets)
    W_test_t  = model(X_test_t)    # (N_test, n_assets)

W_train = W_train_t.numpy()
W_test  = W_test_t.numpy()

train_port_ret_nn = np.sum(W_train * Y_train_np, axis=1)
test_port_ret_nn  = np.sum(W_test  * Y_test_np,  axis=1)

# Market-cap benchmark e equal-weight
market_w = market_w_array
ew_w = np.ones_like(market_w) / len(market_w)

train_port_ret_mc = Y_train_np @ market_w
test_port_ret_mc  = Y_test_np  @ market_w

train_port_ret_ew = Y_train_np @ ew_w
test_port_ret_ew  = Y_test_np  @ ew_w

def summarize(name, R, R_bench=None):
    mean = R.mean()
    std  = R.std()
    print(f"\n=== {name} ===")
    print(f"Rendimento medio 1y: {mean:.4%}")
    print(f"Deviazione std 1y:  {std:.4%}")
    if std > 0:
        print(f"Sharpe (rf=0):      {mean / std:.3f}")
    if R_bench is not None:
        active = R - R_bench
        mean_a = active.mean()
        std_a  = active.std()
        print(f"Active return medio 1y: {mean_a:.4%}")
        if std_a > 0:
            print(f"Information ratio:      {mean_a / std_a:.3f}")

print("\n\n*** CONFRONTO TEST SET ***")
summarize("NN strategy",          test_port_ret_nn, R_bench=test_port_ret_mc)
summarize("Min-var long-only",    test_port_ret_mv, R_bench=test_port_ret_mc)
summarize("Market-cap benchmark", test_port_ret_mc)
summarize("Equal-weight",         test_port_ret_ew, R_bench=test_port_ret_mc)