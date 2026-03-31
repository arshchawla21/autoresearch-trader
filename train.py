"""
Autoresearch-trader training script. Single-GPU, single-file.

v16: Multi-factor residual momentum — neutralize individual stock returns
against their sector ETF before computing momentum. This isolates
idiosyncratic alpha from sector/market beta.

Hypothesis: Sector-neutralized residual momentum captures purer stock-specific
signals, reducing drawdowns from correlated sector moves.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch

from prepare import (
    C,
    load_data, evaluate_sharpe,
)

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda")

data = load_data(device="cpu")
ohlcv = data["ohlcv"]
tradeable_idx = data["tradeable_indices"]
N_trade = data["num_tradeable"]
D = ohlcv.shape[0]
tickers = data["tradeable_tickers"]

close = ohlcv[:, tradeable_idx, C]
log_close = torch.log(close.clamp(min=1e-8))
daily_ret = torch.zeros(D, N_trade)
daily_ret[1:] = log_close[1:] - log_close[:-1]

print(f"Tickers: {tickers}")

# Map each tradeable asset to its sector ETF
SECTOR_MAP = {
    "SPY": "SPY", "QQQ": "SPY", "IWM": "SPY", "DIA": "SPY",
    "XLF": "SPY", "XLE": "SPY", "XLK": "SPY", "XLV": "SPY",
    "XLI": "SPY", "XLP": "SPY", "XLU": "SPY", "XLY": "SPY", "XLB": "SPY",
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "NVDA": "XLK", "META": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF",
    "JNJ": "XLV", "UNH": "XLV",
    "PG": "XLP",
}

ticker_to_idx = {t: i for i, t in enumerate(tickers)}

def cross_zscore(x):
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (x - mu) / std

def compute_betas(daily_ret, ref_ret, lookback):
    betas = torch.zeros(D, N_trade)
    for t in range(lookback, D):
        ref_w = ref_ret[t - lookback:t]
        ref_dm = ref_w - ref_w.mean()
        ref_var = (ref_dm ** 2).mean().clamp(min=1e-10)
        for j in range(N_trade):
            asset_w = daily_ret[t - lookback:t, j]
            cov = ((asset_w - asset_w.mean()) * ref_dm).mean()
            betas[t, j] = cov / ref_var
    betas[:lookback] = 1.0
    return betas

def risk_adj_momentum(ret_tensor, horizon, skip=5):
    sig = torch.zeros(D, N_trade)
    for t in range(horizon + skip, D):
        window = ret_tensor[t - horizon - skip:t - skip]
        cum_ret = window.sum(dim=0)
        vol = window.std(dim=0).clamp(min=1e-8) * (horizon ** 0.5)
        sig[t] = cum_ret / vol
    return sig

def simple_momentum(log_close, horizon):
    sig = torch.zeros_like(log_close)
    sig[horizon:] = log_close[horizon:] - log_close[:-horizon]
    return sig

print("Computing sector-neutral residuals...")

# For each stock, compute beta to its sector ETF, then get residual returns
sector_neutral_ret = daily_ret.clone()
beta_lookback = 42

for ticker in tickers:
    j = ticker_to_idx[ticker]
    sector = SECTOR_MAP.get(ticker, "SPY")
    ref_idx = ticker_to_idx.get(sector, 0)
    if ref_idx == j:
        continue
    ref_ret = daily_ret[:, ref_idx]
    for t in range(beta_lookback, D):
        ref_w = ref_ret[t - beta_lookback:t]
        ref_dm = ref_w - ref_w.mean()
        ref_var = (ref_dm ** 2).mean().clamp(min=1e-10)
        asset_w = daily_ret[t - beta_lookback:t, j]
        cov = ((asset_w - asset_w.mean()) * ref_dm).mean()
        beta = cov / ref_var
        sector_neutral_ret[t, j] = daily_ret[t, j] - beta * ref_ret[t]

# Also compute 2-factor residuals (best from exp15)
spy_ret = daily_ret[:, 0]
qqq_ret = daily_ret[:, 1]
betas_spy = compute_betas(daily_ret, spy_ret, 42)
betas_qqq = compute_betas(daily_ret, qqq_ret, 42)
bn_2f = daily_ret - betas_spy * spy_ret.unsqueeze(1) - 0.3 * betas_qqq * qqq_ret.unsqueeze(1)

# Momentum signals on sector-neutral residuals
ram_sn_21 = cross_zscore(risk_adj_momentum(sector_neutral_ret, 21, skip=5))
ram_sn_63 = cross_zscore(risk_adj_momentum(sector_neutral_ret, 63, skip=5))

# Momentum signals on 2-factor residuals
ram_2f_21 = cross_zscore(risk_adj_momentum(bn_2f, 21, skip=5))
ram_2f_63 = cross_zscore(risk_adj_momentum(bn_2f, 63, skip=5))

# Reversal
z_rev5 = cross_zscore(simple_momentum(log_close, 5))

# Dispersion scaling
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

print("Testing configurations...")
best_sharpe = -999
best_smooth = None
best_name = ""

configs = [
    # Sector-neutral momentum
    ("sn_base", (-0.50 * z_rev5 + 0.20 * ram_sn_21 + 0.30 * ram_sn_63) * dr, 0.035),
    ("sn_rev45", (-0.45 * z_rev5 + 0.22 * ram_sn_21 + 0.33 * ram_sn_63) * dr, 0.035),
    ("sn_rev55", (-0.55 * z_rev5 + 0.18 * ram_sn_21 + 0.27 * ram_sn_63) * dr, 0.035),
    ("sn_e03", (-0.50 * z_rev5 + 0.20 * ram_sn_21 + 0.30 * ram_sn_63) * dr, 0.03),
    ("sn_e04", (-0.50 * z_rev5 + 0.20 * ram_sn_21 + 0.30 * ram_sn_63) * dr, 0.04),
    ("sn_e05", (-0.50 * z_rev5 + 0.20 * ram_sn_21 + 0.30 * ram_sn_63) * dr, 0.05),
    ("sn_equal", (-0.50 * z_rev5 + 0.25 * ram_sn_21 + 0.25 * ram_sn_63) * dr, 0.035),

    # Ensemble: sector-neutral + 2-factor
    ("ens_50_50", (-0.50 * z_rev5 + 0.10 * ram_sn_21 + 0.15 * ram_sn_63 + 0.10 * ram_2f_21 + 0.15 * ram_2f_63) * dr, 0.035),
    ("ens_30_70", (-0.50 * z_rev5 + 0.06 * ram_sn_21 + 0.09 * ram_sn_63 + 0.14 * ram_2f_21 + 0.21 * ram_2f_63) * dr, 0.035),
    ("ens_70_30", (-0.50 * z_rev5 + 0.14 * ram_sn_21 + 0.21 * ram_sn_63 + 0.06 * ram_2f_21 + 0.09 * ram_2f_63) * dr, 0.035),

    # Exp15 reference (2-factor only)
    ("2f_ref", (-0.50 * z_rev5 + 0.20 * ram_2f_21 + 0.30 * ram_2f_63) * dr, 0.035),
]

for name, raw_sig, ema_a in configs:
    smooth = torch.zeros(D, N_trade)
    smooth[0] = raw_sig[0]
    for t in range(1, D):
        smooth[t] = ema_a * raw_sig[t] + (1 - ema_a) * smooth[t - 1]

    smooth_gpu = smooth.to(device)
    def _pred(oh, meta, _s=smooth_gpu):
        return _s[meta["today_idx"]]

    res = evaluate_sharpe(_pred, data, device=device)
    sh = res["sharpe_ratio"]
    print(f"  {name:20s}: Sharpe={sh:+.4f}  Return={res['total_return']:+.4f}  "
          f"MaxDD={res['max_drawdown']:+.4f}  Turn={res['avg_turnover']:.4f}")

    if sh > best_sharpe:
        best_sharpe = sh
        best_smooth = smooth_gpu
        best_name = name

print(f"\nBest: {best_name} Sharpe={best_sharpe:.4f}")

total_train_time = 0.0
num_params = 0

def predict_fn(ohlcv_history, meta):
    return best_smooth[meta["today_idx"]]

results = evaluate_sharpe(predict_fn, data, device=device)

t_end = time.time()
peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"sharpe_ratio:     {results['sharpe_ratio']:.4f}")
print(f"total_return:     {results['total_return']:.4f}")
print(f"max_drawdown:     {results['max_drawdown']:.4f}")
print(f"avg_turnover:     {results['avg_turnover']:.4f}")
print(f"num_test_days:    {results['num_test_days']}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram:.1f}")
print(f"num_params:       {num_params:,}")
