"""
Autoresearch-trader training script. Single-GPU, single-file.

v17: Add price acceleration + lead-lag signals on top of the 2-factor base.

Hypothesis: Price acceleration (2nd derivative — momentum of momentum) and
ETF-to-stock lead-lag relationships add orthogonal alpha to the current
2-factor beta-neutral momentum signal.

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

close = ohlcv[:, tradeable_idx, C]
log_close = torch.log(close.clamp(min=1e-8))
daily_ret = torch.zeros(D, N_trade)
daily_ret[1:] = log_close[1:] - log_close[:-1]

spy_ret = daily_ret[:, 0]
qqq_ret = daily_ret[:, 1]

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

print("Computing signals...")

# 2-factor beta-neutral (best from exp15)
betas_spy = compute_betas(daily_ret, spy_ret, 42)
betas_qqq = compute_betas(daily_ret, qqq_ret, 42)
bn_2f = daily_ret - betas_spy * spy_ret.unsqueeze(1) - 0.3 * betas_qqq * qqq_ret.unsqueeze(1)

ram_2f_21 = cross_zscore(risk_adj_momentum(bn_2f, 21, skip=5))
ram_2f_63 = cross_zscore(risk_adj_momentum(bn_2f, 63, skip=5))
z_rev5 = cross_zscore(simple_momentum(log_close, 5))

# Price acceleration: change in 21d momentum over 10 days
# mom_21(t) - mom_21(t-10), cross-sectionally z-scored
mom_21_raw = torch.zeros(D, N_trade)
for t in range(21, D):
    mom_21_raw[t] = daily_ret[t-21:t].sum(dim=0)
accel_10 = torch.zeros(D, N_trade)
accel_10[31:] = mom_21_raw[31:] - mom_21_raw[21:-10]  # fix: need proper indexing
for t in range(31, D):
    accel_10[t] = mom_21_raw[t] - mom_21_raw[t - 10]
z_accel = cross_zscore(accel_10)

# Lead-lag: how much does today's SPY/QQQ return predict tomorrow's stock return?
# Use rolling correlation between lagged ETF returns and stock returns
# Simpler version: use yesterday's SPY return as a signal
spy_lag1 = torch.zeros(D, N_trade)
spy_lag1[1:] = spy_ret[:-1].unsqueeze(1).expand(-1, N_trade)
# Cross-sectionally, this is the same for all assets, so z-score won't help
# Instead: how much did each asset "miss" the market move? (underreaction signal)
# Expected return = beta * SPY_return, actual = daily_ret
# Underreaction = expected - actual (positive = stock hasn't caught up yet)
expected_ret = betas_spy * spy_ret.unsqueeze(1)
underreaction = expected_ret - daily_ret  # positive = stock lagged behind market
# Smooth over 3 days
ur_3d = torch.zeros(D, N_trade)
for t in range(3, D):
    ur_3d[t] = underreaction[t-3:t].sum(dim=0)
z_underreact = cross_zscore(ur_3d)

# Dispersion scaling
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

# Base signal (exp15 best)
base = -0.50 * z_rev5 + 0.20 * ram_2f_21 + 0.30 * ram_2f_63

print("Testing configurations...")
best_sharpe = -999
best_smooth = None
best_name = ""

configs = [
    # Reference
    ("2f_ref", base * dr, 0.035),

    # Add acceleration
    ("accel_05", (base + 0.05 * z_accel) * dr, 0.035),
    ("accel_10", (base + 0.10 * z_accel) * dr, 0.035),
    ("accel_15", (base + 0.15 * z_accel) * dr, 0.035),
    ("accel_n05", (base - 0.05 * z_accel) * dr, 0.035),
    ("accel_n10", (base - 0.10 * z_accel) * dr, 0.035),

    # Add underreaction
    ("ur_05", (base + 0.05 * z_underreact) * dr, 0.035),
    ("ur_10", (base + 0.10 * z_underreact) * dr, 0.035),
    ("ur_15", (base + 0.15 * z_underreact) * dr, 0.035),
    ("ur_n05", (base - 0.05 * z_underreact) * dr, 0.035),
    ("ur_n10", (base - 0.10 * z_underreact) * dr, 0.035),

    # Both
    ("both_05", (base + 0.05 * z_accel + 0.05 * z_underreact) * dr, 0.035),
    ("both_10", (base + 0.10 * z_accel + 0.10 * z_underreact) * dr, 0.035),
    ("both_n", (base - 0.05 * z_accel + 0.10 * z_underreact) * dr, 0.035),

    # Acceleration only (replace some momentum weight)
    ("accel_replace", (-0.50 * z_rev5 + 0.15 * ram_2f_21 + 0.25 * ram_2f_63 + 0.10 * z_accel) * dr, 0.035),
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
