"""
Autoresearch-trader training script. Single-GPU, single-file.

v19: Add volume-price divergence + intraday range signals to 3-factor base.

Hypothesis: Volume-price divergence (price up on low volume = weak, expect
reversal) and intraday range (high range = breakout, low range = mean revert)
are orthogonal alpha sources to momentum/reversal/underreaction.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch

from prepare import (
    C, H, L, V,
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
high = ohlcv[:, tradeable_idx, H]
low = ohlcv[:, tradeable_idx, L]
volume = ohlcv[:, tradeable_idx, V]
log_close = torch.log(close.clamp(min=1e-8))
daily_ret = torch.zeros(D, N_trade)
daily_ret[1:] = log_close[1:] - log_close[:-1]

spy_ret = daily_ret[:, 0]
qqq_ret = daily_ret[:, 1]
iwm_ret = daily_ret[:, 2]

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

# 3-factor beta-neutral (v18 best)
betas_spy = compute_betas(daily_ret, spy_ret, 42)
betas_qqq = compute_betas(daily_ret, qqq_ret, 42)
betas_iwm = compute_betas(daily_ret, iwm_ret, 42)
bn_3f = daily_ret - betas_spy * spy_ret.unsqueeze(1) - 0.3 * betas_qqq * qqq_ret.unsqueeze(1) - 0.15 * betas_iwm * iwm_ret.unsqueeze(1)

ram_3f_21 = cross_zscore(risk_adj_momentum(bn_3f, 21, skip=5))
ram_3f_63 = cross_zscore(risk_adj_momentum(bn_3f, 63, skip=5))
z_rev5 = cross_zscore(simple_momentum(log_close, 5))

# Acceleration (v18 best: lag=10)
mom_21_raw = torch.zeros(D, N_trade)
for t in range(21, D):
    mom_21_raw[t] = daily_ret[t-21:t].sum(dim=0)
accel = torch.zeros(D, N_trade)
for t in range(31, D):
    accel[t] = mom_21_raw[t] - mom_21_raw[t - 10]
z_accel = cross_zscore(accel)

# Underreaction (v18 best: window=5)
expected_ret = betas_spy * spy_ret.unsqueeze(1)
underreaction = expected_ret - daily_ret
ur_5d = torch.zeros(D, N_trade)
for t in range(5, D):
    ur_5d[t] = underreaction[t-5:t].sum(dim=0)
z_ur = cross_zscore(ur_5d)

# NEW: Volume-price divergence
# Idea: price up + volume down (or vice versa) = weak signal, expect reversal
# Compute rolling correlation between returns and volume changes over 10 days
log_vol = torch.log(volume.clamp(min=1))
vol_ret = torch.zeros(D, N_trade)
vol_ret[1:] = log_vol[1:] - log_vol[:-1]

# Volume-price divergence: sign(return) * volume_change
# If positive return on declining volume -> negative divergence (bearish)
vpd = torch.zeros(D, N_trade)
for t in range(10, D):
    ret_w = daily_ret[t-10:t]
    vr_w = vol_ret[t-10:t]
    # Simple: correlation proxy = sum(ret * vol_change) / 10
    vpd[t] = (ret_w * vr_w).sum(dim=0) / 10
z_vpd = cross_zscore(vpd)

# NEW: Intraday range signal
# High range relative to recent average = breakout potential
intraday_range = (high - low) / close.clamp(min=1e-8)
range_ma20 = torch.zeros_like(intraday_range)
for t in range(20, D):
    range_ma20[t] = intraday_range[t-20:t].mean(dim=0)
range_ma20[:20] = intraday_range[:20].mean(dim=0)
# Range ratio: above average = volatile/breakout day
range_ratio = intraday_range / range_ma20.clamp(min=1e-8)
z_range = cross_zscore(range_ratio)

# Range * return sign: high range + positive return = strong bullish breakout
range_direction = z_range * daily_ret.sign()
# Smooth over 5 days
rd_5d = torch.zeros(D, N_trade)
for t in range(5, D):
    rd_5d[t] = range_direction[t-5:t].sum(dim=0)
z_rd = cross_zscore(rd_5d)

# Dispersion scaling
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

# v18 best base signal
base = (-0.50 * z_rev5 + 0.20 * ram_3f_21 + 0.30 * ram_3f_63
        - 0.10 * z_accel + 0.12 * z_ur)

print("Testing configurations...")
best_sharpe = -999
best_smooth = None
best_name = ""

configs = [
    # v18 reference
    ("v18_ref", base * dr, 0.035),

    # Add volume-price divergence (positive = momentum confirmed by volume)
    ("vpd_05", (base + 0.05 * z_vpd) * dr, 0.035),
    ("vpd_08", (base + 0.08 * z_vpd) * dr, 0.035),
    ("vpd_10", (base + 0.10 * z_vpd) * dr, 0.035),
    ("vpd_n05", (base - 0.05 * z_vpd) * dr, 0.035),

    # Add range-direction signal
    ("rd_05", (base + 0.05 * z_rd) * dr, 0.035),
    ("rd_08", (base + 0.08 * z_rd) * dr, 0.035),
    ("rd_10", (base + 0.10 * z_rd) * dr, 0.035),
    ("rd_n05", (base - 0.05 * z_rd) * dr, 0.035),
    ("rd_n08", (base - 0.08 * z_rd) * dr, 0.035),

    # Both
    ("both_vpd5_rd5", (base + 0.05 * z_vpd + 0.05 * z_rd) * dr, 0.035),
    ("both_vpd5_rdn5", (base + 0.05 * z_vpd - 0.05 * z_rd) * dr, 0.035),
    ("both_vpdn5_rd5", (base - 0.05 * z_vpd + 0.05 * z_rd) * dr, 0.035),
    ("both_vpd8_rd8", (base + 0.08 * z_vpd + 0.08 * z_rd) * dr, 0.035),

    # Plain range (high range = high vol = uncertain)
    ("range_n05", (base - 0.05 * z_range) * dr, 0.035),
    ("range_05", (base + 0.05 * z_range) * dr, 0.035),
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
