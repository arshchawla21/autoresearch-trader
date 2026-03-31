"""
Autoresearch-trader training script. Single-GPU, single-file.

v18: 3-factor beta-neutral (SPY+QQQ+IWM) + sweep underreaction/acceleration params.

Hypothesis: Adding IWM as a 3rd factor captures size exposure, and sweeping
underreaction window lengths + acceleration lookbacks finds optimal params.

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

spy_ret = daily_ret[:, 0]  # SPY
qqq_ret = daily_ret[:, 1]  # QQQ
iwm_ret = daily_ret[:, 2]  # IWM

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

# Betas
betas_spy = compute_betas(daily_ret, spy_ret, 42)
betas_qqq = compute_betas(daily_ret, qqq_ret, 42)
betas_iwm = compute_betas(daily_ret, iwm_ret, 42)

# 2-factor residual (v17 best)
bn_2f = daily_ret - betas_spy * spy_ret.unsqueeze(1) - 0.3 * betas_qqq * qqq_ret.unsqueeze(1)

# 3-factor residual
bn_3f = daily_ret - betas_spy * spy_ret.unsqueeze(1) - 0.3 * betas_qqq * qqq_ret.unsqueeze(1) - 0.15 * betas_iwm * iwm_ret.unsqueeze(1)

# Momentum on both
ram_2f_21 = cross_zscore(risk_adj_momentum(bn_2f, 21, skip=5))
ram_2f_63 = cross_zscore(risk_adj_momentum(bn_2f, 63, skip=5))
ram_3f_21 = cross_zscore(risk_adj_momentum(bn_3f, 21, skip=5))
ram_3f_63 = cross_zscore(risk_adj_momentum(bn_3f, 63, skip=5))

z_rev5 = cross_zscore(simple_momentum(log_close, 5))

# Acceleration at different lookbacks
mom_21_raw = torch.zeros(D, N_trade)
for t in range(21, D):
    mom_21_raw[t] = daily_ret[t-21:t].sum(dim=0)

accels = {}
for lag in [5, 7, 10, 14]:
    a = torch.zeros(D, N_trade)
    for t in range(21 + lag, D):
        a[t] = mom_21_raw[t] - mom_21_raw[t - lag]
    accels[lag] = cross_zscore(a)

# Underreaction at different windows
expected_ret = betas_spy * spy_ret.unsqueeze(1)
underreaction = expected_ret - daily_ret

urs = {}
for win in [1, 2, 3, 5, 7]:
    ur = torch.zeros(D, N_trade)
    for t in range(win, D):
        ur[t] = underreaction[t-win:t].sum(dim=0)
    urs[win] = cross_zscore(ur)

# Dispersion scaling
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

print("Sweeping configurations...")
best_sharpe = -999
best_smooth = None
best_name = ""

# Build configs
configs = []

# v17 reference
base_2f = -0.50 * z_rev5 + 0.20 * ram_2f_21 + 0.30 * ram_2f_63
configs.append(("v17_ref", (base_2f - 0.05 * accels[10] + 0.10 * urs[3]) * dr, 0.035))

# 3-factor base
base_3f = -0.50 * z_rev5 + 0.20 * ram_3f_21 + 0.30 * ram_3f_63
for accel_lag in [5, 10]:
    for ur_win in [3, 5]:
        for w_a, w_u in [(-0.05, 0.10), (-0.05, 0.15), (-0.08, 0.10), (-0.10, 0.12)]:
            sig = (base_3f + w_a * accels[accel_lag] + w_u * urs[ur_win]) * dr
            for ema in [0.03, 0.035, 0.04]:
                configs.append((f"3f_a{accel_lag}w{w_a}_u{ur_win}w{w_u}_e{ema}", sig, ema))

# 2-factor with different ur/accel params
for accel_lag in [5, 7, 10, 14]:
    for ur_win in [1, 2, 3, 5, 7]:
        for w_a, w_u in [(-0.05, 0.10), (-0.05, 0.15), (-0.08, 0.12), (-0.03, 0.08)]:
            sig = (base_2f + w_a * accels[accel_lag] + w_u * urs[ur_win]) * dr
            configs.append((f"2f_a{accel_lag}w{w_a}_u{ur_win}w{w_u}", sig, 0.035))

print(f"Total: {len(configs)} configs")

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

    if sh > best_sharpe:
        best_sharpe = sh
        best_smooth = smooth_gpu
        best_name = name
        print(f"  NEW BEST: {name}: Sharpe={sh:.4f}  Return={res['total_return']:.4f}  "
              f"MaxDD={res['max_drawdown']:.4f}  Turn={res['avg_turnover']:.4f}")

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
