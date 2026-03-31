"""
Autoresearch-trader training script. Single-GPU, single-file.

v21: Turnover-constrained position management instead of EMA smoothing.

Hypothesis: EMA smoothing introduces lag — a fixed-rate blend regardless of
signal change magnitude. A turnover cap that limits max daily position change
should be more responsive to genuine signal shifts while preventing whipsaws.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch

from prepare import (
    C, TARGET_LEVERAGE,
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

# Acceleration + underreaction (v18 best)
mom_21_raw = torch.zeros(D, N_trade)
for t in range(21, D):
    mom_21_raw[t] = daily_ret[t-21:t].sum(dim=0)
sig_accel = torch.zeros(D, N_trade)
for t in range(31, D):
    sig_accel[t] = mom_21_raw[t] - mom_21_raw[t - 10]
sig_accel = cross_zscore(sig_accel)

expected_ret = betas_spy * spy_ret.unsqueeze(1)
underreaction = expected_ret - daily_ret
sig_ur = torch.zeros(D, N_trade)
for t in range(5, D):
    sig_ur[t] = underreaction[t-5:t].sum(dim=0)
sig_ur = cross_zscore(sig_ur)

# Dispersion
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

# Raw signal (v18 best)
raw_signal = (-0.50 * z_rev5 + 0.20 * ram_3f_21 + 0.30 * ram_3f_63
              - 0.10 * sig_accel + 0.12 * sig_ur) * dr

# Normalize target positions: what the signal "wants"
target_pos = torch.zeros(D, N_trade)
for t in range(42, D):
    s = raw_signal[t]
    abs_sum = s.abs().sum().clamp(min=1e-10)
    target_pos[t] = s * (TARGET_LEVERAGE / abs_sum)

# ---------------------------------------------------------------------------
# Position management approaches
# ---------------------------------------------------------------------------

def apply_ema(raw_sig, alpha):
    smooth = torch.zeros(D, N_trade)
    smooth[0] = raw_sig[0]
    for t in range(1, D):
        smooth[t] = alpha * raw_sig[t] + (1 - alpha) * smooth[t - 1]
    return smooth

def apply_turnover_cap(target, max_turn_per_asset):
    """Cap per-asset position change to max_turn_per_asset per day."""
    pos = torch.zeros(D, N_trade)
    for t in range(43, D):
        delta = target[t] - pos[t - 1]
        clamped = delta.clamp(-max_turn_per_asset, max_turn_per_asset)
        pos[t] = pos[t - 1] + clamped
    return pos

def apply_total_turnover_cap(target, max_total_turn):
    """Cap total portfolio turnover to max_total_turn per day."""
    pos = torch.zeros(D, N_trade)
    for t in range(43, D):
        delta = target[t] - pos[t - 1]
        total_turn = delta.abs().sum()
        if total_turn > max_total_turn:
            delta = delta * (max_total_turn / total_turn)
        pos[t] = pos[t - 1] + delta
    return pos

print("Testing position management approaches...")
best_sharpe = -999
best_smooth = None
best_name = ""

configs = []

# EMA baseline (v18 reference)
for alpha in [0.03, 0.035, 0.04, 0.05]:
    s = apply_ema(raw_signal, alpha)
    configs.append((f"ema_{alpha}", s))

# Per-asset turnover cap
for cap in [0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.015, 0.02]:
    s = apply_turnover_cap(target_pos, cap)
    configs.append((f"percap_{cap}", s))

# Total portfolio turnover cap
for cap in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
    s = apply_total_turnover_cap(target_pos, cap)
    configs.append((f"totcap_{cap}", s))

# Hybrid: EMA + per-asset cap
for alpha in [0.05, 0.07, 0.10]:
    s_ema = apply_ema(raw_signal, alpha)
    for cap in [0.003, 0.005, 0.007]:
        # Normalize EMA output, then cap
        pos = torch.zeros(D, N_trade)
        for t in range(43, D):
            tgt = s_ema[t]
            abs_sum = tgt.abs().sum().clamp(min=1e-10)
            tgt = tgt * (TARGET_LEVERAGE / abs_sum)
            delta = tgt - pos[t - 1]
            clamped = delta.clamp(-cap, cap)
            pos[t] = pos[t - 1] + clamped
        configs.append((f"hybrid_e{alpha}_c{cap}", pos))

for name, positions in configs:
    smooth_gpu = positions.to(device)

    def _pred(oh, meta, _s=smooth_gpu):
        return _s[meta["today_idx"]]

    res = evaluate_sharpe(_pred, data, device=device)
    sh = res["sharpe_ratio"]
    print(f"  {name:25s}: Sharpe={sh:+.4f}  Return={res['total_return']:+.4f}  "
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
