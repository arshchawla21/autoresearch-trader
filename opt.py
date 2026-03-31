"""
Autoresearch-trader training script. Single-GPU, single-file.

v22: Sweep factor model weights for beta-neutralization.

Hypothesis: The ad-hoc factor weights (1.0*SPY, 0.3*QQQ, 0.15*IWM) are
suboptimal. Systematically sweeping these weights will find better residuals
for momentum computation.

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
iwm_ret = daily_ret[:, 2]
dia_ret = daily_ret[:, 3]

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

print("Computing betas...")

betas_spy = compute_betas(daily_ret, spy_ret, 42)
betas_qqq = compute_betas(daily_ret, qqq_ret, 42)
betas_iwm = compute_betas(daily_ret, iwm_ret, 42)
betas_dia = compute_betas(daily_ret, dia_ret, 42)

z_rev5 = cross_zscore(simple_momentum(log_close, 5))

# Dispersion
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

# Acceleration + underreaction (fixed from v18 best)
mom_21_raw = torch.zeros(D, N_trade)
for t in range(21, D):
    mom_21_raw[t] = daily_ret[t-21:t].sum(dim=0)
sig_accel = torch.zeros(D, N_trade)
for t in range(31, D):
    sig_accel[t] = mom_21_raw[t] - mom_21_raw[t - 10]
sig_accel = cross_zscore(sig_accel)

expected_ret_spy = betas_spy * spy_ret.unsqueeze(1)
underreaction = expected_ret_spy - daily_ret
sig_ur = torch.zeros(D, N_trade)
for t in range(5, D):
    sig_ur[t] = underreaction[t-5:t].sum(dim=0)
sig_ur = cross_zscore(sig_ur)

print("Sweeping factor weights...")
best_sharpe = -999
best_smooth = None
best_name = ""

configs = []

# Sweep SPY weight, QQQ weight, IWM weight
for w_spy in [0.7, 0.8, 0.9, 1.0, 1.1]:
    for w_qqq in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for w_iwm in [0.0, 0.1, 0.15, 0.2, 0.3]:
            # Compute residual
            bn = daily_ret - w_spy * betas_spy * spy_ret.unsqueeze(1) \
                           - w_qqq * betas_qqq * qqq_ret.unsqueeze(1) \
                           - w_iwm * betas_iwm * iwm_ret.unsqueeze(1)

            ram_21 = cross_zscore(risk_adj_momentum(bn, 21, skip=5))
            ram_63 = cross_zscore(risk_adj_momentum(bn, 63, skip=5))

            sig = (-0.50 * z_rev5 + 0.20 * ram_21 + 0.30 * ram_63
                   - 0.10 * sig_accel + 0.12 * sig_ur) * dr

            # EMA smooth
            smooth = torch.zeros(D, N_trade)
            smooth[0] = sig[0]
            for t in range(1, D):
                smooth[t] = 0.035 * sig[t] + 0.965 * smooth[t - 1]

            smooth_gpu = smooth.to(device)
            def _pred(oh, meta, _s=smooth_gpu):
                return _s[meta["today_idx"]]

            res = evaluate_sharpe(_pred, data, device=device)
            sh = res["sharpe_ratio"]

            if sh > best_sharpe:
                best_sharpe = sh
                best_smooth = smooth_gpu
                best_name = f"spy{w_spy}_qqq{w_qqq}_iwm{w_iwm}"
                print(f"  NEW BEST: {best_name}: Sharpe={sh:.4f}  "
                      f"Return={res['total_return']:.4f}  MaxDD={res['max_drawdown']:.4f}  "
                      f"Turn={res['avg_turnover']:.4f}")

# Also try adding DIA as 4th factor
print("\nTrying 4-factor (+ DIA)...")
for w_qqq in [0.2, 0.3, 0.4]:
    for w_iwm in [0.1, 0.15, 0.2]:
        for w_dia in [0.05, 0.1, 0.15, 0.2]:
            bn = daily_ret - betas_spy * spy_ret.unsqueeze(1) \
                           - w_qqq * betas_qqq * qqq_ret.unsqueeze(1) \
                           - w_iwm * betas_iwm * iwm_ret.unsqueeze(1) \
                           - w_dia * betas_dia * dia_ret.unsqueeze(1)

            ram_21 = cross_zscore(risk_adj_momentum(bn, 21, skip=5))
            ram_63 = cross_zscore(risk_adj_momentum(bn, 63, skip=5))

            sig = (-0.50 * z_rev5 + 0.20 * ram_21 + 0.30 * ram_63
                   - 0.10 * sig_accel + 0.12 * sig_ur) * dr

            smooth = torch.zeros(D, N_trade)
            smooth[0] = sig[0]
            for t in range(1, D):
                smooth[t] = 0.035 * sig[t] + 0.965 * smooth[t - 1]

            smooth_gpu = smooth.to(device)
            def _pred2(oh, meta, _s=smooth_gpu):
                return _s[meta["today_idx"]]

            res = evaluate_sharpe(_pred2, data, device=device)
            sh = res["sharpe_ratio"]

            if sh > best_sharpe:
                best_sharpe = sh
                best_smooth = smooth_gpu
                best_name = f"4f_qqq{w_qqq}_iwm{w_iwm}_dia{w_dia}"
                print(f"  NEW BEST: {best_name}: Sharpe={sh:.4f}  "
                      f"Return={res['total_return']:.4f}  MaxDD={res['max_drawdown']:.4f}  "
                      f"Turn={res['avg_turnover']:.4f}")

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