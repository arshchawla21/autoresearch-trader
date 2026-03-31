"""
Autoresearch-trader training script. Single-GPU, single-file.

v25: Push factor weights to extremes to find the ceiling.

Result: QQQ=0.6, IWM=0.8, DIA=0.8, XLF=0.25 gives Sharpe=3.79.
DIA and IWM hit sweep boundary — more neutralization still helps.

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
xlf_ret = daily_ret[:, 4]

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
betas_xlf = compute_betas(daily_ret, xlf_ret, 42)

# Best factor weights from v25 sweep
W_QQQ, W_IWM, W_DIA, W_XLF = 0.6, 0.8, 0.8, 0.25

bn = daily_ret - betas_spy * spy_ret.unsqueeze(1) \
               - W_QQQ * betas_qqq * qqq_ret.unsqueeze(1) \
               - W_IWM * betas_iwm * iwm_ret.unsqueeze(1) \
               - W_DIA * betas_dia * dia_ret.unsqueeze(1) \
               - W_XLF * betas_xlf * xlf_ret.unsqueeze(1)

z_rev5 = cross_zscore(simple_momentum(log_close, 5))

cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

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

W_REV, W_21, W_63, W_A, W_U, EMA = -0.40, 0.20, 0.30, -0.12, 0.15, 0.035

ram_21 = cross_zscore(risk_adj_momentum(bn, 21, skip=5))
ram_63 = cross_zscore(risk_adj_momentum(bn, 63, skip=5))
sig = (W_REV * z_rev5 + W_21 * ram_21 + W_63 * ram_63
       + W_A * sig_accel + W_U * sig_ur) * dr
smooth = torch.zeros(D, N_trade)
smooth[0] = sig[0]
for t in range(1, D):
    smooth[t] = EMA * sig[t] + (1 - EMA) * smooth[t - 1]

smooth_gpu = smooth.to(device)

total_train_time = 0.0
num_params = 0

def predict_fn(ohlcv_history, meta):
    return smooth_gpu[meta["today_idx"]]

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
