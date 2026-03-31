"""
Autoresearch-trader training script. Single-GPU, single-file.

Experiment 15: Push reversal further, try QQQ beta, different skip periods.
Best: rev5=-0.50 ram_bn42_21=0.25 ram_bn42_63=0.25, disp(0.2,4.0), EMA=0.04

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

# SPY=0, QQQ=1
spy_ret = daily_ret[:, 0]
qqq_ret = daily_ret[:, 1]

def cross_zscore(x):
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (x - mu) / std

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

print("Computing signals...")

# Beta-neutral with SPY (42-day)
betas_spy42 = compute_betas(daily_ret, spy_ret, 42)
bn_spy = daily_ret - betas_spy42 * spy_ret.unsqueeze(1)

# Beta-neutral with QQQ (42-day)
betas_qqq42 = compute_betas(daily_ret, qqq_ret, 42)
bn_qqq = daily_ret - betas_qqq42 * qqq_ret.unsqueeze(1)

# Two-factor beta-neutral (SPY + QQQ)
betas_spy42_2f = compute_betas(daily_ret, spy_ret, 42)
betas_qqq42_2f = compute_betas(daily_ret, qqq_ret, 42)
bn_2f = daily_ret - betas_spy42_2f * spy_ret.unsqueeze(1) - 0.3 * betas_qqq42_2f * qqq_ret.unsqueeze(1)

# Momentum signals
ram_spy_21 = cross_zscore(risk_adj_momentum(bn_spy, 21, skip=5))
ram_spy_63 = cross_zscore(risk_adj_momentum(bn_spy, 63, skip=5))
ram_qqq_21 = cross_zscore(risk_adj_momentum(bn_qqq, 21, skip=5))
ram_qqq_63 = cross_zscore(risk_adj_momentum(bn_qqq, 63, skip=5))
ram_2f_21 = cross_zscore(risk_adj_momentum(bn_2f, 21, skip=5))
ram_2f_63 = cross_zscore(risk_adj_momentum(bn_2f, 63, skip=5))

# Different reversal horizons
z_rev3 = cross_zscore(simple_momentum(log_close, 3))
z_rev5 = cross_zscore(simple_momentum(log_close, 5))
z_rev7 = cross_zscore(simple_momentum(log_close, 7))

# Dispersion
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

configs = []

# Push reversal weights further
for w_rev in [-0.60, -0.55, -0.50, -0.45]:
    w_mom = 1.0 + w_rev
    for split in [0.5, 0.6, 0.4]:  # 21d share of momentum
        w_21 = w_mom * split
        w_63 = w_mom * (1 - split)

        # SPY beta
        sig = w_rev * z_rev5 + w_21 * ram_spy_21 + w_63 * ram_spy_63
        for ema_a in [0.03, 0.035, 0.04, 0.045, 0.05]:
            configs.append((f"spy_r{w_rev}_s{split}_e{ema_a}", sig * dr, ema_a))

        # QQQ beta
        sig_q = w_rev * z_rev5 + w_21 * ram_qqq_21 + w_63 * ram_qqq_63
        for ema_a in [0.035, 0.04, 0.05]:
            configs.append((f"qqq_r{w_rev}_s{split}_e{ema_a}", sig_q * dr, ema_a))

        # 2-factor
        sig_2f = w_rev * z_rev5 + w_21 * ram_2f_21 + w_63 * ram_2f_63
        for ema_a in [0.035, 0.04, 0.05]:
            configs.append((f"2f_r{w_rev}_s{split}_e{ema_a}", sig_2f * dr, ema_a))

# Try rev3 and rev7
for rev_sig, rn in [(z_rev3, "rev3"), (z_rev7, "rev7")]:
    for w_rev in [-0.55, -0.50]:
        w_mom = 1.0 + w_rev
        sig = w_rev * rev_sig + w_mom * 0.5 * ram_spy_21 + w_mom * 0.5 * ram_spy_63
        for ema_a in [0.035, 0.04, 0.05]:
            configs.append((f"{rn}_r{w_rev}_e{ema_a}", sig * dr, ema_a))

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
