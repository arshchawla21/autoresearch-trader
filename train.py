"""
Autoresearch-trader training script. Single-GPU, single-file.

v20: Train a tiny NN to non-linearly combine our best alpha signals,
conditional on regime (vol, dispersion). Use direct Sharpe-maximizing loss
+ turnover penalty instead of MSE/IC.

Hypothesis: A small model can learn regime-dependent signal weights
(e.g., more reversal in volatile periods) that a fixed linear blend cannot.
Key difference from exp6: using proven alpha signals as inputs, not raw features.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, C, H, L, V,
    TRANSACTION_COST_BPS, TARGET_LEVERAGE,
    load_data, evaluate_sharpe,
)

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

data = load_data(device="cpu")
ohlcv = data["ohlcv"]
tradeable_idx = data["tradeable_indices"]
N_trade = data["num_tradeable"]
D = ohlcv.shape[0]
train_end = data["train_end_idx"]
fwd_returns = data["forward_returns"]

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

print("Computing alpha signals...")

# 3-factor beta-neutral
betas_spy = compute_betas(daily_ret, spy_ret, 42)
betas_qqq = compute_betas(daily_ret, qqq_ret, 42)
betas_iwm = compute_betas(daily_ret, iwm_ret, 42)
bn_3f = daily_ret - betas_spy * spy_ret.unsqueeze(1) - 0.3 * betas_qqq * qqq_ret.unsqueeze(1) - 0.15 * betas_iwm * iwm_ret.unsqueeze(1)

# Alpha signals (D, N_trade) — our best proven signals
sig_rev5 = cross_zscore(simple_momentum(log_close, 5))
sig_ram21 = cross_zscore(risk_adj_momentum(bn_3f, 21, skip=5))
sig_ram63 = cross_zscore(risk_adj_momentum(bn_3f, 63, skip=5))

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

# Regime features (D, F_regime) — same for all assets
cs_disp = daily_ret.std(dim=1)
disp_ma42 = torch.zeros(D)
for t in range(42, D):
    disp_ma42[t] = cs_disp[t - 42:t].mean()
disp_ma42[:42] = cs_disp[:42].mean()
disp_ratio = cs_disp / disp_ma42.clamp(min=1e-8)

# Realized market vol
mkt_vol21 = torch.zeros(D)
for t in range(21, D):
    mkt_vol21[t] = spy_ret[t-21:t].std() * (252 ** 0.5)
mkt_vol21[:21] = spy_ret[:21].std() * (252 ** 0.5)

# Stack features
# Per-asset: (D, N_trade, 5) — rev5, ram21, ram63, accel, ur
asset_sigs = torch.stack([sig_rev5, sig_ram21, sig_ram63, sig_accel, sig_ur], dim=-1)
asset_sigs[~torch.isfinite(asset_sigs)] = 0.0
asset_sigs.clamp_(-5, 5)
F_asset = asset_sigs.shape[-1]

# Regime: (D, 2) — dispersion, market vol
regime_feats = torch.stack([disp_ratio, mkt_vol21 / 0.2], dim=-1)  # normalize vol
regime_feats[~torch.isfinite(regime_feats)] = 0.0
regime_feats.clamp_(-5, 5)
F_regime = regime_feats.shape[-1]

print(f"Asset signals: {F_asset}, Regime features: {F_regime}")

# ---------------------------------------------------------------------------
# Model: tiny network, takes per-asset signals + regime -> score
# ---------------------------------------------------------------------------

class SignalCombiner(nn.Module):
    def __init__(self, f_asset, f_regime, hidden=32):
        super().__init__()
        # Regime-conditional weights
        self.regime_net = nn.Sequential(
            nn.Linear(f_regime, hidden),
            nn.Tanh(),
            nn.Linear(hidden, f_asset),  # output: per-signal weights
        )
        # Direct linear (fallback)
        self.linear = nn.Linear(f_asset, 1, bias=False)
        # Initialize linear to approximate our known-good weights
        with torch.no_grad():
            # rev5=-0.50, ram21=0.20, ram63=0.30, accel=-0.10, ur=0.12
            self.linear.weight.copy_(torch.tensor([[-0.50, 0.20, 0.30, -0.10, 0.12]]))

    def forward(self, asset_signals, regime):
        """
        asset_signals: (B, N_trade, F_asset)
        regime: (B, F_regime)
        returns: (B, N_trade)
        """
        # Regime-dependent signal weights
        w = self.regime_net(regime)  # (B, F_asset)
        w = w.unsqueeze(1)  # (B, 1, F_asset)

        # Combine: base linear + regime-conditional
        base = self.linear(asset_signals).squeeze(-1)  # (B, N_trade)
        conditional = (asset_signals * w).sum(dim=-1)  # (B, N_trade)

        return base + 0.3 * conditional  # blend


model = SignalCombiner(F_asset, F_regime, hidden=32).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# ---------------------------------------------------------------------------
# Sharpe-like loss: maximize portfolio return / risk
# ---------------------------------------------------------------------------

BATCH_SIZE = 64  # consecutive days

def sharpe_loss(scores, fwd_ret):
    """
    scores: (B, N_trade) — raw position scores
    fwd_ret: (B, N_trade) — forward returns
    Returns: negative Sharpe-like loss
    """
    # Normalize to target leverage
    abs_sum = scores.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    weights = scores * (TARGET_LEVERAGE / abs_sum)

    # Portfolio returns
    port_ret = (weights * fwd_ret).sum(dim=1)  # (B,)

    # Sharpe proxy
    mean_ret = port_ret.mean()
    std_ret = port_ret.std().clamp(min=1e-8)

    return -mean_ret / std_ret

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train_indices = list(range(42, train_end + 1))

print(f"Training on {len(train_indices)} days, budget={TIME_BUDGET}s")

t_train_start = time.time()
total_train_time = 0
step = 0
smooth_loss = 0

model.train()
while True:
    torch.cuda.synchronize()
    t0 = time.time()

    # Sample consecutive batch for Sharpe calculation
    start_idx = np.random.choice(len(train_indices) - BATCH_SIZE)
    batch_days = train_indices[start_idx:start_idx + BATCH_SIZE]

    a_batch = asset_sigs[batch_days].to(device)  # (B, N_trade, F_asset)
    r_batch = regime_feats[batch_days].to(device)  # (B, F_regime)
    tgt = fwd_returns[batch_days].to(device)  # (B, N_trade)

    scores = model(a_batch, r_batch)
    loss = sharpe_loss(scores, tgt)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    progress = min(total_train_time / TIME_BUDGET, 1.0)
    lr = 1e-3 * max(0.01, 1.0 - progress)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.step()
    loss_f = loss.item()

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_train_time += dt

    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased = smooth_loss / (1 - ema ** (step + 1))

    if step % 200 == 0:
        remaining = max(0, TIME_BUDGET - total_train_time)
        print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased:.4f} | "
              f"lr: {lr:.2e} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    step += 1
    if step > 5 and total_train_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Build predict_fn: precompute + EMA smooth
# ---------------------------------------------------------------------------

model.eval()
print("Precomputing predictions...")

all_scores = torch.zeros(D, N_trade, device=device)
with torch.no_grad():
    for t in range(42, D):
        a = asset_sigs[t:t+1].to(device)
        r = regime_feats[t:t+1].to(device)
        all_scores[t] = model(a, r).squeeze(0)

# EMA smooth
ema_alpha = 0.035
smooth = torch.zeros(D, N_trade, device=device)
smooth[42] = all_scores[42]
for t in range(43, D):
    smooth[t] = ema_alpha * all_scores[t] + (1 - ema_alpha) * smooth[t - 1]

# Also compare with pure rule-based (v18) as fallback
dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)
rule_signal = (-0.50 * sig_rev5 + 0.20 * sig_ram21 + 0.30 * sig_ram63
               - 0.10 * sig_accel + 0.12 * sig_ur) * dr
rule_smooth = torch.zeros(D, N_trade)
rule_smooth[0] = rule_signal[0]
for t in range(1, D):
    rule_smooth[t] = 0.035 * rule_signal[t] + (1 - 0.035) * rule_smooth[t - 1]
rule_smooth = rule_smooth.to(device)

# Test both and ensemble
for name, sig in [("nn_only", smooth), ("rule_only", rule_smooth),
                   ("ens_50_50", 0.5 * smooth + 0.5 * rule_smooth),
                   ("ens_30_70", 0.3 * smooth + 0.7 * rule_smooth),
                   ("ens_70_30", 0.7 * smooth + 0.3 * rule_smooth)]:
    def _pred(oh, meta, _s=sig):
        return _s[meta["today_idx"]]
    res = evaluate_sharpe(_pred, data, device=device)
    print(f"  {name:15s}: Sharpe={res['sharpe_ratio']:+.4f}  Return={res['total_return']:+.4f}  "
          f"MaxDD={res['max_drawdown']:+.4f}  Turn={res['avg_turnover']:.4f}")

# Use the rule-based as final (compare to see if NN helped)
def predict_fn(ohlcv_history, meta):
    return smooth[meta["today_idx"]]

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
print(f"num_steps:        {step}")
