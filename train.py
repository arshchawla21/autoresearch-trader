"""
Autoresearch-trader training script. Single-GPU, single-file.

This is the ONLY file you modify. You have total freedom:
  - Feature engineering from raw OHLCV (prepare.py gives you raw data)
  - Any strategy: ML model, algorithmic rules, hybrid, ensemble
  - Any architecture, loss, optimizer, training procedure
  - Or no training at all (pure rule-based strategies are valid)

The only contract: at the end, call evaluate_sharpe(predict_fn, data)
where predict_fn(ohlcv_history, meta) -> (num_tradeable,) position scores.

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
    TIME_BUDGET, O, H, L, C, V,
    load_data, evaluate_sharpe,
)

# ---------------------------------------------------------------------------
# Feature engineering (from raw OHLCV — build whatever you want here)
# ---------------------------------------------------------------------------

LOOKBACK = 60  # days of history per sample

def compute_features(ohlcv, tradeable_indices, macro_indices):
    """
    Compute features from raw OHLCV tensor.

    Args:
        ohlcv: (D, N_all, 5) raw data
        tradeable_indices: LongTensor of tradeable ticker positions
        macro_indices: LongTensor of macro ticker positions

    Returns:
        asset_features: (D, N_tradeable, F_asset) per-asset features
        macro_features: (D, F_macro) cross-asset features
    """
    close_t = ohlcv[:, tradeable_indices, C]    # (D, N_trade)
    high_t = ohlcv[:, tradeable_indices, H]
    low_t = ohlcv[:, tradeable_indices, L]
    vol_t = ohlcv[:, tradeable_indices, V]

    D, N = close_t.shape
    feats = []

    # Log returns at multiple horizons
    log_close = torch.log(close_t.clamp(min=1e-8))
    for lag in [1, 5, 21]:
        ret = torch.zeros(D, N)
        ret[lag:] = log_close[lag:] - log_close[:-lag]
        feats.append(ret)

    # Realized volatility (21-day rolling std of daily returns)
    daily_ret = torch.zeros(D, N)
    daily_ret[1:] = log_close[1:] - log_close[:-1]
    vol_21 = torch.zeros(D, N)
    for t in range(21, D):
        vol_21[t] = daily_ret[t-21:t].std(dim=0) * (252 ** 0.5)
    feats.append(vol_21)

    # RSI(14) — normalized to [-1, 1]
    delta = torch.zeros(D, N)
    delta[1:] = close_t[1:] - close_t[:-1]
    gain = delta.clamp(min=0)
    loss_val = (-delta).clamp(min=0)
    alpha = 1.0 / 14
    avg_gain = torch.zeros(D, N)
    avg_loss = torch.zeros(D, N)
    for t in range(1, D):
        avg_gain[t] = alpha * gain[t] + (1 - alpha) * avg_gain[t-1]
        avg_loss[t] = alpha * loss_val[t] + (1 - alpha) * avg_loss[t-1]
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - 100 / (1 + rs)) / 50 - 1  # [-1, 1]
    feats.append(rsi)

    # Bollinger band position
    for t_bb in range(20, D):
        pass  # skip full BB for simplicity in baseline
    sma20 = torch.zeros(D, N)
    std20 = torch.zeros(D, N)
    for t in range(20, D):
        window = close_t[t-20:t]
        sma20[t] = window.mean(dim=0)
        std20[t] = window.std(dim=0)
    bb_pos = (close_t - sma20) / (2 * std20 + 1e-10)
    bb_pos.clamp_(-3, 3)
    feats.append(bb_pos)

    # Volume ratio (log of today / 20-day avg)
    vol_ma20 = torch.zeros(D, N)
    for t in range(20, D):
        vol_ma20[t] = vol_t[t-20:t].mean(dim=0)
    vol_ratio = torch.log((vol_t + 1) / (vol_ma20 + 1))
    feats.append(vol_ratio)

    # Range (high-low / close)
    hlrange = (high_t - low_t) / (close_t + 1e-10)
    feats.append(hlrange)

    asset_features = torch.stack(feats, dim=-1)  # (D, N, F)
    asset_features[~torch.isfinite(asset_features)] = 0.0
    asset_features.clamp_(-5, 5)

    # Macro features (VIX, yields, etc.)
    macro_feats = []
    if len(macro_indices) > 0:
        macro_close = ohlcv[:, macro_indices, C]  # (D, N_macro)
        log_macro = torch.log(macro_close.clamp(min=1e-8))
        # Level (log-scaled)
        macro_feats.append(log_macro / 4)
        # 1-day returns
        macro_ret = torch.zeros_like(log_macro)
        macro_ret[1:] = log_macro[1:] - log_macro[:-1]
        macro_feats.append(macro_ret)

    if macro_feats:
        macro_features = torch.cat(macro_feats, dim=-1)  # (D, F_macro)
    else:
        macro_features = torch.zeros(D, 1)

    macro_features[~torch.isfinite(macro_features)] = 0.0
    macro_features.clamp_(-5, 5)

    return asset_features, macro_features


# ---------------------------------------------------------------------------
# Model (replace with anything you want)
# ---------------------------------------------------------------------------

class TradingModel(nn.Module):
    """
    Baseline: shared MLP over flattened lookback window + macro context.
    """
    def __init__(self, num_assets, num_asset_features, num_macro_features,
                 lookback=LOOKBACK, hidden=128, dropout=0.1):
        super().__init__()
        self.num_assets = num_assets
        self.lookback = lookback

        asset_in = lookback * num_asset_features
        macro_in = lookback * num_macro_features

        self.asset_enc = nn.Sequential(
            nn.Linear(asset_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.macro_enc = nn.Sequential(
            nn.Linear(macro_in, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, asset_feat, macro_feat):
        """
        asset_feat: (B, lookback, A, F_asset)
        macro_feat: (B, lookback, F_macro)
        returns: (B, A)
        """
        B, T, A, F = asset_feat.shape
        x_a = asset_feat.permute(0, 2, 1, 3).reshape(B * A, T * F)
        x_a = self.asset_enc(x_a).view(B, A, -1)

        x_m = macro_feat.reshape(B, -1)
        x_m = self.macro_enc(x_m)
        x_m = x_m.unsqueeze(1).expand(-1, A, -1)

        x = torch.cat([x_a, x_m], dim=-1)
        return self.head(x).squeeze(-1)  # (B, A)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN = 128
DROPOUT = 0.1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.3

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

data = load_data(device="cpu")
print(f"Assets: {data['num_tradeable']} tradeable, {data['num_all']} total")
print(f"Train: 0–{data['train_end_idx']}  Test: {data['test_start_idx']}–{data['ohlcv'].shape[0]-2}")

# Feature engineering
print("Computing features from raw OHLCV...")
asset_features, macro_features = compute_features(
    data["ohlcv"], data["tradeable_indices"], data["macro_indices"]
)
F_asset = asset_features.shape[-1]
F_macro = macro_features.shape[-1]
print(f"Features: {F_asset} per asset, {F_macro} macro")

# Build training samples: windows of LOOKBACK days
train_end = data["train_end_idx"]
train_indices = list(range(LOOKBACK, train_end + 1))
fwd_returns = data["forward_returns"]  # (D, N_tradeable)

# Model
model = TradingModel(
    num_assets=data["num_tradeable"],
    num_asset_features=F_asset,
    num_macro_features=F_macro,
    lookback=LOOKBACK,
    hidden=HIDDEN,
    dropout=DROPOUT,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
model = torch.compile(model, dynamic=False)

print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        return (1.0 - progress) / WARMDOWN_RATIO

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_train_start = time.time()
total_train_time = 0
smooth_loss = 0
step = 0

model.train()
while True:
    torch.cuda.synchronize()
    t0 = time.time()

    # Sample a random batch of training windows
    batch_idx = np.random.choice(train_indices, size=BATCH_SIZE, replace=True)
    a_batch = torch.stack([asset_features[i - LOOKBACK:i] for i in batch_idx]).to(device)
    m_batch = torch.stack([macro_features[i - LOOKBACK:i] for i in batch_idx]).to(device)
    tgt = torch.stack([fwd_returns[i] for i in batch_idx]).to(device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = model(a_batch, m_batch)
        loss = F.mse_loss(scores.float(), tgt.float())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    progress = min(total_train_time / TIME_BUDGET, 1.0)
    lrm = get_lr(progress)
    for g in optimizer.param_groups:
        g["lr"] = LR * lrm

    optimizer.step()
    loss_f = loss.item()

    if math.isnan(loss_f) or loss_f > 1e6:
        print("FAIL"); exit(1)

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_train_time += dt

    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased = smooth_loss / (1 - ema ** (step + 1))

    if step % 20 == 0:
        remaining = max(0, TIME_BUDGET - total_train_time)
        print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased:.6f} | "
              f"lr: {LR*lrm:.2e} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    step += 1
    if step > 5 and total_train_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Build predict_fn for evaluation
# ---------------------------------------------------------------------------

model.eval()

# Precompute full feature tensors (eval uses the same features)
_af_full = asset_features.to(device)
_mf_full = macro_features.to(device)

def predict_fn(ohlcv_history, meta):
    """
    Called by evaluate_sharpe for each test day.
    ohlcv_history: (T, N_all, 5) — raw OHLCV up to today.
    We ignore it here and use our precomputed features indexed by today_idx.
    """
    t = meta["today_idx"]
    if t < LOOKBACK:
        return torch.zeros(data["num_tradeable"], device=device)
    a = _af_full[t - LOOKBACK:t].unsqueeze(0)  # (1, LB, A, F)
    m = _mf_full[t - LOOKBACK:t].unsqueeze(0)  # (1, LB, F_m)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = model(a, m)
    return scores.squeeze(0).float()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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