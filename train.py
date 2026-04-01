"""
v25-weighted+asymm: Cross-sectional L/S MR with magnitude weighting + asymmetric SL/TP.

Two improvements over v23:
1. Weight positions by return magnitude (bigger deviation = more weight)
2. Asymmetric SL/TP: wider TP (0.7%) vs SL (0.5%) to let winners run
"""

import time
import numpy as np
import torch
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

SL_PCT = 0.005   # 0.5% stop
TP_PCT = 0.007   # 0.7% take profit (let winners run)
N_LONG = 2
N_SHORT = 2
LOOKBACK = 5


def build_strategy(train_data):
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < LOOKBACK + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    ret = (closes[-1] - closes[-1 - LOOKBACK]) / np.maximum(np.abs(closes[-1 - LOOKBACK]), 1e-8)

    valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < N_LONG + N_SHORT:
        return []

    valid_rets = ret[valid_idx]
    sorted_indices = np.argsort(valid_rets)

    long_picks = valid_idx[sorted_indices[:N_LONG]]
    short_picks = valid_idx[sorted_indices[-N_SHORT:]]

    # Magnitude-based weighting within each side
    long_mags = np.abs(ret[long_picks])
    short_mags = np.abs(ret[short_picks])

    total_mag = long_mags.sum() + short_mags.sum()
    if total_mag < 1e-10:
        # Equal weight fallback
        w_long = np.full(N_LONG, 0.5 / N_LONG)
        w_short = np.full(N_SHORT, 0.5 / N_SHORT)
    else:
        w_long = 0.5 * long_mags / long_mags.sum()
        w_short = 0.5 * short_mags / short_mags.sum()

    orders = []
    for i, idx in enumerate(long_picks):
        op = float(opens[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": float(w_long[i]),
            "stop_loss": op * (1 - SL_PCT),
            "take_profit": op * (1 + TP_PCT),
        })

    for i, idx in enumerate(short_picks):
        op = float(opens[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "short",
            "weight": float(w_short[i]),
            "stop_loss": op * (1 + SL_PCT),
            "take_profit": op * (1 - TP_PCT),
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% "
          f"dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% "
          f"trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
