"""
v23-xsect-optimized: Cross-sectional L/S MR with optimized params.

Simple but proven approach:
- Each bar, rank all 28 stocks by their recent return
- Go long the biggest losers (likely to revert up)
- Go short the biggest winners (likely to revert down)
- Equal weight, trade every bar
- Market-neutral by construction

No ML — the cross-sectional spread IS the signal.
"""

import time
import numpy as np
import torch
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

# Trade parameters
SL_PCT = 0.005   # 0.5% stop
TP_PCT = 0.005   # 0.5% take profit
N_LONG = 2       # number of stocks to go long
N_SHORT = 2      # number of stocks to go short
LOOKBACK = 5     # bars of return to fade


def build_strategy(train_data):
    """No training needed — pure rule-based."""
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "n_stocks": len(train_data["tradeable_indices"]),
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < LOOKBACK + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    n_stocks = strategy["n_stocks"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    # Recent return to fade
    ret = (closes[-1] - closes[-1 - LOOKBACK]) / np.maximum(np.abs(closes[-1 - LOOKBACK]), 1e-8)

    # Filter valid stocks
    valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < N_LONG + N_SHORT:
        return []

    # Rank by return: lowest returns → long, highest → short
    valid_rets = ret[valid_idx]
    sorted_indices = np.argsort(valid_rets)

    long_picks = valid_idx[sorted_indices[:N_LONG]]
    short_picks = valid_idx[sorted_indices[-N_SHORT:]]

    total_positions = N_LONG + N_SHORT
    w = 1.0 / total_positions

    orders = []
    for idx in long_picks:
        op = float(opens[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": w,
            "stop_loss": op * (1 - SL_PCT),
            "take_profit": op * (1 + TP_PCT),
        })

    for idx in short_picks:
        op = float(opens[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "short",
            "weight": w,
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
