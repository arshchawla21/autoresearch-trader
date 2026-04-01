"""
v24-xsect-volfilter: Cross-sectional L/S MR with volatility-based filtering.

Building on v23 (best so far). Add filters:
- Skip low-volatility bars (weak signal, noise-dominated)
- Volume spike filter (higher volume = stronger conviction)
- Require minimum absolute return magnitude to avoid noise trades
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
    """Compute volatility stats from training data for filtering."""
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    # Compute average bar range per stock (for filtering)
    ranges = []
    for t in range(1, T):
        hi = ohlcv[t, tidx, H]
        lo = ohlcv[t, tidx, L]
        cl = ohlcv[t - 1, tidx, C]
        safe_cl = np.maximum(np.abs(cl), 1e-8)
        bar_range = (hi - lo) / safe_cl
        ranges.append(bar_range)
    avg_range = np.nanmedian(np.stack(ranges), axis=0)  # per-stock median range

    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "n_stocks": n_stocks,
        "avg_range": avg_range,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < LOOKBACK + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    n_stocks = strategy["n_stocks"]
    avg_range = strategy["avg_range"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]
    vols = ohlcv_np[:bar_idx, tidx_np, V]
    highs = ohlcv_np[:bar_idx, tidx_np, H]
    lows = ohlcv_np[:bar_idx, tidx_np, L]

    # Recent return to fade
    ret = (closes[-1] - closes[-1 - LOOKBACK]) / np.maximum(np.abs(closes[-1 - LOOKBACK]), 1e-8)

    # Current bar range relative to average (is this stock "active"?)
    last_range = (highs[-1] - lows[-1]) / np.maximum(np.abs(closes[-2]), 1e-8)
    range_ratio = last_range / np.maximum(avg_range, 1e-8)

    # Volume spike (last bar vs 10-bar average)
    avg_vol = np.mean(vols[-10:], axis=0)
    vol_ratio = vols[-1] / np.maximum(avg_vol, 1e-8)

    # Filter: valid stocks with meaningful moves
    valid_mask = ((opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
                  & (range_ratio > 0.5)   # not dead/illiquid bars
                  & (np.abs(ret) > 0.001))  # at least 0.1% move to fade
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < N_LONG + N_SHORT:
        # Fallback: relax filters
        valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < N_LONG + N_SHORT:
            return []

    # Rank by return: lowest → long, highest → short
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
