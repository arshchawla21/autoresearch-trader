"""
v11-volume-filter: Fade low-volume moves, avoid high-volume moves.

Hypothesis: price moves on low relative volume lack conviction and are more
likely to revert. High-volume moves signal real information and should not
be faded. This is a structural improvement with clear economic rationale.

Uses v9 stops (0.7x ATR SL, 4.0x ATR TP) as the known-good baseline.
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()

    closes = ohlcv[:, tidx, C]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    volumes = ohlcv[:, tidx, V]

    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)

    # Compute median volume per stock for relative volume comparison
    median_vol = np.nanmedian(volumes, axis=0)

    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
        "median_vol": median_vol,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 3:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]
    median_vol = strategy["median_vol"]

    opens = ohlcv[bar_idx, tidx, O].numpy()

    # 1-bar momentum (open-to-close of previous bar)
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    prev_vol = ohlcv[bar_idx - 1, tidx, V].numpy()
    momentum = (prev_close - prev_open) / np.maximum(prev_open, 1e-8)

    # Relative volume: how does last bar's volume compare to median
    rel_vol = prev_vol / np.maximum(median_vol, 1e-8)

    valid = np.where(
        (opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum) & ~np.isnan(rel_vol)
    )[0]
    if len(valid) == 0:
        return []

    # Score: high absolute momentum + low relative volume = best fade candidate
    # Penalize high-volume moves (they're more likely real)
    scores = np.abs(momentum[valid]) / np.maximum(rel_vol[valid], 0.1)

    best = valid[np.argmax(scores)]

    ticker = tickers[best]
    op = float(opens[best])
    mom = float(momentum[best])
    atr = float(avg_atr[best])
    rv = float(rel_vol[best])

    if abs(mom) < 0.0005:
        return []

    # Skip if volume was very high — this move may be real news
    if rv > 2.0:
        return []

    direction = "short" if mom > 0 else "long"

    sl_dist = max(atr * 0.7, 0.001) * op
    tp_dist = max(atr * 4.0, 0.005) * op

    if direction == "long":
        sl = op - sl_dist
        tp = op + tp_dist
    else:
        sl = op + sl_dist
        tp = op - tp_dist

    return [{
        "ticker": ticker,
        "direction": direction,
        "weight": 1.0,
        "stop_loss": sl,
        "take_profit": tp,
    }]


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print("\n---")
    print(f"sharpe_ratio:     {results.get('sharpe_ratio', 0):.4f}")
    print(f"total_return:     {results.get('total_return', 0)*100:.2f}%")
    print(f"max_drawdown:     {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"win_rate:         {results.get('win_rate', 0)*100:.1f}%")
    print(f"avg_trades/bar:   {results.get('avg_daily_trades', 0):.2f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
