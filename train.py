"""
v8-ls-1bar: Equal-weight L/S, 1-bar lookback (open-to-close of prev bar).

Hypothesis: faster 1-bar signal captures more immediate reversions.
"""

import time
import numpy as np
from prepare import O, H, L, C, evaluate

t_start = time.time()


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    closes = ohlcv[:, tidx, C]
    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 2:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    opens = ohlcv[bar_idx, tidx, O].numpy()
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (prev_close - prev_open) / np.maximum(prev_open, 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) < 4:
        return []

    ranked = valid[np.argsort(momentum[valid])]
    n = len(ranked)
    half = n // 2
    longs = ranked[:half]
    shorts = ranked[half:]

    total_positions = len(longs) + len(shorts)
    w = 1.0 / total_positions

    orders = []
    for idx in longs:
        op = float(opens[idx])
        atr = float(avg_atr[idx])
        sl_dist = max(atr * 0.7, 0.001) * op
        tp_dist = max(atr * 1.0, 0.0015) * op
        orders.append({
            "ticker": tickers[idx], "direction": "long", "weight": w,
            "stop_loss": op - sl_dist, "take_profit": op + tp_dist,
        })

    for idx in shorts:
        op = float(opens[idx])
        atr = float(avg_atr[idx])
        sl_dist = max(atr * 0.7, 0.001) * op
        tp_dist = max(atr * 1.0, 0.0015) * op
        orders.append({
            "ticker": tickers[idx], "direction": "short", "weight": w,
            "stop_loss": op + sl_dist, "take_profit": op - tp_dist,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f}")
