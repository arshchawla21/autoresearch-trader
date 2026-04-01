"""
v16-ls-asym: Equal-weight L/S with asymmetric stops (TP > SL).

Uses the v6 L/S framework with 25% bar range SL but 35% bar range TP.
When TP hits, we win 40% more than we lose on SL hit.
With ~55% WR, this asymmetry compounds into higher returns.

Also: use the 50% range version alongside for comparison.
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
    bar_range_pct = (highs - lows) / np.maximum(closes, 1e-8)
    median_range = np.nanmedian(bar_range_pct, axis=0)
    print(f"  Median 15m bar range: {np.mean(median_range)*100:.3f}%")
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "median_range": median_range,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 3:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    median_range = strategy["median_range"]

    opens = ohlcv[bar_idx, tidx, O].numpy()
    past_close = ohlcv[bar_idx - 2, tidx, C].numpy()
    curr_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (curr_close - past_close) / np.maximum(past_close, 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) < 4:
        return []

    ranked = valid[np.argsort(momentum[valid])]
    n = len(ranked)
    half = n // 2
    longs = ranked[:half]
    shorts = ranked[half:]

    total = len(longs) + len(shorts)
    w = 1.0 / total

    orders = []

    for idx in longs:
        op = float(opens[idx])
        mr = float(median_range[idx])
        sl_dist = max(mr * 0.25, 0.0004) * op
        tp_dist = max(mr * 0.35, 0.0006) * op  # TP 40% wider than SL
        orders.append({
            "ticker": tickers[idx], "direction": "long", "weight": w,
            "stop_loss": op - sl_dist, "take_profit": op + tp_dist,
        })

    for idx in shorts:
        op = float(opens[idx])
        mr = float(median_range[idx])
        sl_dist = max(mr * 0.25, 0.0004) * op
        tp_dist = max(mr * 0.35, 0.0006) * op
        orders.append({
            "ticker": tickers[idx], "direction": "short", "weight": w,
            "stop_loss": op + sl_dist, "take_profit": op - tp_dist,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f}")
