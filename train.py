"""
v5-momentum-follow: FOLLOW momentum instead of fading. All stocks, every bar.

Hypothesis: maybe this market regime trends more than it reverts.
Trade with the direction of 2-bar momentum on all stocks.
"""

import time
import numpy as np
from prepare import O, H, L, C, V, evaluate

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
    if bar_idx < 3:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    opens = ohlcv[bar_idx, tidx, O].numpy()
    past_close = ohlcv[bar_idx - 2, tidx, C].numpy()
    curr_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (curr_close - past_close) / np.maximum(past_close, 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) == 0:
        return []

    abs_mom = np.abs(momentum[valid])
    total = abs_mom.sum()
    if total < 1e-10:
        return []

    weights = abs_mom / total
    orders = []

    for i, idx in enumerate(valid):
        w = float(weights[i])
        if w < 0.005:
            continue

        ticker = tickers[idx]
        op = float(opens[idx])
        mom = float(momentum[idx])
        atr = float(avg_atr[idx])

        # FOLLOW momentum (not fade)
        direction = "long" if mom > 0 else "short"

        sl_dist = max(atr * 0.7, 0.001) * op
        tp_dist = max(atr * 1.0, 0.0015) * op

        if direction == "long":
            sl = op - sl_dist
            tp = op + tp_dist
        else:
            sl = op + sl_dist
            tp = op - tp_dist

        orders.append({
            "ticker": ticker,
            "direction": direction,
            "weight": w,
            "stop_loss": sl,
            "take_profit": tp,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f}")
