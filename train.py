"""
v23-weighted-multi: Weighted multi-stock - heavy on #1, lighter on #2-3.

Hypothesis: combine the best of concentrated (v20) and diversified (v21).
Give 60% weight to the biggest mover, 20% each to #2 and #3. This keeps
the signal strength of concentration while adding some diversification.
Zero fees means the extra trades are free.
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
    n = min(3, len(valid))
    top_indices = valid[np.argsort(-abs_mom)[:n]]

    weights = [0.6, 0.2, 0.2][:n]
    # Renormalize if fewer than 3
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    orders = []
    for i, idx in enumerate(top_indices):
        ticker = tickers[idx]
        op = float(opens[idx])
        mom = float(momentum[idx])
        atr = float(avg_atr[idx])

        direction = "short" if mom > 0 else "long"

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
            "weight": weights[i],
            "stop_loss": sl,
            "take_profit": tp,
        })

    return orders


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
