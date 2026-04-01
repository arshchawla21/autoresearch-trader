"""
v7-ls-momentum-weighted: L/S portfolio, weight proportional to signal strength.

Hypothesis: stocks with stronger momentum deserve larger positions.
Weighting by |momentum| concentrates capital on the strongest signals
while maintaining market neutrality.
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
    if len(valid) < 4:
        return []

    # Split into longs (negative momentum) and shorts (positive momentum)
    mom_valid = momentum[valid]
    longs = valid[mom_valid < 0]
    shorts = valid[mom_valid > 0]

    if len(longs) == 0 or len(shorts) == 0:
        return []

    # Weight by absolute momentum within each side, then split 50/50
    long_abs = np.abs(momentum[longs])
    short_abs = np.abs(momentum[shorts])
    long_weights = 0.5 * long_abs / long_abs.sum()
    short_weights = 0.5 * short_abs / short_abs.sum()

    orders = []
    for i, idx in enumerate(longs):
        op = float(opens[idx])
        atr = float(avg_atr[idx])
        sl_dist = max(atr * 0.7, 0.001) * op
        tp_dist = max(atr * 1.0, 0.0015) * op
        orders.append({
            "ticker": tickers[idx], "direction": "long",
            "weight": float(long_weights[i]),
            "stop_loss": op - sl_dist, "take_profit": op + tp_dist,
        })

    for i, idx in enumerate(shorts):
        op = float(opens[idx])
        atr = float(avg_atr[idx])
        sl_dist = max(atr * 0.7, 0.001) * op
        tp_dist = max(atr * 1.0, 0.0015) * op
        orders.append({
            "ticker": tickers[idx], "direction": "short",
            "weight": float(short_weights[i]),
            "stop_loss": op + sl_dist, "take_profit": op - tp_dist,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f}")
