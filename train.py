"""
v4-meanrev-optimized: Optimized mean-reversion with VIX filter and adaptive stops.

Hypothesis: mean-reversion works better with (1) more diversification (5 stocks),
(2) tighter TP relative to SL for higher win rate, (3) VIX filter to avoid
trading during panic periods, and (4) longer lookback for stronger signals.
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


def build_strategy(train_data):
    """Compute per-stock volatility stats and VIX index."""
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    midx = train_data["macro_indices"].numpy()

    closes = ohlcv[:, tidx, C]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]

    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)

    # Find VIX index in macro tickers
    all_tickers = train_data["all_tickers"]
    vix_idx = None
    for i, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_idx = i
            break

    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
        "vix_idx": vix_idx,
    }


def generate_orders(strategy, data, bar_idx):
    """Mean-reversion with VIX filter and tighter risk management."""
    lookback = 8
    if bar_idx < lookback + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]
    vix_idx = strategy["vix_idx"]

    # VIX filter: don't trade if VIX is very high (panic)
    if vix_idx is not None:
        vix_close = float(ohlcv[bar_idx - 1, vix_idx, C])
        if vix_close > 30:
            return []

    opens = ohlcv[bar_idx, tidx, O].numpy()
    past_close = ohlcv[bar_idx - lookback, tidx, C].numpy()
    current_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (current_close - past_close) / np.maximum(past_close, 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) == 0:
        return []

    # Pick top 5 stocks with strongest moves to fade
    abs_mom = np.abs(momentum[valid])
    top_k = min(5, len(valid))
    top_indices = valid[np.argsort(-abs_mom)[:top_k]]

    orders = []
    weight = 1.0 / top_k

    for idx in top_indices:
        ticker = tickers[idx]
        op = float(opens[idx])
        mom = float(momentum[idx])
        atr = float(avg_atr[idx])

        if abs(mom) < 0.0015:
            continue

        direction = "short" if mom > 0 else "long"

        # Tighter: 1.2x ATR stop, 1.0x ATR take-profit (higher win rate)
        sl_dist = max(atr * 1.2, 0.002) * op
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
            "weight": weight,
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
