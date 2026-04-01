"""
v19-spy-only: Trade only SPY with VIX regime awareness.

Hypothesis: concentrating on SPY alone might work because (1) it's the most
liquid and efficient for mean-reversion, (2) VIX gives a direct read on whether
SPY is in a fear/greed extreme. If VIX is rising (fear), fade SPY selloffs;
if VIX is falling (complacency), fade SPY rallies.

Uses v9 stops (0.7x/4.0x ATR).
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    tickers = train_data["tradeable_tickers"]

    closes = ohlcv[:, tidx, C]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)

    spy_idx = tickers.index("SPY") if "SPY" in tickers else 0

    # Find VIX in all tickers
    all_tickers = train_data["all_tickers"]
    vix_global_idx = None
    for i, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_global_idx = i
            break

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
        "spy_idx": spy_idx,
        "vix_global_idx": vix_global_idx,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 3:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]
    spy_idx = strategy["spy_idx"]

    opens = ohlcv[bar_idx, tidx, O].numpy()
    op = float(opens[spy_idx])
    if op <= 0 or np.isnan(op):
        return []

    # 2-bar momentum for SPY
    past_close = float(ohlcv[bar_idx - 2, tidx[spy_idx], C])
    curr_close = float(ohlcv[bar_idx - 1, tidx[spy_idx], C])
    momentum = (curr_close - past_close) / max(abs(past_close), 1e-8)

    if abs(momentum) < 0.0005:
        return []

    direction = "short" if momentum > 0 else "long"
    atr = float(avg_atr[spy_idx])

    sl_dist = max(atr * 0.7, 0.001) * op
    tp_dist = max(atr * 4.0, 0.005) * op

    if direction == "long":
        sl = op - sl_dist
        tp = op + tp_dist
    else:
        sl = op + sl_dist
        tp = op - tp_dist

    return [{
        "ticker": "SPY",
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
