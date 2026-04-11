#!/usr/bin/env python3
"""
train.py — Strategy v5: RSI + Bollinger Double Confirmation
============================================================
Hypothesis: z-score mean reversion worked because USD/JPY 15m reverts from
extremes. But z-score triggers on any std deviation, which is noisy. A
stricter "overbought / oversold" gate using both Bollinger Bands AND a
classical RSI should increase hit rate at the cost of fewer trades. This
tests whether the mean-reversion edge is structural (strong trigger pays
for the lost volume) or opportunistic (frequent weak triggers are
what make it work).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

BB_LOOKBACK = 20
BB_K = 2.0

RSI_LOOKBACK = 14
RSI_HI = 70.0
RSI_LO = 30.0


def _rsi(closes: np.ndarray, n: int) -> float:
    diffs = np.diff(closes)
    gains = np.clip(diffs, 0.0, None)
    losses = np.clip(-diffs, 0.0, None)
    if len(gains) < n:
        return 50.0
    avg_gain = float(np.mean(gains[-n:]))
    avg_loss = float(np.mean(losses[-n:]))
    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < max(BB_LOOKBACK, RSI_LOOKBACK) + 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].dropna().values
    if len(closes) < BB_LOOKBACK + 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    window = closes[-BB_LOOKBACK:]
    mu = float(np.mean(window))
    sd = float(np.std(window, ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    px = float(closes[-1])
    upper = mu + BB_K * sd
    lower = mu - BB_K * sd
    rsi = _rsi(closes, RSI_LOOKBACK)

    if px >= upper and rsi >= RSI_HI:
        direction = -1
    elif px <= lower and rsi <= RSI_LO:
        direction = 1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
