#!/usr/bin/env python3
"""
train.py — Strategy v12: Session-Filtered Z-Score Mean Reversion
==================================================================
Hypothesis: v4 worked with 8.7 trades/day but that average hides huge
session variance. Tokyo session (00-08 UTC) is typically quieter and more
range-bound → MR works. London-NY overlap (13-16 UTC) is full of
directional news → MR breaks. Test: run v4's z-score fade ONLY during
specific UTC-hour windows and see if selective trading beats the average.
"""

from __future__ import annotations

import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LOOKBACK = 20
Z_ENTRY = 1.5

# Tokyo + early London window (UTC). Avoid the 13-16 UTC NY-London overlap.
ALLOWED_UTC_HOURS = set(range(0, 13))  # 00:00..12:59 UTC


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < Z_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].dropna()
    if len(closes) < Z_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    now = closes.index[-1]
    hour_utc = now.tz_convert("UTC").hour if now.tzinfo is not None else now.hour
    if hour_utc not in ALLOWED_UTC_HOURS:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    window = closes.iloc[-Z_LOOKBACK:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    z = (float(closes.iloc[-1]) - mu) / sd

    if z > Z_ENTRY:
        direction = -1
    elif z < -Z_ENTRY:
        direction = 1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
