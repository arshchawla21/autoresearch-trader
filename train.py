#!/usr/bin/env python3
"""
train.py — v14-jpy-trend-follow
===============================
Hypothesis: v11 looked like "dip in uptrend" but may just be trend
persistence at the 24h horizon. Test pure JPY 24h trend-follow — enter
every bar in the direction of the 96-bar return. If momentum persists,
win rate should beat 44% with 15/10 brackets.
"""

from __future__ import annotations

import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

TREND_LB = 96
TREND_MIN = 0.0005  # 5bps


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < TREND_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].values
    last = float(closes[-1])
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    trend = last / prev - 1.0

    if trend > TREND_MIN:
        direction = 1
    elif trend < -TREND_MIN:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
