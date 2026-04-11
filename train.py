#!/usr/bin/env python3
"""
train.py — v17-gold-div-trend-gate
==================================
Hypothesis: v15 intersection (v11 ∩ v4) got 43.6% win, positive return,
but only 1.3 trades/day — the z-score requirement in v11 was the binding
constraint. Keep the trend direction gate but drop the z-score, so v4's
gold divergence fires freely as long as its direction aligns with the
24h JPY trend. Should triple trade count at similar quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

TREND_LB = 96
LB_SHORT = 4
MIN_MOVE = 0.0005


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    gold = prices.get("GC=F")
    if pair is None or gold is None or len(pair) < TREND_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    pc = pair["close"]
    closes = pc.values.astype(float)
    last = float(closes[-1])
    prev_trend = float(closes[-1 - TREND_LB])
    if prev_trend <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    trend = last / prev_trend - 1.0

    # v4: 4-bar gold divergence
    p_prev = float(closes[-1 - LB_SHORT])
    gc = gold["close"]
    try:
        g_now = float(gc.asof(pair.index[-1]))
        g_prev = float(gc.asof(pair.index[-1 - LB_SHORT]))
    except Exception:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if np.isnan(g_now) or np.isnan(g_prev) or g_prev <= 0 or p_prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    jr = np.log(last / p_prev)
    gr = np.log(g_now / g_prev)

    gold_sig = 0
    if jr > MIN_MOVE and gr > MIN_MOVE:
        gold_sig = -1   # co-move anomaly, fade JPY up
    elif jr < -MIN_MOVE and gr < -MIN_MOVE:
        gold_sig = 1

    # Direction gate: must agree with 24h trend direction
    if gold_sig == 1 and trend > 0:
        direction = 1
    elif gold_sig == -1 and trend < 0:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
