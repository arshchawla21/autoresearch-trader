#!/usr/bin/env python3
"""
train.py — v15-confluence-v11-v4
================================
Hypothesis: v11 (trend-aligned dip MR, 42.3% win) + v4 (gold divergence
fade, 49.2% win at 10/10) are the two best signals so far, and they
measure different things. Take the intersection: only enter when both
agree on direction. Structural confluence filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
TREND_LB = 96
LB_GOLD = 4
MIN_MOVE = 0.0005


def _v11_signal(pair: pd.DataFrame) -> int:
    if len(pair) < TREND_LB + 1:
        return 0
    closes = pair["close"].values.astype(float)
    last = float(closes[-1])
    win = closes[-Z_LB:]
    sd = float(win.std(ddof=1))
    if sd <= 0:
        return 0
    z = (last - float(win.mean())) / sd
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return 0
    trend = last / prev - 1.0
    if trend > 0 and z < -Z_ENTRY:
        return 1
    if trend < 0 and z > Z_ENTRY:
        return -1
    return 0


def _v4_signal(pair: pd.DataFrame, gold: pd.DataFrame | None) -> int:
    if gold is None or len(pair) < LB_GOLD + 1:
        return 0
    pc = pair["close"]
    p_now = float(pc.iloc[-1])
    p_prev = float(pc.iloc[-1 - LB_GOLD])
    if p_prev <= 0:
        return 0
    gc = gold["close"]
    try:
        g_now = float(gc.asof(pair.index[-1]))
        g_prev = float(gc.asof(pair.index[-1 - LB_GOLD]))
    except Exception:
        return 0
    if np.isnan(g_now) or np.isnan(g_prev) or g_prev <= 0:
        return 0
    jr = np.log(p_now / p_prev)
    gr = np.log(g_now / g_prev)
    if jr > MIN_MOVE and gr > MIN_MOVE:
        return -1  # fade JPY up (co-move anomaly)
    if jr < -MIN_MOVE and gr < -MIN_MOVE:
        return 1
    return 0


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    s11 = _v11_signal(pair)
    s4 = _v4_signal(pair, prices.get("GC=F"))

    if s11 == 0 or s4 == 0 or s11 != s4:
        direction = 0
    else:
        direction = s11
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
