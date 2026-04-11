#!/usr/bin/env python3
"""
train.py — v4-gold-jpy-divergence
=================================
Hypothesis: USD/JPY and gold (XAU/USD) are structurally inversely correlated
(USD strength drives JPY up / gold down). When they move in the *same*
direction intraday, it's an anomaly that tends to resolve by USD/JPY
reverting. Fade USD/JPY's current direction when gold confirms divergence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 10.0
SL_PIPS = 10.0

LB = 4                 # 1h lookback
MIN_JPY_MOVE = 0.0005  # 5bps
MIN_GOLD_MOVE = 0.0005


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    gold = prices.get("GC=F")
    if pair is None or gold is None or len(pair) < LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    p_now = float(pair["close"].iloc[-1])
    p_prev = float(pair["close"].iloc[-1 - LB])
    if p_prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    idx_now = pair.index[-1]
    idx_prev = pair.index[-1 - LB]
    gc = gold["close"]
    try:
        g_now = float(gc.asof(idx_now))
        g_prev = float(gc.asof(idx_prev))
    except Exception:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if np.isnan(g_now) or np.isnan(g_prev) or g_prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    jpy_ret = np.log(p_now / p_prev)
    gld_ret = np.log(g_now / g_prev)

    # Anomaly: both moved same direction with meaningful magnitude.
    if jpy_ret > MIN_JPY_MOVE and gld_ret > MIN_GOLD_MOVE:
        direction = -1          # fade JPY up
    elif jpy_ret < -MIN_JPY_MOVE and gld_ret < -MIN_GOLD_MOVE:
        direction = 1           # fade JPY down
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
