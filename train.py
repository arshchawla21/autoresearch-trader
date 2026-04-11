#!/usr/bin/env python3
"""
train.py — Strategy v4: Z-Score Mean Reversion
================================================
Hypothesis: on a 15m timeframe USD/JPY over-extends frequently within the
session and reverts. Both momentum experiments (v2, v3) lost — the opposite
direction should therefore pay. Compute a 20-bar rolling z-score of the
close, fade extremes: short when the pair is >1.5σ above its mean, long
when <-1.5σ below. This is the canonical "fade the spike" trade.
"""

from __future__ import annotations

import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LOOKBACK = 20           # 5 hours of 15m bars
Z_ENTRY = 1.5             # sigma threshold


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < Z_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].dropna()
    if len(closes) < Z_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    window = closes.iloc[-Z_LOOKBACK:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    z = (float(closes.iloc[-1]) - mu) / sd

    if z > Z_ENTRY:
        direction = -1    # fade the rally
    elif z < -Z_ENTRY:
        direction = 1     # fade the dip
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
