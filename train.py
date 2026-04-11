#!/usr/bin/env python3
"""
train.py — v16-triple-vote
==========================
Hypothesis: v15 proved confluence of 2 diverse signals lifts win rate
into profitable territory (+0.22%, 43.6% win) but only 1.3 trades/day.
Add a third independent signal — Nikkei-JPY divergence fade — and take
2-of-3 majority vote. Three structurally distinct bets (trend-aligned
pullback, gold co-move anomaly, equity-FX decoupling) should produce
~3–5× more trades while maintaining quality through voting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
TREND_LB = 96
LB_SHORT = 4
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


def _gold_div_signal(pair: pd.DataFrame, gold: pd.DataFrame | None) -> int:
    if gold is None or len(pair) < LB_SHORT + 1:
        return 0
    pc = pair["close"]
    p_now = float(pc.iloc[-1])
    p_prev = float(pc.iloc[-1 - LB_SHORT])
    if p_prev <= 0:
        return 0
    gc = gold["close"]
    try:
        g_now = float(gc.asof(pair.index[-1]))
        g_prev = float(gc.asof(pair.index[-1 - LB_SHORT]))
    except Exception:
        return 0
    if np.isnan(g_now) or np.isnan(g_prev) or g_prev <= 0:
        return 0
    jr = np.log(p_now / p_prev)
    gr = np.log(g_now / g_prev)
    if jr > MIN_MOVE and gr > MIN_MOVE:
        return -1
    if jr < -MIN_MOVE and gr < -MIN_MOVE:
        return 1
    return 0


def _nikkei_div_signal(pair: pd.DataFrame, nk: pd.DataFrame | None) -> int:
    """
    Nikkei-JPY divergence. Normally risk-on → Nikkei up + USD/JPY up (positive
    correlation). When they diverge (opposite directions), fade USD/JPY's move.
    """
    if nk is None or len(pair) < LB_SHORT + 1:
        return 0
    pc = pair["close"]
    p_now = float(pc.iloc[-1])
    p_prev = float(pc.iloc[-1 - LB_SHORT])
    if p_prev <= 0:
        return 0
    nkc = nk["close"]
    try:
        n_now = float(nkc.asof(pair.index[-1]))
        n_prev = float(nkc.asof(pair.index[-1 - LB_SHORT]))
    except Exception:
        return 0
    if np.isnan(n_now) or np.isnan(n_prev) or n_prev <= 0:
        return 0
    jr = np.log(p_now / p_prev)
    nr = np.log(n_now / n_prev)
    if jr > MIN_MOVE and nr < -MIN_MOVE:
        return -1  # JPY rallied while Nikkei fell → fade JPY up
    if jr < -MIN_MOVE and nr > MIN_MOVE:
        return 1
    return 0


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    votes = [
        _v11_signal(pair),
        _gold_div_signal(pair, prices.get("GC=F")),
        _nikkei_div_signal(pair, prices.get("^N225")),
    ]
    long_v = sum(1 for v in votes if v == 1)
    short_v = sum(1 for v in votes if v == -1)

    if long_v >= 2 and short_v == 0:
        direction = 1
    elif short_v >= 2 and long_v == 0:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
