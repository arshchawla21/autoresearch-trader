#!/usr/bin/env python3
"""
train.py — v35-percentile-rank-mr
=================================
Hypothesis: A self-adapting signal needs no warmup fit and no fixed
threshold. Rank the current 4-bar return against the last 500 4-bar
returns (~5 days). When current move is in the extreme 5% tails, it is
a regime-relative outlier. Fade extreme dips in an uptrend and extreme
rallies in a downtrend. Structurally distinct from z-score because it
uses the empirical distribution rather than a gaussian assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

RANK_LB = 500
RANK_LOW = 0.05
RANK_HIGH = 0.95
MOVE_LB = 4
TREND_LB = 96


def _pct_rank(arr: np.ndarray, x: float) -> float:
    return float((arr <= x).mean())


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    closes = pair["close"].values.astype(float)
    if len(closes) < max(RANK_LB + MOVE_LB, TREND_LB) + 5:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    last = closes[-1]
    # trend gate
    trend = last / closes[-1 - TREND_LB] - 1.0

    # 4-bar returns window
    recent = closes[-(RANK_LB + MOVE_LB + 1):]
    rets = np.log(recent[MOVE_LB:] / recent[:-MOVE_LB])
    cur = rets[-1]
    hist = rets[-RANK_LB - 1 : -1]  # exclude current
    if len(hist) < RANK_LB:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    rk = _pct_rank(hist, cur)

    d = 0
    if trend > 0 and rk <= RANK_LOW:
        d = 1
    elif trend < 0 and rk >= RANK_HIGH:
        d = -1
    return {"direction": d, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
