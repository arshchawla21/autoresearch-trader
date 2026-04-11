#!/usr/bin/env python3
"""
train.py — v40-squeeze-fade
===========================
Hypothesis: v39 showed squeeze-breakouts hit only 36.94% — below the
TP=15/SL=10 random baseline. That means the breakouts reliably FAIL:
price breaks the 20-bar range during vol compression, then snaps back.
Fade the break instead — go opposite direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

BB_LB = 20
BB_MED_LB = 100
BRK_LB = 20


def _bb_width(closes: np.ndarray, n: int) -> float:
    if len(closes) < n:
        return float("nan")
    w = closes[-n:]
    sd = float(w.std(ddof=1))
    mu = float(w.mean())
    if mu <= 0:
        return float("nan")
    return (4.0 * sd) / mu  # ~2 sigma full width normalized


def _bb_width_series(closes: np.ndarray, n: int, out_len: int) -> np.ndarray:
    arr = np.full(out_len, np.nan)
    for i in range(n - 1, len(closes)):
        w = closes[i - n + 1 : i + 1]
        sd = w.std(ddof=1)
        mu = w.mean()
        if mu > 0:
            arr[i - (len(closes) - out_len)] = (4.0 * sd) / mu
    return arr


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    closes = pair["close"].values.astype(float)
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    n = len(closes)
    need = BB_LB + BB_MED_LB + 5
    if n < need:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # current BB width and its 100-bar median
    widths = np.full(BB_MED_LB, np.nan)
    for k in range(BB_MED_LB):
        end = n - k
        start = end - BB_LB
        if start < 0:
            continue
        w = closes[start:end]
        sd = w.std(ddof=1)
        mu = w.mean()
        if mu > 0:
            widths[k] = (4.0 * sd) / mu
    widths = widths[~np.isnan(widths)]
    if len(widths) < 20:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    cur = widths[0]
    med = float(np.median(widths))
    in_squeeze = cur < med

    if not in_squeeze:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # break of 20-bar high/low by last close (excluding current bar from the range)
    prior_hi = float(highs[-BRK_LB - 1 : -1].max())
    prior_lo = float(lows[-BRK_LB - 1 : -1].min())
    last = float(closes[-1])

    if last > prior_hi:
        return {"direction": -1, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if last < prior_lo:
        return {"direction": 1, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
