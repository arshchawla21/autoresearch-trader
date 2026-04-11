#!/usr/bin/env python3
"""
train.py — v79-vwap-fade
========================
Hypothesis: Session VWAP is a mean-reversion anchor distinct from
rolling z-score. Compute an intraday VWAP from 07:00 UTC onwards
(London session onward). When price deviates >1.5σ from VWAP, fade
the deviation back toward VWAP. Different statistical reference than
a rolling window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PIP = 0.01
TP_MIN, TP_MAX = 6.0, 20.0
SL_MIN, SL_MAX = 4.0, 14.0
ATR_LB = 20

VWAP_START_HOUR = 7
DEV_SIGMA = 1.5


def _atr_pips(pair: pd.DataFrame, n: int) -> float:
    if len(pair) < n + 1:
        return 15.0
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    closes = pair["close"].values.astype(float)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    if len(tr) < n:
        return 15.0
    return float(tr[-n:].mean()) / PIP


def _session_dev(pair: pd.DataFrame) -> tuple[float, int]:
    """Return (deviation_in_sigmas, trend_sign) for intraday VWAP since 07:00 UTC."""
    ts_now = pair.index[-1]
    if ts_now.tzinfo is None:
        return 0.0, 0
    day_start = ts_now.normalize() + pd.Timedelta(hours=VWAP_START_HOUR)
    if ts_now < day_start:
        return 0.0, 0
    window = pair[(pair.index >= day_start) & (pair.index <= ts_now)]
    if len(window) < 8:
        return 0.0, 0
    highs = window["high"].values.astype(float)
    lows = window["low"].values.astype(float)
    closes = window["close"].values.astype(float)
    vols = window["volume"].values.astype(float) if "volume" in window.columns else np.ones(len(window))
    if vols.sum() <= 0:
        vols = np.ones(len(window))
    tp = (highs + lows + closes) / 3.0
    vwap = float((tp * vols).sum() / vols.sum())
    # Use close std as sigma estimate over the session
    sd = float(closes.std(ddof=1)) if len(closes) > 2 else 0.0
    if sd <= 0:
        return 0.0, 0
    dev = (closes[-1] - vwap) / sd
    # Session trend: sign of last minus first
    trend_sign = 1 if closes[-1] > closes[0] else -1
    return dev, trend_sign


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": 15.0, "sl_pips": 10.0}

    atr = _atr_pips(pair, ATR_LB)
    tp = max(TP_MIN, min(TP_MAX, 1.5 * atr))
    sl = max(SL_MIN, min(SL_MAX, 1.0 * atr))

    dev, trend_sign = _session_dev(pair)
    direction = 0
    # Fade deviations beyond threshold back toward VWAP
    if dev > DEV_SIGMA:
        direction = -1
    elif dev < -DEV_SIGMA:
        direction = 1
    return {"direction": direction, "tp_pips": tp, "sl_pips": sl}
