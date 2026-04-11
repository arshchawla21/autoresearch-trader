#!/usr/bin/env python3
"""
train.py — v98-v14-momentum
============================
Hypothesis: v97 (historic MR replay) got 39.45% win rate — BELOW
random (40% at 1.5:1). That's not noise — it's a regime signal.
USD/JPY 15m z-score extremes now CONTINUE, not revert. Flip the
direction: z>+1.5 → LONG, z<-1.5 → SHORT. Same 2-of-3 vote.
"""

from __future__ import annotations

import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LOOKBACK = 20
Z_ENTRY = 1.5

VIX_LOOKBACK = 96
VIX_PCTL = 0.60

ALLOWED_UTC_HOURS = set(range(0, 13))


def _zscore_signal(pair: pd.DataFrame) -> int:
    closes = pair["close"].dropna()
    if len(closes) < Z_LOOKBACK + 1:
        return 0
    window = closes.iloc[-Z_LOOKBACK:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return 0
    z = (float(closes.iloc[-1]) - mu) / sd
    if z > Z_ENTRY:
        return 1
    if z < -Z_ENTRY:
        return -1
    return 0


def _vix_gated_signal(pair: pd.DataFrame, vix: pd.DataFrame | None) -> int:
    base = _zscore_signal(pair)
    if base == 0 or vix is None:
        return 0
    vix_closes = vix["close"].dropna()
    if len(vix_closes) < VIX_LOOKBACK:
        return 0
    vix_window = vix_closes.iloc[-VIX_LOOKBACK:]
    threshold = float(vix_window.quantile(VIX_PCTL))
    if float(vix_closes.iloc[-1]) > threshold:
        return 0
    return base


def _session_signal(pair: pd.DataFrame) -> int:
    closes = pair["close"].dropna()
    if len(closes) < 1:
        return 0
    now = closes.index[-1]
    hour_utc = now.tz_convert("UTC").hour if now.tzinfo is not None else now.hour
    if hour_utc not in ALLOWED_UTC_HOURS:
        return 0
    return _zscore_signal(pair)


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    votes = [
        _zscore_signal(pair),
        _vix_gated_signal(pair, prices.get("^VIX")),
        _session_signal(pair),
    ]
    long_votes = sum(1 for v in votes if v == 1)
    short_votes = sum(1 for v in votes if v == -1)

    if long_votes >= 2 and short_votes == 0:
        direction = 1
    elif short_votes >= 2 and long_votes == 0:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
