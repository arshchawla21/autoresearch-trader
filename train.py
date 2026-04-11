#!/usr/bin/env python3
"""
train.py — v77-london-orb
=========================
Hypothesis: Opening range breakout is a completely different signal
family from v69's pullback MR. The London session open (~07:00 UTC)
has the sharpest directional flow in FX. Compute the range of the
first 4 bars (7:00-8:00 UTC), then on bars 5-20 of the London session,
enter long on a close above the range high, short on a close below
the range low. Bracket with ATR-scaled TP=1.5×ATR, SL=1.0×ATR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PIP = 0.01
TP_MIN, TP_MAX = 8.0, 25.0
SL_MIN, SL_MAX = 6.0, 18.0
ATR_LB = 20

# London session: UTC 07:00 to 15:00. First 4 bars (07:00-08:00) form range.
LONDON_OPEN_HOUR = 7
LONDON_RANGE_BARS = 4
LONDON_TRADE_END = 12  # stop taking new entries after 12:00 UTC


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


def _session_range(pair: pd.DataFrame) -> tuple[float, float] | None:
    """Return (high, low) of today's London opening range (bars 07:00-08:00 UTC)."""
    if len(pair) < LONDON_RANGE_BARS:
        return None
    ts_now = pair.index[-1]
    if ts_now.tzinfo is None:
        return None
    day_start = ts_now.normalize()
    # Opening range starts at LONDON_OPEN_HOUR UTC
    range_start = day_start + pd.Timedelta(hours=LONDON_OPEN_HOUR)
    range_end = range_start + pd.Timedelta(hours=1)  # 4 x 15min = 1h
    window = pair[(pair.index >= range_start) & (pair.index < range_end)]
    if len(window) < LONDON_RANGE_BARS:
        return None
    return float(window["high"].max()), float(window["low"].min())


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": 15.0, "sl_pips": 10.0}

    atr = _atr_pips(pair, ATR_LB)
    tp = max(TP_MIN, min(TP_MAX, 1.5 * atr))
    sl = max(SL_MIN, min(SL_MAX, 1.0 * atr))

    ts = pair.index[-1]
    if ts.tzinfo is None:
        return {"direction": 0, "tp_pips": tp, "sl_pips": sl}
    hour_utc = ts.hour
    # Only trade during london post-range window (08:00 - 12:00 UTC)
    if hour_utc < LONDON_OPEN_HOUR + 1 or hour_utc >= LONDON_TRADE_END:
        return {"direction": 0, "tp_pips": tp, "sl_pips": sl}

    rng = _session_range(pair)
    if rng is None:
        return {"direction": 0, "tp_pips": tp, "sl_pips": sl}
    hi, lo = rng

    c = float(pair["close"].iloc[-1])
    # Need a real breakout margin at least 2 pips past the range
    margin = 2.0 * PIP
    if c > hi + margin:
        direction = 1
    elif c < lo - margin:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": tp, "sl_pips": sl}
