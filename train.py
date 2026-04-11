#!/usr/bin/env python3
"""
train.py — Strategy v13: Stop-Run Reversal
============================================
Hypothesis: when a 15m bar pokes a new N-bar high/low AND closes back
inside the prior range, that's a "stop run" — the market ran stops above
a swing level and reversed. This signature is distinct from pure z-score
MR: it requires a specific candle shape (wick out, close back in). If the
edge is real, fading these bars should pay consistently.
"""

from __future__ import annotations

import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

LOOKBACK = 12      # 3 hours of 15m bars to define the range
WICK_RATIO = 0.4   # min fraction of bar range that is wick on the breakout side


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < LOOKBACK + 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    df = pair.dropna(subset=["close"]).tail(LOOKBACK + 1)
    if len(df) < LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    prior = df.iloc[:-1]
    curr = df.iloc[-1]

    prior_high = float(prior["high"].max())
    prior_low = float(prior["low"].min())

    c_open = float(curr["open"])
    c_high = float(curr["high"])
    c_low = float(curr["low"])
    c_close = float(curr["close"])
    c_range = c_high - c_low
    if c_range <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # Bullish-break-failed: high > prior_high BUT close back below prior_high
    if c_high > prior_high and c_close < prior_high:
        upper_wick = c_high - max(c_open, c_close)
        if upper_wick / c_range >= WICK_RATIO:
            return {"direction": -1, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # Bearish-break-failed: low < prior_low BUT close back above prior_low
    if c_low < prior_low and c_close > prior_low:
        lower_wick = min(c_open, c_close) - c_low
        if lower_wick / c_range >= WICK_RATIO:
            return {"direction": 1, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
