#!/usr/bin/env python3
"""
train.py — v7-nikkei-lead-tokyo
===============================
Hypothesis: during the Tokyo session (00-05 UTC), Nikkei 225 direction
leads USD/JPY — when Nikkei rallies, BoJ / risk-on flows push JPY weaker
(USD/JPY up). Trade USD/JPY only in Tokyo hours, aligned with the 2h
Nikkei return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

NK_LB = 8                  # 2h Nikkei lookback
NK_THRESHOLD = 0.0015      # 15bps Nikkei move required
TOKYO_HOURS = set(range(0, 6))  # 00:00–05:59 UTC


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    nk = prices.get("^N225")
    if pair is None or nk is None or len(pair) == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    ts = pair.index[-1]
    hour_utc = ts.tz_convert("UTC").hour if ts.tzinfo is not None else ts.hour
    if hour_utc not in TOKYO_HOURS:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    nk_close = nk["close"]
    if len(nk_close) < NK_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    now = float(nk_close.iloc[-1])
    prev = float(nk_close.iloc[-1 - NK_LB])
    if prev <= 0 or np.isnan(now) or np.isnan(prev):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    nk_ret = now / prev - 1.0

    if nk_ret > NK_THRESHOLD:
        direction = 1
    elif nk_ret < -NK_THRESHOLD:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
