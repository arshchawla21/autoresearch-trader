#!/usr/bin/env python3
"""
train.py — Strategy v3: 10Y Yield Momentum
============================================
Hypothesis: USD/JPY has the tightest empirical correlation with US 10Y
yields of any major pair because BoJ is pinned near zero while the Fed
isn't. Short-term changes in ^TNX should lead USD/JPY intraday — long
when yields are rising, short when they're falling.
"""

from __future__ import annotations

import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

TNX_LOOKBACK = 6          # ~90 minutes of yield data
TNX_THRESHOLD = 0.002     # 0.2% relative move in the yield index


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    tnx = prices.get("^TNX")
    if tnx is None or len(tnx) < TNX_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = tnx["close"].dropna()
    if len(closes) < TNX_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    recent = float(closes.iloc[-1])
    past = float(closes.iloc[-TNX_LOOKBACK - 1])
    if past <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    yield_ret = (recent - past) / past
    if yield_ret > TNX_THRESHOLD:
        direction = 1
    elif yield_ret < -TNX_THRESHOLD:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
