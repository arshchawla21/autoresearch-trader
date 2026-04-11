#!/usr/bin/env python3
"""
train.py — Strategy v2: DXY Momentum Co-Move
=============================================
Hypothesis: USD/JPY is dominated by the USD leg intraday. If the broad
dollar (DX-Y.NYB) has trended up over the last N 15m bars, the pair should
be biased higher on the next bar — we ride the DXY lead. Symmetric short
when DXY trends down. Stay flat otherwise.
"""

from __future__ import annotations

import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

# DXY lookback in 15m bars. 8 bars = 2 hours of dollar direction.
DXY_LOOKBACK = 8
# Minimum absolute DXY return (fraction) required to take a trade.
# 0.05% over 2 hours on DXY is a meaningful directional move.
DXY_THRESHOLD = 0.0005


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    dxy = prices.get("DX-Y.NYB")
    if dxy is None or len(dxy) < DXY_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = dxy["close"].dropna()
    if len(closes) < DXY_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    recent = float(closes.iloc[-1])
    past = float(closes.iloc[-DXY_LOOKBACK - 1])
    if past <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    dxy_ret = (recent - past) / past
    if dxy_ret > DXY_THRESHOLD:
        direction = 1
    elif dxy_ret < -DXY_THRESHOLD:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
