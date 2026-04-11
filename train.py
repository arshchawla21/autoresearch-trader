#!/usr/bin/env python3
"""
train.py — v1-dxy-catchup
=========================
Hypothesis: USD/JPY lags the broader USD index on short horizons.
If USD_CHF (DXY proxy) has moved more than USD/JPY in the last hour,
USD/JPY will close the gap in the direction DXY already moved. This is
a pure cross-asset lead-lag play, no internal-price model.
"""

from __future__ import annotations

import pandas as pd

TP_PIPS = 12.0
SL_PIPS = 8.0

LAG_BARS = 4                 # 1h lookback
SPREAD_THRESHOLD = 0.0010    # DXY must lead USD/JPY by >= 10bps


def _pct_ret(df: pd.DataFrame | None, n: int) -> float | None:
    if df is None:
        return None
    closes = df["close"].dropna()
    if len(closes) < n + 1:
        return None
    prev = float(closes.iloc[-1 - n])
    if prev <= 0:
        return None
    return float(closes.iloc[-1]) / prev - 1.0


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    jpy_ret = _pct_ret(prices.get("JPY=X"), LAG_BARS)
    dxy_ret = _pct_ret(prices.get("DX-Y.NYB"), LAG_BARS)
    if jpy_ret is None or dxy_ret is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    spread = dxy_ret - jpy_ret
    if spread > SPREAD_THRESHOLD:
        direction = 1
    elif spread < -SPREAD_THRESHOLD:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
