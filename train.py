#!/usr/bin/env python3
"""
train.py — Strategy v6: VIX-Gated Mean Reversion
==================================================
Hypothesis: v4 (z-score MR) worked at Sharpe 4.84. Mean reversion is known
to break down during risk-off shocks where trends run further than normal.
Gate the z-score fade by VIX regime: only take MR trades when VIX is below
its own rolling median (calm regime). When VIX is elevated, stay flat —
don't catch falling knives.
"""

from __future__ import annotations

import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LOOKBACK = 20
Z_ENTRY = 1.5
VIX_LOOKBACK = 96          # ~1 day of 15m bars of VIX
VIX_PCTL = 0.60            # only trade when VIX is below 60th pct of its day


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < Z_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].dropna()
    if len(closes) < Z_LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    window = closes.iloc[-Z_LOOKBACK:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    z = (float(closes.iloc[-1]) - mu) / sd

    # VIX regime gate
    vix = prices.get("^VIX")
    calm_regime = True
    if vix is not None and len(vix) >= VIX_LOOKBACK:
        vix_closes = vix["close"].dropna()
        if len(vix_closes) >= VIX_LOOKBACK:
            recent_vix = float(vix_closes.iloc[-1])
            vix_window = vix_closes.iloc[-VIX_LOOKBACK:]
            pctl_threshold = float(vix_window.quantile(VIX_PCTL))
            calm_regime = recent_vix <= pctl_threshold

    if not calm_regime:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    if z > Z_ENTRY:
        direction = -1
    elif z < -Z_ENTRY:
        direction = 1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
