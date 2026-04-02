#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 10: Optimized VIX Regime Momentum

The original VIX regime strategy achieved Sharpe 3.44.
This optimized version:
- Uses VIX level to select momentum vs mean-reversion
- Takes top 3 / bottom 3 performers based on 5-bar returns
- Adds modest yield curve signal (^TNX trend)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Optimized VIX regime-based momentum/mean-reversion strategy.

    Core logic:
    - VIX > 20: Mean-reversion regime (fade strong moves)
    - VIX <= 20: Momentum regime (follow trends)
    - Long/short top/bottom 3 performers based on 5-bar returns
    """
    # Get current VIX level
    vix_sym = "^VIX"
    if vix_sym not in prices or len(prices[vix_sym]) < 5:
        return [0.0] * len(symbols)

    vix_df = prices[vix_sym]
    current_vix = float(vix_df["close"].iloc[-1])

    # VIX regime
    VIX_THRESHOLD = 20.0
    high_vol_regime = current_vix > VIX_THRESHOLD

    # Optional: Check 10Y yield trend for additional context
    tnx_sym = "^TNX"
    tnx_trend = 0.0
    if tnx_sym in prices and len(prices[tnx_sym]) >= 5:
        tnx_values = prices[tnx_sym]["close"].values
        if tnx_values[-5] > 0:
            tnx_trend = (tnx_values[-1] - tnx_values[-5]) / tnx_values[-5]

    # Compute 5-bar returns for all tradeable symbols
    returns = {}
    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 6:
            returns[sym] = 0.0
            continue
        df = prices[sym]
        recent_close = float(df["close"].iloc[-1])
        past_close = float(df["close"].iloc[-6])
        if past_close > 0:
            ret = (recent_close - past_close) / past_close
        else:
            ret = 0.0
        returns[sym] = ret

    # Sort by returns
    sorted_syms = sorted(returns.keys(), key=lambda s: returns[s])

    # Initialize weights
    weights = {sym: 0.0 for sym in symbols}

    # Select positions based on regime
    if high_vol_regime:
        # Mean-reversion: long worst, short best
        for sym in sorted_syms[:3]:
            weights[sym] = 1.0 / 6.0
        for sym in sorted_syms[-3:]:
            weights[sym] = -1.0 / 6.0
    else:
        # Momentum: long best, short worst
        for sym in sorted_syms[-3:]:
            weights[sym] = 1.0 / 6.0
        for sym in sorted_syms[:3]:
            weights[sym] = -1.0 / 6.0

    # Yield curve adjustment: if yields rising rapidly, reduce exposure
    exposure_scale = 1.0
    if tnx_trend > 0.02:  # >2% move in yields
        exposure_scale = 0.8

    # Return weights in order
    return [weights[sym] * exposure_scale for sym in symbols]
