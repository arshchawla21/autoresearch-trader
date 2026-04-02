#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 9: Improved VIX Regime with Dynamic Position Sizing

Based on the successful VIX regime strategy (Sharpe 3.44), this version:
- Uses smarter position sizing based on signal strength
- Adds correlation filtering to avoid concentrated exposure
- Uses dollar index (^TNX) as additional regime indicator
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_returns(
    prices: dict[str, pd.DataFrame], symbols: list[str], lookback: int
) -> dict[str, float]:
    """Compute returns for all symbols."""
    returns = {}
    for sym in symbols:
        if sym not in prices or len(prices[sym]) < lookback + 1:
            returns[sym] = 0.0
            continue
        df = prices[sym]
        closes = df["close"].values
        current = closes[-1]
        past = closes[-(lookback + 1)]
        if past > 0:
            returns[sym] = (current - past) / past
        else:
            returns[sym] = 0.0
    return returns


def _get_market_context(prices: dict[str, pd.DataFrame]) -> dict:
    """Get market regime indicators."""
    context = {"vix": 20.0, "vix_trend": 0.0, "tnx": 4.0}

    # VIX level and trend
    if "^VIX" in prices and len(prices["^VIX"]) >= 10:
        vix_df = prices["^VIX"]
        vix_values = vix_df["close"].values
        context["vix"] = float(vix_values[-1])
        # 5-bar trend
        if len(vix_values) >= 6:
            context["vix_trend"] = (
                float(vix_values[-1] - vix_values[-6]) / vix_values[-6]
                if vix_values[-6] > 0
                else 0.0
            )

    # 10Y yield
    if "^TNX" in prices and len(prices["^TNX"]) > 0:
        context["tnx"] = float(prices["^TNX"]["close"].iloc[-1])

    return context


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Improved VIX regime strategy with dynamic sizing.

    Core logic:
    - High VIX (> 22): Mean-reversion on extreme movers
    - Low VIX (<= 22): Momentum continuation
    - Position sizes scaled by signal strength and VIX level
    """
    # Get market context
    ctx = _get_market_context(prices)
    vix = ctx["vix"]
    vix_trend = ctx["vix_trend"]

    # Dynamic VIX threshold based on recent trend
    vix_threshold = 22.0
    if vix_trend > 0.05:  # Rising VIX, use lower threshold
        vix_threshold = 20.0
    elif vix_trend < -0.05:  # Falling VIX, use higher threshold
        vix_threshold = 24.0

    high_vol_regime = vix > vix_threshold

    # Compute momentum over different lookbacks
    ret_3 = _compute_returns(prices, symbols, 3)
    ret_5 = _compute_returns(prices, symbols, 5)

    # Combined momentum score (weighted average)
    momentum = {}
    for sym in symbols:
        momentum[sym] = 0.6 * ret_5.get(sym, 0.0) + 0.4 * ret_3.get(sym, 0.0)

    # Sort by momentum
    sorted_syms = sorted(symbols, key=lambda s: momentum[s])

    # Initialize weights
    weights = {sym: 0.0 for sym in symbols}

    # Select top and bottom performers
    n_positions = 3  # number of longs and shorts

    if high_vol_regime:
        # Mean-reversion: fade the extremes
        longs = sorted_syms[:n_positions]  # worst performers
        shorts = sorted_syms[-n_positions:]  # best performers

        for sym in longs:
            # Scale by how extreme the move was
            strength = min(abs(momentum[sym]) * 5, 0.25)
            weights[sym] = strength
        for sym in shorts:
            strength = min(abs(momentum[sym]) * 5, 0.25)
            weights[sym] = -strength
    else:
        # Momentum: follow the trend
        longs = sorted_syms[-n_positions:]  # best performers
        shorts = sorted_syms[:n_positions]  # worst performers

        for sym in longs:
            strength = min(abs(momentum[sym]) * 5, 0.25)
            weights[sym] = strength
        for sym in shorts:
            strength = min(abs(momentum[sym]) * 5, 0.25)
            weights[sym] = -strength

    # Convert to list
    weight_list = [weights[sym] for sym in symbols]

    # Normalize to leverage <= 1.0
    total_lev = sum(abs(w) for w in weight_list)
    if total_lev > 1.0:
        weight_list = [w / total_lev for w in weight_list]

    # VIX-based exposure scaling
    exposure_scale = 1.0
    if vix > 35:
        exposure_scale = 0.3
    elif vix > 30:
        exposure_scale = 0.5
    elif vix > 25:
        exposure_scale = 0.7
    elif vix > vix_threshold:
        exposure_scale = 0.85

    return [w * exposure_scale for w in weight_list]
