#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 5: Sector Rotation with Relative Strength

Hypothesis: Different market sectors show persistent relative strength.
We rank sectors by risk-adjusted momentum and rotate into the strongest,
while using VIX to modulate overall exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_sharpe_of_returns(returns: np.ndarray) -> float:
    """Compute Sharpe-like ratio of a return series."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret > 0:
        return mean_ret / std_ret
    return 0.0


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Sector rotation based on risk-adjusted momentum.

    1. Compute 10-bar returns for each symbol
    2. Rank by return/volatility ratio (risk-adjusted momentum)
    3. Long top 3, short bottom 3 (or flat if in between)
    4. Scale by VIX level (reduce exposure when high)
    """
    # Get VIX level
    vix_level = 20.0
    if "^VIX" in prices and len(prices["^VIX"]) > 0:
        vix_level = float(prices["^VIX"]["close"].iloc[-1])

    # Compute risk-adjusted momentum for each symbol
    momentum_scores = {}

    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 11:
            momentum_scores[sym] = 0.0
            continue

        df = prices[sym]
        closes = df["close"].values

        # Get last 10 closes
        recent = closes[-10:]

        # Compute returns
        rets = np.diff(recent) / recent[:-1]

        # Risk-adjusted return
        if len(rets) > 1 and np.std(rets) > 0:
            score = np.mean(rets) / np.std(rets)
        else:
            score = 0.0

        momentum_scores[sym] = score

    # Sort by momentum score
    sorted_syms = sorted(symbols, key=lambda s: momentum_scores[s])

    # Initialize weights
    weights = {sym: 0.0 for sym in symbols}

    # Long top 3 performers
    longs = sorted_syms[-3:]
    # Short bottom 3 performers
    shorts = sorted_syms[:3]

    # Equal weight within longs/shorts
    for sym in longs:
        weights[sym] = 1.0 / 6.0
    for sym in shorts:
        weights[sym] = -1.0 / 6.0

    # Apply VIX-based scaling
    vix_scale = 1.0
    if vix_level > 30:
        vix_scale = 0.3
    elif vix_level > 25:
        vix_scale = 0.5
    elif vix_level > 20:
        vix_scale = 0.7

    # Return weights in order
    return [weights[sym] * vix_scale for sym in symbols]
