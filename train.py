#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 3: Multi-Timeframe Momentum with Market Regime Filter

Hypothesis: Combining short-term and medium-term momentum signals
provides more robust trend detection. We use:
- Short-term momentum: 5-bar returns
- Medium-term momentum: 20-bar returns
- Market regime: VIX trend (rising/falling) and 10Y yield direction

Only trade when both timeframes agree on direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_returns(
    prices: dict[str, pd.DataFrame], symbols: list[str], lookback: int
) -> dict[str, float]:
    """Compute returns over lookback periods for all symbols."""
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


def _get_trend(
    prices: dict[str, pd.DataFrame], symbol: str, lookback: int = 10
) -> float:
    """Get recent trend direction (-1 to 1) for an indicator."""
    if symbol not in prices or len(prices[symbol]) < lookback + 1:
        return 0.0
    df = prices[symbol]
    closes = df["close"].values
    recent = closes[-lookback:]
    # Simple linear regression slope normalized by mean
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    mean_price = np.mean(recent)
    if mean_price > 0:
        normalized_slope = slope / mean_price
        # Clamp to [-1, 1] range
        return np.tanh(normalized_slope * 100)
    return 0.0


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Multi-timeframe momentum with market regime filter.

    1. Compute 5-bar (short) and 20-bar (medium) returns
    2. Only take positions when both agree in direction
    3. Position strength = average of short and medium momentum
    4. Scale by VIX trend (reduce size when VIX rising sharply)
    5. Use 10Y yield trend as additional risk-off filter
    """
    # Compute momentum signals
    short_ret = _compute_returns(prices, symbols, lookback=5)
    medium_ret = _compute_returns(prices, symbols, lookback=20)

    # Get market regime indicators
    vix_trend = _get_trend(prices, "^VIX", lookback=10)
    tnx_trend = _get_trend(prices, "^TNX", lookback=10)

    # Current VIX level
    vix_level = 20.0
    if "^VIX" in prices and len(prices["^VIX"]) > 0:
        vix_level = prices["^VIX"]["close"].iloc[-1]

    weights = []
    for sym in symbols:
        s = short_ret.get(sym, 0.0)
        m = medium_ret.get(sym, 0.0)

        # Only trade when both timeframes agree
        if s * m <= 0:  # Different signs or one is zero
            weights.append(0.0)
            continue

        # Average momentum
        avg_mom = (s + m) / 2

        # Position size based on momentum magnitude
        # Scale to reasonable range (-0.15 to 0.15 per position)
        pos_size = np.sign(avg_mom) * min(abs(avg_mom) * 5, 0.15)

        weights.append(pos_size)

    # Normalize to leverage <= 1.0
    total_lev = sum(abs(w) for w in weights)
    if total_lev > 1.0:
        weights = [w / total_lev for w in weights]

    # Regime-based scaling
    scale = 1.0

    # Reduce exposure when VIX is rising (volatility expansion)
    if vix_trend > 0.3:
        scale *= 0.6
    elif vix_trend > 0.1:
        scale *= 0.8

    # Reduce exposure when yields rising rapidly (tightening)
    if tnx_trend > 0.5:
        scale *= 0.7

    # Cap positions when VIX very high (uncertainty)
    if vix_level > 30:
        scale *= 0.5
    elif vix_level > 25:
        scale *= 0.75

    weights = [w * scale for w in weights]

    return weights
