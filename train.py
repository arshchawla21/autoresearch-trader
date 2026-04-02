#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 2: Multi-Asset Z-Score Statistical Arbitrage with VIX Filter

Hypothesis: Price deviations from short-term moving averages create
mean-reversion opportunities. We scale positions by the z-score of
the deviation, with position sizing modulated by VIX regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Global cache for computed features
_cache: dict = {}


def _compute_zscores(
    prices: dict[str, pd.DataFrame], symbols: list[str], lookback: int = 20
) -> dict[str, float]:
    """Compute z-score of current price vs SMA for each symbol."""
    zscores = {}
    for sym in symbols:
        if sym not in prices or len(prices[sym]) < lookback + 1:
            zscores[sym] = 0.0
            continue
        df = prices[sym]
        closes = df["close"].values

        # Recent window
        recent = closes[-lookback:]
        current = closes[-1]

        sma = np.mean(recent)
        std = np.std(recent)

        if std > 0:
            zscore = (current - sma) / std
        else:
            zscore = 0.0

        zscores[sym] = zscore

    return zscores


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Z-score statistical arbitrage strategy.

    1. Compute z-score of current price vs 20-bar SMA for each symbol
    2. Go short when z-score > 1.0 (overbought), long when z-score < -1.0 (oversold)
    3. Position sizes proportional to z-score magnitude
    4. VIX scaling: reduce exposure when VIX > 25 (high uncertainty)
    """
    # Get VIX for scaling
    vix_sym = "^VIX"
    vix_value = 20.0  # default
    if vix_sym in prices and len(prices[vix_sym]) > 0:
        vix_value = prices[vix_sym]["close"].iloc[-1]

    # Compute z-scores
    zscores = _compute_zscores(prices, symbols, lookback=20)

    # Z-score thresholds
    Z_THRESHOLD = 1.0
    MAX_POS_SIZE = 0.15  # max 15% per position

    weights = []
    for sym in symbols:
        z = zscores.get(sym, 0.0)

        if abs(z) < Z_THRESHOLD:
            # No signal
            weights.append(0.0)
        else:
            # Position inversely proportional to z-score (mean reversion)
            # High positive z -> short, High negative z -> long
            raw_weight = -np.sign(z) * min(abs(z) * 0.1, MAX_POS_SIZE)
            weights.append(raw_weight)

    # Normalize to leverage <= 1.0
    total_leverage = sum(abs(w) for w in weights)
    if total_leverage > 1.0:
        weights = [w / total_leverage for w in weights]

    # VIX-based scaling: reduce exposure in high vol
    if vix_value > 25:
        weights = [w * 0.5 for w in weights]  # 50% reduction
    elif vix_value > 20:
        weights = [w * 0.75 for w in weights]  # 25% reduction

    return weights
