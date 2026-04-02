#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 7: Fading Short-Term Noise with Trend Confirmation

Hypothesis: 1-bar moves are often noise (mean-reverting), but if they
align with medium-term trend, they have higher probability of continuing.
We look for:
- 1-bar momentum (immediate move)
- 15-bar trend (directional bias)
- When they agree -> stronger signal
- When they disagree -> fade the 1-bar move
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
    Fade noise strategy with trend alignment.

    For each symbol:
    - Compute 1-bar return (recent move)
    - Compute 15-bar trend (longer direction)
    - When both same sign: momentum continuation
    - When opposite signs: fade the recent move
    - Position size based on VIX regime
    """
    # Get VIX level
    vix_level = 20.0
    if "^VIX" in prices and len(prices["^VIX"]) > 0:
        vix_level = float(prices["^VIX"]["close"].iloc[-1])

    weights = []

    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 16:
            weights.append(0.0)
            continue

        df = prices[sym]
        closes = df["close"].values

        # 1-bar momentum
        mom1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0

        # 15-bar trend
        trend15 = (closes[-1] - closes[-16]) / closes[-16] if closes[-16] > 0 else 0.0

        # Determine position
        if mom1 * trend15 > 0:
            # Same direction: trend continuation
            signal = np.sign(mom1)
            strength = abs(mom1) + abs(trend15) * 0.5
        elif mom1 * trend15 < 0:
            # Opposite: fade the recent move (mean reversion)
            signal = -np.sign(mom1)
            strength = abs(mom1) * 1.5  # stronger fade signal
        else:
            signal = 0.0
            strength = 0.0

        # Cap position size
        pos = signal * min(strength * 2, 0.15)
        weights.append(pos)

    # Normalize
    total_lev = sum(abs(w) for w in weights)
    if total_lev > 1.0:
        weights = [w / total_lev for w in weights]

    # VIX scaling: reduce exposure in high vol
    vix_scale = 1.0
    if vix_level > 30:
        vix_scale = 0.4
    elif vix_level > 25:
        vix_scale = 0.6
    elif vix_level > 20:
        vix_scale = 0.8

    return [w * vix_scale for w in weights]
