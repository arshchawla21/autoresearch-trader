#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

You must implement the function:

    trade(prices, current_idx, symbols) -> list[float]

Parameters
----------
prices : dict[str, pd.DataFrame]
    Mapping from symbol name to a DataFrame of OHLCV data with a
    DatetimeIndex. Contains ALL data from the start of the dataset up
    to and including the current 5-minute candle. This means:
      - During evaluation (days 31–60), you always have at least 30
        trading days of history to work with for training, feature
        engineering, etc.
      - Both tradeable symbols and market indicators (^VIX, ^TNX, GLD,
        TLT, DX-Y.NYB) are included — use them freely for signals.

current_idx : int
    The integer index of the current candle in the aligned close-price
    matrix. Mostly useful for the harness; you probably won't need it.

symbols : list[str]
    Ordered list of tradeable symbols. Your returned weight vector must
    match this ordering. Currently 15 symbols:
    ["AAPL","MSFT","NVDA","AMZN","TSLA","META","GOOGL","JPM","XOM","UNH",
     "SPY","QQQ","IWM","XLF","XLE"]

Returns
-------
list[float]
    A weight vector of length len(symbols). Each weight is the fraction
    of the portfolio to allocate to that asset.
      - Positive = long, Negative = short.
      - Constraint: sum(|weights|) <= 1.0 (net leverage ≤ 1).
        If you exceed 1.0, the harness rescales automatically.
      - Example: [0.0, 0.25, 0.5, 0.0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        means 25% long MSFT, 50% long NVDA, 25% short TSLA, rest flat.

Notes
-----
- You have full freedom: ML models, technical indicators, statistical
  methods, heuristics, regime detection, anything.
- You can import any library (numpy, pandas, sklearn, torch, etc.).
- You can define helper functions, classes, global state, caches.
- The function is called once per 5-min candle during evaluation
  (~78 calls/day × ~22 eval days ≈ 1,700 calls). Keep it fast enough
  to finish in reasonable time, but don't over-optimize prematurely.
- Global state persists across calls within a single backtest run,
  so you can cache computed features, trained models, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: VIX Regime-Based Momentum/Mean-Reversion
# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis: High volatility (VIX > threshold) favors mean-reversion,
# while low volatility favors momentum continuation.
# We trade the strongest/weakest momentum signals based on short-term returns.


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    VIX regime-based momentum/mean-reversion strategy.

    Uses VIX to detect volatility regime:
    - High VIX (> 20): Trade mean-reversion (fade strong moves)
    - Low VIX (<= 20): Trade momentum (follow trends)

    Selects top/bottom 3 performers based on 5-bar returns.
    """
    # Get current VIX level
    vix_sym = "^VIX"
    if vix_sym not in prices or len(prices[vix_sym]) < 5:
        # Fallback to flat if no VIX data
        return [0.0] * len(symbols)

    vix_df = prices[vix_sym]
    current_vix = vix_df["close"].iloc[-1]

    # VIX threshold for regime switching
    VIX_THRESHOLD = 20.0
    high_vol_regime = current_vix > VIX_THRESHOLD

    # Compute 5-bar returns for all tradeable symbols
    returns = {}
    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 6:
            returns[sym] = 0.0
            continue
        df = prices[sym]
        recent_close = df["close"].iloc[-1]
        past_close = df["close"].iloc[-6]  # 5 bars ago
        ret = (recent_close - past_close) / past_close if past_close > 0 else 0.0
        returns[sym] = ret

    # Sort by returns
    sorted_syms = sorted(returns.keys(), key=lambda s: returns[s])

    # Initialize weights
    weights = {sym: 0.0 for sym in symbols}

    if high_vol_regime:
        # Mean-reversion: long worst performers, short best performers
        # Take bottom 3 (weakest) as longs
        longs = sorted_syms[:3]
        # Take top 3 (strongest) as shorts
        shorts = sorted_syms[-3:]

        for sym in longs:
            weights[sym] = 1.0 / 6.0
        for sym in shorts:
            weights[sym] = -1.0 / 6.0
    else:
        # Momentum: long best performers, short worst performers
        # Take top 3 (strongest) as longs
        longs = sorted_syms[-3:]
        # Take bottom 3 (weakest) as shorts
        shorts = sorted_syms[:3]

        for sym in longs:
            weights[sym] = 1.0 / 6.0
        for sym in shorts:
            weights[sym] = -1.0 / 6.0

    # Return weights in the order of symbols list
    return [weights[sym] for sym in symbols]
