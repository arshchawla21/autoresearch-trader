#!/usr/bin/env python3
"""
train.py — Strategy Implementation (USD/JPY 15m forex)
========================================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy v1: Random Baseline
----------------------------
At every 15m bar, if flat, flip a 25% biased coin:
  - 75% stay flat
  - 12.5% go long
  - 12.5% go short
With TP = 15 pips, SL = 10 pips on USD/JPY. That produces ~1 entry attempt
per hour on average (4 bars × 25% = 1 trade/hr expectation while flat).
This is the random floor every real strategy must beat.
"""

from __future__ import annotations

import random

import pandas as pd


# Baseline TP / SL — every strategy in this project uses fixed-size trades
# and only decides direction. TP/SL levels may be tuned across experiments.
TP_PIPS = 15.0
SL_PIPS = 10.0

# Fixed RNG so the random baseline is reproducible across runs.
_RNG = random.Random(42)


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    """
    Called on every 15m bar when the bot is flat.

    Returns a dict:
        {"direction": -1 | 0 | 1, "tp_pips": float, "sl_pips": float}

    direction  1 → open long USD/JPY at this bar's close
    direction -1 → open short USD/JPY at this bar's close
    direction  0 → stay flat
    """
    r = _RNG.random()
    if r < 0.125:
        direction = 1
    elif r < 0.25:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
