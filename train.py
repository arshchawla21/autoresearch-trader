"""
Autoresearch-trader baseline script.

Baseline: Pure Random Intraday Trading
- No machine learning or indicators.
- Trades roughly 3 times a day (11.5% chance per 15m interval).
- Randomly picks 1 stock.
- Randomly goes long or short.
- Uses fixed 0.5% stop-loss and 1.0% take-profit.
- Allocates 100% of the portfolio to that single trade.

Usage: uv run train.py
"""

import os
import time
import random
import numpy as np

from prepare import O, evaluate

t_start = time.time()

# Set seed for a reproducible baseline
random.seed(42)

def build_strategy(train_data):
    """
    For a random baseline, we don't need to train any models.
    We just return the ticker information so we can pick from them later.
    """
    print("Baseline: Skipping training, returning empty strategy.")
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
    }


def generate_orders(strategy, data, bar_idx):
    """
    Randomly generates trades a few times a day.
    """
    # A standard trading day has 26 15-minute intervals. 
    # If we want to trade ~3 times a day, we have a ~11.5% chance to trade each bar.
    if random.random() > 0.115:
        return []

    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]

    # Get the opening prices for the current interval
    opens = ohlcv[bar_idx, tradeable_idx, O]

    # Filter out any stocks that have missing data (NaN or 0) for this interval
    valid_indices = []
    for i, op in enumerate(opens):
        if not np.isnan(float(op)) and float(op) > 0:
            valid_indices.append(i)

    # If no stocks are trading right now, do nothing
    if not valid_indices:
        return []

    # 1. Pick a random stock
    idx = random.choice(valid_indices)
    ticker = tickers[idx]
    open_price = float(opens[idx])

    # 2. Pick a random direction
    direction = random.choice(["long", "short"])

    # 3. Calculate fixed percentage stop-loss and take-profit
    stop_pct = 0.005   # 0.5% stop loss
    profit_pct = 0.01  # 1.0% take profit

    if direction == "long":
        sl = open_price * (1.0 - stop_pct)
        tp = open_price * (1.0 + profit_pct)
    else:
        sl = open_price * (1.0 + stop_pct)
        tp = open_price * (1.0 - profit_pct)

    # Return a single order utilizing the full portfolio weight
    return [{
        "ticker": ticker,
        "direction": direction,
        "weight": 1.0,  # All-in on this random pick
        "stop_loss": sl,
        "take_profit": tp,
    }]


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")

    t_end = time.time()

    print("\n---")
    print(f"sharpe_ratio:     {results.get('sharpe_ratio', 0):.4f}")
    print(f"total_return:     {results.get('total_return', 0)*100:.2f}%")
    print(f"max_drawdown:     {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"win_rate:         {results.get('win_rate', 0)*100:.1f}%")
    print(f"avg_trades/bar:   {results.get('avg_daily_trades', 0):.2f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")