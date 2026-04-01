"""
v1-baseline: Pure random intraday trading baseline.
"""

import os
import time
import random
import numpy as np

from prepare import O, evaluate

t_start = time.time()
random.seed(42)


def build_strategy(train_data):
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
    }


def generate_orders(strategy, data, bar_idx):
    if random.random() > 0.115:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    opens = data["ohlcv"][bar_idx, tidx, O]

    valid = [i for i, op in enumerate(opens) if not np.isnan(float(op)) and float(op) > 0]
    if not valid:
        return []

    idx = random.choice(valid)
    op = float(opens[idx])
    direction = random.choice(["long", "short"])

    sl_pct, tp_pct = 0.005, 0.01
    if direction == "long":
        sl, tp = op * (1 - sl_pct), op * (1 + tp_pct)
    else:
        sl, tp = op * (1 + sl_pct), op * (1 - tp_pct)

    return [{"ticker": tickers[idx], "direction": direction, "weight": 1.0,
             "stop_loss": sl, "take_profit": tp}]


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
