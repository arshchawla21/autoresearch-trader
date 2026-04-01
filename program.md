Here is the updated `program.md` reflecting the shift to 15-minute data, the 60-day yfinance limit, the 30/30 train/eval split, and the strict intraday (no overnight holds) trading rule.

***

# autoresearch-trader

LLMs building intraday stock trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch-trader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-trader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — raw OHLCV data download + fixed evaluation harness. **Read-only.**
   - `train.py` — the file you modify. **Everything lives here**: feature engineering, strategy logic, model architecture, training loop, or no training at all.
4. **Verify data exists**: Check that `~/.cache/autoresearch-trader/processed_v2/` contains processed data. If not, tell the human to run `uv run prepare.py download`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What prepare.py gives you

`prepare.py` downloads the **last ~60 days of 15-minute OHLCV data** for 28 tradeable assets (ETFs + major stocks) and 6 macro/reference tickers (VIX, Treasury yields, Gold, Dollar, Bonds, HY credit). All data is aligned to common 15-minute intervals and stored as:

- `ohlcv` — `(T, N_all, 5)` tensor with channels `[Open, High, Low, Close, Volume]`, where `T` is the number of 15-minute intervals.
- Ticker lists, index mappings

**That's it.** No features are precomputed. You get raw price and volume data. What you do with it is entirely up to you.

## Trading paradigm: intraday with stop-loss / take-profit

This is a strict **intraday** system. Every position must open and close within the same trading day (no overnight holds). Your strategy:

1. **Enters at the Open** price of a 15-minute bar.
2. **Must set a stop-loss and take-profit** (price levels) on every trade.
3. **Exits** when the stop-loss or take-profit is hit (checked against the bar's High/Low). If neither triggers, the trade closes by the end of the day.
4. Can trade **long or short** on any stock in the tradeable pool.
5. Has a **portfolio weight budget of 1.0** — you can concentrate on 1 stock or spread across many.

The key challenge: calibrating stop-loss and take-profit levels for intraday volatility. Too tight = stopped out by 15-minute noise. Too wide = held to close with no edge. The 15-minute High/Low range is your friend for calibration.

## Experimentation

Each experiment runs on a single GPU. The `build_strategy` function runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it simply as: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. **Everything is fair game:**
  - Feature engineering: compute technical indicators, cross-asset signals, or microstructure features from the 15-minute OHLCV.
  - Strategy type: deep learning, gradient boosting, pure algorithmic/rule-based, statistical arbitrage, momentum, mean reversion, volatility trading.
  - Signal generation: use any method to decide which stocks to trade, when, and in which direction.
  - Stop-loss / take-profit calibration: ATR-based, volatility-based, fixed percentage, adaptive — whatever works.
  - Position sizing: how to allocate the 1.0 weight budget across trades.
  - Or skip training entirely — a pure rule-based strategy that just implements `generate_orders` is perfectly valid.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It provides raw OHLCV data and the fixed evaluation harness.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.
- Look at future data in `generate_orders`. You see `data["ohlcv"][:bar_idx]` (history) and `data["ohlcv"][bar_idx, :, O]` (the current 15m open). The current bar's High/Low/Close are **unknown** to you.

**The goal is simple: get the highest `sharpe_ratio` on the 30-day evaluation set.** Everything is fair game: try different approaches, signal combinations, ML models, or pure rule-based systems.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## The strategy contract

Your strategy must implement two functions:

### `build_strategy(train_data) -> strategy`

Called once with the first 30 days of 15-minute data. Use this to:
- Compute features, fit models, calibrate parameters
- Analyze historical patterns
- Or do nothing if your strategy is purely reactive

`train_data` contains: `ohlcv` (T_train, N_all, 5), `dates`, `all_tickers`, `tradeable_tickers`, `tradeable_indices`, `macro_indices`, `num_tradeable`, `num_all`.

### `generate_orders(strategy, data, bar_idx) -> list[order]`

Called sequentially for every 15-minute interval during the 30-day evaluation period. Returns a list of orders for that specific interval:

```python
{
    "ticker": "AAPL",           # must be in tradeable_tickers
    "direction": "long",        # or "short"
    "weight": 0.25,             # fraction of portfolio (0, 1]
    "stop_loss": 148.50,        # price level
    "take_profit": 153.20,      # price level
}
```

Rules:
- Total weight across all orders must be <= 1.0
- `data["ohlcv"][:bar_idx]` is known history (use freely)
- `data["ohlcv"][bar_idx, :, 0]` is the open of the current 15-minute bar (available)
- `data["ohlcv"][bar_idx, :, 1:4]` is the current bar's H/L/C — **DO NOT ACCESS** (future data)

## Evaluation: 30 days train, 30 days eval

Because high-resolution intraday data is limited to ~60 days via `yfinance`, the evaluation utilizes a **single train/test split**:

- **Train:** The first 30 days of the dataset (used by `build_strategy`).
- **Evaluate:** The last 30 days of the dataset (out-of-sample walk-forward testing using `generate_orders`).

The primary metric is the **annualized Sharpe ratio** generated during the 30-day out-of-sample evaluation period.

## Metrics

- `sharpe_ratio` — annualized Sharpe ratio of the 30-day eval period (PRIMARY)
- `total_return` — cumulative return over the 30-day eval period
- `max_drawdown` — worst peak-to-trough drawdown during evaluation
- `win_rate` — fraction of profitable trades/intervals
- `avg_daily_trades` — average number of trades executed per day

Only `sharpe_ratio` is used for ranking experiments.

## results.tsv format

```
tag sharpe_ratio  total_return  max_drawdown  win_rate  avg_daily_trades  build_seconds total_seconds peak_vram_mb  description
```

Record every run. After each experiment, append one row. Commit each successful attempt.