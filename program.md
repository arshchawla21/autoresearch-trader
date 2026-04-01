# autoresearch-trader

LLMs building intraday stock trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch-trader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-trader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — raw OHLCV data download + fixed day-trading evaluation harness. **Read-only.**
   - `train.py` — the file you modify. **Everything lives here**: feature engineering, strategy logic, model architecture, training loop, or no training at all.
4. **Verify data exists**: Check that `~/.cache/autoresearch-trader/processed_v2/` contains processed data. If not, tell the human to run `uv run prepare.py download`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What prepare.py gives you

`prepare.py` downloads ~10 years of daily OHLCV data for 28 tradeable assets (ETFs + major stocks) and 6 macro/reference tickers (VIX, Treasury yields, Gold, Dollar, Bonds, HY credit). All data is aligned to common trading days and stored as:

- `ohlcv` — `(D, N_all, 5)` tensor with channels `[Open, High, Low, Close, Volume]`
- Ticker lists, index mappings

**That's it.** No features are precomputed. You get raw price and volume data. What you do with it is entirely up to you.

## Trading paradigm: day trading with stop-loss / take-profit

This is a **day-trading** system. Every position opens and closes within the same trading day. Your strategy:

1. **Enters at the Open** price each day
2. **Must set a stop-loss and take-profit** (price levels) on every trade
3. **Exits** when stop-loss or take-profit is hit (checked against daily High/Low), or at the day's Close if neither triggers
4. Can trade **long or short** on any stock in the tradeable pool
5. Has a **portfolio weight budget of 1.0** — you can concentrate on 1 stock or spread across many
6. Pays **10 bps round-trip** transaction costs (5 bps per side)

The key challenge: calibrating stop-loss and take-profit levels well. Too tight = stopped out constantly. Too wide = held to close with no edge. The daily High/Low range is your friend for calibration.

## Experimentation

Each experiment runs on a single GPU. The `build_strategy` function runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it simply as: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. **Everything is fair game:**
  - Feature engineering: compute any technical indicators, statistical features, cross-asset signals, regime indicators from the raw OHLCV.
  - Strategy type: deep learning, gradient boosting, pure algorithmic/rule-based, statistical arbitrage, momentum, mean reversion, volatility trading, hybrid approaches, ensembles.
  - Signal generation: use any method to decide which stocks to trade, when, and in which direction.
  - Stop-loss / take-profit calibration: ATR-based, volatility-based, fixed percentage, adaptive — whatever works.
  - Position sizing: how to allocate the 1.0 weight budget across trades.
  - Or skip training entirely — a pure rule-based strategy that just implements `generate_orders` is perfectly valid.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It provides raw OHLCV data and the fixed evaluation harness.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.
- Look at future data in `generate_orders`. You see `data["ohlcv"][:day_idx]` (history) and `data["ohlcv"][day_idx, :, O]` (today's open). Today's High/Low/Close are **unknown** to you.

**The goal is simple: get the highest average `sharpe_ratio` across all 19 test slices.** Everything is fair game: try different approaches, signal combinations, ML models, or pure rule-based systems.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## The strategy contract

Your strategy must implement two functions:

### `build_strategy(train_data) -> strategy`

Called once per evaluation slice with the training data. Use this to:
- Compute features, fit models, calibrate parameters
- Analyze historical patterns
- Or do nothing if your strategy is purely reactive

`train_data` contains: `ohlcv` (D_train, N_all, 5), `dates`, `all_tickers`, `tradeable_tickers`, `tradeable_indices`, `macro_indices`, `num_tradeable`, `num_all`.

### `generate_orders(strategy, data, day_idx) -> list[order]`

Called once per trading day during evaluation. Returns a list of orders:

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
- `data["ohlcv"][:day_idx]` is known history (use freely)
- `data["ohlcv"][day_idx, :, 0]` is today's open (available)
- `data["ohlcv"][day_idx, :, 1:4]` is today's H/L/C — **DO NOT ACCESS** (future data)

## Evaluation: 19 expanding-window slices

The evaluation uses **walk-forward backtesting** across 19 six-month test windows:

- 10 years of data split into 20 six-month periods
- Slice k (k=0..18): train on [month 0, month 6*(k+1)], test on [month 6*(k+1), month 6*(k+2)]
- First slice: 6 months of training, test on months 6-12
- Last slice: ~9.5 years of training, test on months 114-120
- Strategy is rebuilt from scratch for each slice

The primary metric is the **average annualized Sharpe ratio** across all 19 slices.

## Metrics

- `sharpe_ratio` — average annualized Sharpe across slices (PRIMARY)
- `total_return` — average cumulative return per slice
- `max_drawdown` — average worst peak-to-trough per slice
- `win_rate` — average fraction of profitable trading days
- `avg_daily_trades` — average number of trades per day

Only `sharpe_ratio` is used for ranking experiments.

## results.tsv format

```
tag	sharpe_ratio	total_return	max_drawdown	win_rate	avg_daily_trades	build_seconds	total_seconds	peak_vram_mb	description
```

Record every run. After each experiment, append one row. Commit each successful attempt.
