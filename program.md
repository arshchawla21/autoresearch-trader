# autoresearch-trader

LLMs building stock trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch-trader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-trader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — raw OHLCV data download + fixed backtesting evaluation. **Read-only.**
   - `train.py` — the file you modify. **Everything lives here**: feature engineering, strategy logic, model architecture, training loop, or no training at all.
4. **Verify data exists**: Check that `~/.cache/autoresearch-trader/` contains processed data. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What prepare.py gives you

`prepare.py` is a thin data provider. It downloads ~5 years of daily OHLCV data for 28 tradeable assets (ETFs + major stocks) and 6 macro/reference tickers (VIX, Treasury yields, Gold, Dollar, Bonds, HY credit). All data is aligned to common trading days and stored as:

- `ohlcv` — `(D, N_all, 5)` tensor with channels `[Open, High, Low, Close, Volume]`
- `forward_returns` — `(D, N_tradeable)` simple close-to-close returns (used by evaluation)
- Ticker lists, index mappings, train/test date split

**That's it.** No features are precomputed. You get raw price and volume data. What you do with it is entirely up to you.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. **Everything is fair game:**
  - Feature engineering: compute any technical indicators, statistical features, cross-asset signals, regime indicators, whatever you want from the raw OHLCV.
  - Strategy type: deep learning, gradient boosting, pure algorithmic/rule-based trading, statistical arbitrage, momentum, mean reversion, volatility trading, hybrid approaches, ensembles.
  - Model architecture, loss function, optimizer, hyperparameters.
  - Position sizing logic, risk management, turnover control.
  - Or skip training entirely — a pure rule-based strategy that just implements `predict_fn` is perfectly valid.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It provides raw OHLCV data and the fixed evaluation harness.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_sharpe` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest `sharpe_ratio`.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the whole approach — try raw algorithmic trading, different kinds of ML models, even a hybrid. The world is your oyster.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful Sharpe gains, but it should not blow up dramatically.


**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## The predict_fn contract

The only interface between your strategy and the evaluation is:

```python
predict_fn(ohlcv_history, meta) -> (num_tradeable,) tensor
```

- `ohlcv_history`: `(T, N_all, 5)` — raw OHLCV sliced up to today (no future leakage).
- `meta`: dict with ticker lists, index mappings, `today_idx`.
- Returns: raw position scores for tradeable assets. The evaluator normalizes these to target leverage and applies transaction costs.

Your `predict_fn` can do anything internally: run a trained neural net, apply rules to the raw data, look up a precomputed signal table — whatever produces good scores.

## Metrics

The primary metric is **annualized Sharpe ratio** from a walk-forward backtest on the held-out test period (mid-2025 to present). The `evaluate_sharpe` function also reports:

- `total_return` — cumulative return over the test period
- `max_drawdown` — worst peak-to-trough decline
- `avg_turnover` — average daily portfolio turnover

Only `sharpe_ratio` is used for ranking experiments.

## results.tsv format

```
tag	sharpe_ratio	total_return	max_drawdown	avg_turnover	training_seconds	total_seconds	peak_vram_mb	num_params	description
```

Record every run. After each experiment, append one row.