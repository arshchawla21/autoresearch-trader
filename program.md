# program.md — Autonomous Trading Research Agent

## Mission

You are an autonomous research agent. Your goal is to **discover novel, profitable intraday trading strategies** by iteratively modifying `train.py`. You are evaluated on a fixed backtesting harness (`prepare.py`) that you **cannot change**.

You are not fine-tuning hyperparameters on a known approach. You are **inventing strategies**. Be radical. Try things that might fail spectacularly. The only constraint is the `trade()` API contract.

---

## Session Startup Protocol

Every time you begin a new research session, follow these steps **in order**:

### 1. Agree on a run tag
Propose a tag based on today's date (e.g. `apr03`). The branch `autoresearch-trader/<tag>` must not already exist — this is a fresh run. If the date-based tag is taken, append a letter (e.g. `apr03b`).

### 2. Create the branch
```bash
git checkout -b autoresearch-trader/<tag>
```
from current `master` (or `main`).

### 3. Read the in-scope files
The repo is small. Read these files for full context:
- `README.md` — repository context (if it exists).
- `prepare.py` — raw OHLCV data download + fixed backtesting evaluation. **Read-only. Never modify.**
- `train.py` — the file you modify. **Everything lives here**: feature engineering, strategy logic, model architecture, training loop, or no training at all.

### 4. Verify data exists
Check that `~/.cache/autoresearch-trader/` contains `.parquet` files. If not, tell the human to run:
```bash
uv run prepare.py --download
```

### 5. Initialize results.tsv
If `results.tsv` does not exist, create it with this header row:

```
run_tag	strategy_name	sharpe_ratio	total_return	sortino_ratio	calmar_ratio	max_drawdown	win_rate	profit_factor	n_trades	notes	timestamp
```

The baseline random strategy will be the first entry after the initial run.

### 6. Confirm and go
Print a summary of what you see and confirm you're ready to begin experimenting.

---

## The API Contract

You modify **only** `train.py`. You must implement:

```python
def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
```

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `prices` | `dict[str, pd.DataFrame]` | ALL historical OHLCV data from the dataset start up to the current 5-min candle. Keys include 15 tradeable symbols + 5 market indicators. |
| `current_idx` | `int` | Index of the current candle in the aligned price matrix. |
| `symbols` | `list[str]` | Ordered list of the 15 tradeable symbols. Your output must match this order. |

### Output

A `list[float]` of length 15 (one weight per tradeable symbol).

- **Positive** = long position, **Negative** = short position.
- **Constraint**: `sum(|weights|) <= 1.0`. If you exceed this, the harness auto-rescales.
- **Example**: `[0, 0.25, 0.5, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` → 25% long MSFT, 50% long NVDA, 25% short TSLA.

### What you get for free

At the first evaluation candle (day 31), you already have **~30 trading days** (~2,340 five-minute candles) of OHLCV data for every symbol. Use this for:
- Training ML models (fit on days 1–30, predict on day 31+)
- Computing technical indicators that need lookback windows
- Regime detection / clustering
- Building covariance matrices
- Literally anything

### Tradeable universe (15 symbols)

**Stocks**: AAPL, MSFT, NVDA, AMZN, TSLA, META, GOOGL, JPM, XOM, UNH
**ETFs**: SPY, QQQ, IWM, XLF, XLE

### Market indicators (read-only, cannot trade)

^VIX (volatility), ^TNX (10Y yield), GLD (gold), TLT (bonds), DX-Y.NYB (dollar index)

---

## Evaluation

Run the backtest:
```bash
uv run prepare.py
```

This prints and returns:
- **Total Return** — cumulative P&L over the eval window
- **Sharpe Ratio** — risk-adjusted return (annualized, 5% risk-free rate)
- **Sortino Ratio** — like Sharpe but only penalizes downside volatility
- **Calmar Ratio** — return / max drawdown
- **Max Drawdown** — worst peak-to-trough decline
- **Win Rate** — fraction of candles with positive return
- **Profit Factor** — gross profits / gross losses

Your primary optimization target is **Sharpe ratio**, but pay attention to max drawdown and total return too.

---

## Recording Results

After every experiment, **append a row** to `results.tsv`:

```
apr03	momentum_cross	1.847	0.0312	2.105	3.42	-0.0091	0.523	1.34	1716	SMA 10/30 cross on NVDA+TSLA	2026-04-03T14:22:00
```

Fields (tab-separated):
1. `run_tag` — the branch tag
2. `strategy_name` — short descriptive name you invent
3. `sharpe_ratio` — from backtest output
4. `total_return` — from backtest output
5. `sortino_ratio` — from backtest output
6. `calmar_ratio` — from backtest output
7. `max_drawdown` — from backtest output
8. `win_rate` — from backtest output
9. `profit_factor` — from backtest output
10. `n_trades` — from backtest output
11. `notes` — brief description of what you tried
12. `timestamp` — ISO timestamp of when you ran it

Then commit:
```bash
git add train.py results.tsv
git commit -m "<strategy_name>: sharpe=<X>, return=<Y>"
```

This way we can `git log` to see the full history of experiments and `git diff` between any two to see exactly what changed.

---

## Strategy Ideas — Go Wild

The whole point is to explore broadly. Here are categories to consider, but **do not limit yourself to these**:

### Technical / Algorithmic
- Moving average crossovers (SMA, EMA, DEMA, TEMA)
- Bollinger Band mean reversion
- RSI / MACD / Stochastic oscillator signals
- VWAP deviation strategies
- Order flow imbalance (volume analysis)
- Pairs trading / statistical arbitrage between correlated assets
- Breakout detection (Donchian channels, ATR-based)

### Statistical / Quantitative
- Kalman filter for trend estimation
- Hidden Markov Models for regime detection
- PCA on returns → trade principal components
- Cointegration-based pairs (Engle-Granger, Johansen)
- GARCH volatility forecasting → vol-targeting
- Copula-based dependency modeling
- Bayesian online changepoint detection

### Machine Learning
- Gradient-boosted trees (XGBoost/LightGBM) on engineered features
- LSTM / GRU on raw price sequences
- Transformer-based sequence models
- Reinforcement learning (Q-learning, policy gradient on the weight space)
- Autoencoders for anomaly detection → contrarian signals
- Random forests on microstructure features
- Online learning (no batch training — update every candle)

### Cross-Asset / Macro
- VIX regime switching (risk-on / risk-off allocation)
- Dollar index as a leading indicator for equities
- Yield curve slope → sector rotation
- Gold/bond flight-to-safety signals
- Correlation breakdown detection → crisis alpha

### Exotic / Creative
- Entropy-based position sizing
- Fractal dimension of recent price paths
- Genetic programming to evolve trading rules
- Attention-weighted ensemble of multiple sub-strategies
- Adversarial strategy: detect and fade momentum chasers
- Information-theoretic features (transfer entropy between assets)
- Topological data analysis on price manifolds

### Meta-Strategies
- Ensemble: run 5 different strategies, vote on weights
- Adaptive: measure recent Sharpe of sub-strategies, reallocate
- Anti-overfit: train on odd days, validate on even, only deploy if both work
- Kelly criterion position sizing on top of any signal

---

## Rules of Engagement

1. **Only edit `train.py`**. Never touch `prepare.py`.
2. **Record every experiment** in `results.tsv` and commit to git.
3. **Be novel**. If your last 3 experiments were all moving-average variants, pivot to something completely different.
4. **Fail fast**. If an idea gives a Sharpe < 0.5 after the first run, don't tweak it — try a different approach entirely.
5. **Iterate in 15-minute bursts**. Implement → run → record → reflect → pivot or refine.
6. **Use the indicators**. VIX, treasury yields, dollar index, gold, and bonds are there for a reason. Cross-asset signals are underexplored.
7. **Watch for overfitting**. You're backtesting on 22 trading days of 5-min data. If your Sharpe is above 5.0, you're probably overfitting. Be suspicious.
8. **Speed matters somewhat**. The backtest calls `trade()` ~1,700 times. If a single call takes >100ms, the total runtime becomes painful. Cache aggressively.
9. **Think about why**. Before implementing, write a 1-sentence hypothesis: "I believe X because Y." After running, check if the results support the hypothesis.

---

## Example Workflow

```
Human: Let's start a new research session.

Agent:
1. Tag: apr03 — checking branch doesn't exist... ✓
2. Creating branch: git checkout -b autoresearch-trader/apr03
3. Reading prepare.py, train.py...
4. Data check: ~/.cache/autoresearch-trader/ has 20 parquet files ✓
5. Creating results.tsv with header row ✓

Ready. Let me start with the random baseline to establish a floor.

[runs uv run prepare.py, records results]

Baseline: Sharpe=-0.02, Return=-0.1%. As expected for random.

Hypothesis: "VIX regime switching should work because high-VIX
environments favor defensive positioning (long XLE/UNH, short TSLA/NVDA)
while low-VIX favors risk-on (long NVDA/TSLA/QQQ)."

[implements VIX regime strategy in train.py]
[runs backtest]
[records to results.tsv]
[git add + commit]

Result: Sharpe=0.82, Return=+1.2%. Promising. But let me try
something completely different next...

Hypothesis: "PCA on rolling returns can extract latent factors.
Trading the first principal component's momentum should capture
broad market moves while reducing noise."

[implements PCA strategy]
...
```

---

## Final Notes

- The harness uses `np.random.default_rng()` without a fixed seed in the baseline, so the random strategy's results will vary slightly between runs. That's fine — it's a baseline.
- All data is 5-minute candles. There are roughly 78 candles per trading day (6.5 hours × 12 candles/hour).
- The evaluation window is approximately days 31–60, giving ~22 trading days and ~1,700 eval candles.
- The risk-free rate is set to 5% annual for Sharpe calculation.
- Transaction costs are NOT modeled. This is intentional — focus on signal quality first, worry about friction later.
- If you want to do in-strategy training (e.g., fit an ML model on the warmup period), do it **once** on the first call to `trade()` and cache the model in a global variable. Don't retrain every candle unless you have a specific reason.

Good luck. Find alpha.