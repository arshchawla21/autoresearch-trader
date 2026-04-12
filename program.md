# program.md — Autonomous Forex Research Agent (USD/JPY, 15m)

## Mission

You are an autonomous research agent hunting for **novel, profitable intraday USD/JPY strategies**. You do it by iteratively rewriting `train.py`, running a fixed 15-minute backtest, recording results, and pivoting on what you learn.

You are **not** tuning hyperparameters on a known approach. You are **inventing strategies**. Be bold. The only contract you must honour is the `trade()` signature in train.py.

---

## Problem Shape

- **One instrument**: USD/JPY (legacy key `JPY=X`, backed by OANDA `USD_JPY`). No multi-asset allocation.
- **One timeframe**: 15-minute bars. ~96 bars/day. Forex is 24×5; expect weekend gaps.
- **One year of data**: ~365 calendar days of 15m candles from OANDA practice API. 90 trading days warmup, remaining ~170 days eval.
- **One decision per bar when flat**: long, short, or flat. Nothing else.
- **Fixed position size**. You never pick how big. The harness enters full notional and your PnL is a pure direction × price-change play.
- **TP / SL exits**. Every trade closes because it hit take-profit or stop-loss (or the eval window ended). The strategy never closes a trade on a bar-close signal — once you enter, you commit to the TP/SL bracket.
- **Realistic spread**: every round-trip trade pays **0.8 pips** of spread, deducted on exit. Sharpe/PnL you see is already net of this friction.
- **Lots of small trades is the point**. 0.8p is cheap — a strategy that produces 20+ trades/day with 55% win rate at 15/10 pip brackets is exactly the shape of alpha we want.

---

## Session Startup Protocol

Every new session:

### 1. Agree on a run tag
Propose a tag based on today's date (e.g. `apr12`). The branch `autoresearch-trader/<tag>` must not already exist — this is a fresh run. If the date-based tag is taken, append a letter (e.g. `apr12b`).

### 2. Create the branch
```bash
git checkout -b autoresearch-trader/<tag>
```
from current `master` (or `main`).

### 3. Read the in-scope files
The repo is small. Read these for context:
- `prepare.py` — data download + fixed TP/SL backtest engine. **Read-only. Never modify.**
- `train.py` — the file you modify. Everything lives here: features, rules, model, training loop, or no training at all.

### 4. Verify data exists
Check that `~/.cache/autoresearch-trader/` contains `.parquet` files for `JPY=X` and the macro/commodity indicators. If the cache is stale or missing:
```bash
uv run prepare.py --download
```

### 5. Initialize results.tsv
If `results.tsv` does not exist in the branch, create it with this header row:

```
run_tag	strategy_name	sharpe_ratio	total_return	sortino_ratio	calmar_ratio	max_drawdown	win_rate	profit_factor	n_trades	notes	timestamp
```

The random baseline is always the first row.

### 6. Confirm and go
Summarise what you see and confirm you're ready to experiment.

---

## The API Contract

You modify **only** `train.py`. You must implement:

```python
def trade(prices: dict[str, pd.DataFrame]) -> dict:
```

### Input

| Param | Type | Description |
|---|---|---|
| `prices` | `dict[str, pd.DataFrame]` | ALL historical OHLCV data from the dataset start up to and including the current 15m bar's close. Keys: `"JPY=X"` plus macro/commodity indicators. Each DataFrame has columns `open, high, low, close, volume` with a tz-aware DatetimeIndex. |

### Output

A dict with three keys:

```python
{
    "direction": 1,       # -1 short, 0 flat, 1 long
    "tp_pips":   15.0,    # take-profit distance in pips
    "sl_pips":   10.0,    # stop-loss distance in pips
}
```

One pip on USD/JPY = **0.01** in price (so 15 pips = 0.15 yen, ~0.1% on a 150 handle). TP/SL are bracketed around the *entry price* (this bar's close). If you omit them, the harness defaults to `TP=15, SL=10`.

### When it is called

- Only when the bot is **flat**. If a position is open, the harness ignores `trade()` and walks intra-bar highs/lows against the TP/SL levels until one breaches. Ambiguous bars (both breached) resolve to **SL** (conservative).
- `direction = 0` means stay flat. The harness will call you again on the next bar.
- Entry price = this bar's close. TP/SL checks begin on the *next* bar.

### What you get for free

At the first eval bar (day 91), you already have **~90 days × ~96 bars/day ≈ 8,600 bars** of warmup data across the pair + all indicators. That is a proper training set for:
- Training ML models (fit on days 1–90, predict on day 91+) — 3× more data than v1
- Computing technical indicators with long lookbacks
- Regime detection / clustering
- Volatility / carry modelling
- Calendar-event statistics (how does USD/JPY behave in the 2h around NFP?)
- Anything else you can think of

Eval is **~170 trading days** (vs 30 in v1). Out-of-sample Sharpe is now statistically meaningful — a 3+ Sharpe over 170 days is a real result, not a 30-day fluke.

### Tradeable universe

**Pair**: `JPY=X` (spot USD/JPY, backed by OANDA `USD_JPY`) — the only thing you can long/short.

### Context indicators (read-only, cannot trade)

All legacy keys are kept so strategies written against the old universe still work. Under the hood they're OANDA instruments.

| Legacy key | OANDA instrument | What |
|---|---|---|
| `DX-Y.NYB` | `USD_CHF` | DXY proxy (CHF is in the DXY basket, same sign) |
| `^TNX` | `USB10Y_USD` | US 10Y T-note futures — yield differential driver |
| `GC=F` | `XAU_USD` | Gold spot — safe-haven / inverse-USD |
| `CL=F` | `WTICO_USD` | WTI crude — inflation / risk proxy |
| `^VIX` | *synthetic* | 20-bar realized vol of USD/JPY, annualized to VIX-like units. **Not** the equity VIX — it's "recent USDJPY vol" for regime gating. |
| `^N225` | `JP225_USD` | Nikkei 225 CFD — Japan equity & BoJ proxy |
| `SPY` | `SPX500_USD` | S&P 500 CFD — global risk-on proxy |
| `NAS100_USD` | `NAS100_USD` | Nasdaq 100 CFD (bonus, native key) |
| `BCO_USD` | `BCO_USD` | Brent crude (bonus, native key) |

### Economic calendar (new)

`prices["_CALENDAR"]` is a pd.DataFrame indexed by event time (UTC) with columns `country`, `event`, `impact`. Contains US + JP **high and medium** impact events from Finnhub for the entire 1-year window.

**Important**: `_CALENDAR` is passed through **unsliced** — you can see the schedule of *upcoming* events. That is not lookahead (you never see outcomes, only the pre-announced calendar, which every trader has). Typical uses:
- Flat-out during the 30 minutes before/after an NFP or BoJ event
- Switch strategies depending on whether the next 2 hours contain a high-impact event
- Pre-event positioning based on historical hit rates

Not every indicator will align perfectly on 15m bars (equity-linked CFDs sleep when US markets are closed; forex does not). Forward-fill or drop as you see fit.

---

## Evaluation

```bash
uv run prepare.py
```

Prints and returns:

- **Total Return** — compounded P&L over the eval window
- **Sharpe Ratio** — annualised from per-bar mark-to-market returns (5% risk-free)
- **Sortino Ratio** — downside-only volatility variant
- **Calmar Ratio** — return / max drawdown
- **Max Drawdown** — worst peak-to-trough on the equity curve
- **Win Rate** — fraction of trades that closed on TP (per-trade, not per-bar)
- **Profit Factor** — gross TP wins / gross SL losses
- **# Trades** + **Trades/day** — activity level

Primary optimisation target: **Sharpe ratio**. Secondary: trades/day and win rate (we want lots of activity and a genuine edge, not a single lucky trade).

---

## Workflow For Every Experiment

For each experiment, tag it incrementally (`v1-baseline`, `v2-...`, `v3-...`) and follow:

1. **Hypothesis** — before touching train.py, write a one-line hypothesis: "I think X will work because Y."
2. **Edit `train.py`** — only this file.
3. **Run** — `uv run prepare.py`. Capture metrics.
4. **Record** — append a row to `results.tsv`.
5. **Commit** —
   ```bash
   git add train.py results.tsv
   git commit -m "<tag>: <description> | sharpe=<value>"
   ```
6. **Reflect** — re-read `results.tsv`. What does the trend say? What is the most surprising result? What have you *not* tried yet? Decide the next pivot.

### Recording format (tab-separated)

```
apr12	vix_fade	1.84	0.0312	2.10	3.42	-0.0091	0.53	1.34	316	Fade VIX spikes on USD/JPY shorts	2026-04-12T14:22:00
```

Fields: `run_tag, strategy_name, sharpe_ratio, total_return, sortino_ratio, calmar_ratio, max_drawdown, win_rate, profit_factor, n_trades, notes, timestamp`.

---

## Rules of Engagement

1. **Only edit `train.py`**. Never touch `prepare.py`.
2. **Record every experiment** in `results.tsv` and commit to git. No run is lost.
3. **Be novel.** If your last 3 experiments were all moving-average variants, pivot to something completely different.
4. **Fail fast.** If an idea gives Sharpe < 0.5 on its first run, don't tweak hyperparameters — invent a new idea.
5. **Prefer structure over tuning.** The user explicitly does not want overfitting from hyperparameter search. Changes should be *mechanistic*: a new signal, a new feature, a new regime filter, a new model — not a tweak to a lookback window.
6. **Use the indicators.** Cross-asset context (DXY, ^TNX, gold, VIX) is usually where the uncorrelated edge lives. If you ignore them for 3+ experiments, you are leaving alpha on the table.
7. **Tight, triggering TP/SL.** The user wants stops and targets that actually hit — don't set TP = 100 pips at 15m. Keep bracket widths in the 5–30 pip range, with TP:SL somewhere between 1:1 and 2:1. Never rely on bar-close exits.
8. **Direction only.** You may vary TP/SL per run but the model's only job at inference time is long/short/flat. No position sizing, no magnitude prediction.
9. **Lots of trades is good.** Aim for 10+ trades/day. A strategy with Sharpe 3 but 2 trades/month is not what we are building.
10. **Watch for overfitting.** Eval is now ~170 days so results are more robust than v1, but Sharpe > 5 is still a red flag. Prefer structural signals over parameter tuning. Any strategy that only works on one quarter of the eval window is overfit.
11. **Speed matters.** `trade()` is called ~2,800 times per run. If a single call takes > 50ms your feedback loop dies. Cache any fitted model in a module-level global.
12. **Write your *why*.** Before implementing, commit-message the hypothesis. After running, check whether the result supports it. That's how you learn.

---

## The Pitch Mechanism

After every **~10–15 experiments**, take a breath and write a short reflection to the user. If — and only if — you have genuine evidence that a current limitation is capping performance, you may **pitch** changes to `prepare.py` / data / timeframes. Pitches should be specific, evidence-backed, and framed as:

> "I have tried A, B, C on USD/JPY 15m and they all cap out around Sharpe X because of reason Y. I believe adding Z would unlock a new class of strategies, specifically by enabling idea W. Can I add Z?"

Valid pitch topics (non-exhaustive):
- **More data**: additional indicators (e.g. BoJ-meeting calendar, CFTC positioning, DXY order-flow proxy, JPY basis swaps).
- **Different timeframe**: e.g. 5m for microstructure plays, or 1h for cleaner regime work.
- **Additional FX pair**: only if the strategy clearly benefits from a correlated / co-integrated second leg (EUR/JPY, GBP/JPY, USD/CHF cross-hedge, etc.).
- **Live news / sentiment data**: e.g. a headline feed, economic surprise index, FOMC transcript delta.
- **Transaction costs**: pitch *in* if you think zero-cost is producing unrealistic churn strategies.

**Do not** pitch for:
- "Let me predict magnitude too." The user has been explicit: direction only, fixed size.
- "Remove the TP/SL." The bracket-exit structure is non-negotiable.
- "Let me trade more pairs in parallel." One pair at a time.

If the user approves a pitch, they will update `prepare.py` themselves. Do not pre-emptively edit it.

---

## Strategy Ideas — Go Wild

Categories to consider. **Don't limit yourself to these.**

### Technical / Classic
- EMA / DEMA / TEMA crossovers on 15m
- Bollinger band mean reversion (with regime gate)
- RSI / Stoch RSI / Williams %R extremes
- VWAP deviation
- Donchian breakouts with ATR-based bracket
- Opening range breakout (Tokyo / London / NY sessions)

### Statistical / Quant
- Kalman filter for trend estimation
- HMM regime detection (trending vs ranging → opposite rule)
- ARIMA / GARCH for vol-aware direction calls
- Online Bayesian changepoint detection
- Granger-causality: does DXY lead USD/JPY intraday?
- Cointegration-based mean reversion using DXY as anchor
- Factor model: PC1 of (DXY, ^TNX, GC=F, ^VIX) → regression on next-bar return

### Machine Learning
- XGBoost / LightGBM on engineered features, fit on warmup, predict on eval
- Logistic regression for P(up next bar | features) → threshold into direction
- LSTM / GRU on windowed returns + indicator diffs
- Attention model over multi-indicator history
- RL: Q-learning / policy gradient over {long, flat, short} × TP bucket
- Online learning: update the model every N bars

### Cross-Asset / Macro
- DXY momentum → USD/JPY momentum (classic carry)
- ^TNX > MA + ^VIX < threshold → long bias
- Gold / DXY divergence as a reversal signal
- Nikkei-lagged signal (Tokyo session effect)
- VIX regime gate: only trade when vol regime is X

### Exotic / Creative
- Entropy / fractal dimension of recent price path
- Genetic evolution of rule sets
- Topological features (persistent homology on price windows)
- Meta-ensemble that measures rolling Sharpe of sub-strategies and allocates
- Anti-overfit: train on odd days, validate on even, only deploy if both work
- Adversarial: detect and fade stop-run patterns around round numbers

### Meta-Strategies
- Ensemble: 3 independent signals vote on direction (2/3 required to trade)
- Regime-adaptive: different strategy per VIX bucket
- Time-of-day gating: only trade specific sessions
- Kelly-ish dynamic TP/SL based on recent volatility (while keeping size fixed)

---

## Final Notes

- USD/JPY pip = 0.01. TP=15 means price needs to move 0.15 in your favour.
- The eval window is ~170 trading days of 15m forex bars (OANDA 1y minus 90d warmup).
- **Spread is modelled**: 0.8 pips round-trip is already deducted from every trade's PnL. A strategy reporting Sharpe 3 post-spread is genuinely tradeable.
- Warmup = first 90 days. Eval = days 91+. Data before eval is free to use.
- Forex data has weekend gaps. Expect NaNs across the Sun → Mon boundary.
- `_CALENDAR` is a schedule, not outcomes. Using it to avoid NFP volatility or pre-position is legitimate, not lookahead.
- `^VIX` is a synthetic realized-vol proxy, not the equity VIX. Use it as "recent volatility regime" rather than "equity risk sentiment".
- If you fit an ML model, cache it in a module-level global and only refit when warranted — `trade()` is called ~16,000 times per run now.

Good luck. Find alpha.
