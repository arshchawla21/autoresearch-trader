# Findings Log

## apr12 — USD/JPY 15m, 97 experiments (v1–v100)

**Champion: v80 (v69_parkinson)** — Sharpe **-0.55**, return **+3.60%**, win **49.14%**, 409 trades (1.8/day), MaxDD -0.92%, Calmar 3.93, PF 1.25. Random baseline Sharpe = -6.65.

---

### What worked (structural lifts, ranked by impact)

- **Mean-reversion is the regime.** Reversed signal diagnostic (v69_reversed) hit 32.3% win — proves the forward 49% edge is real, not random noise.
- **Parkinson H/L volatility > ATR > Garman-Klass** for regime detection. Replacing ATR with Parkinson for the spike-skip filter and Z-threshold scaling was the single biggest lift (v61→v80 Sharpe -0.70 → -0.55).
- **ATR-adaptive brackets with tight floors** (TP_MIN=6, SL_MIN=4, TP_MAX=20, SL_MAX=14) beat fixed 15/10. Bound the SL floor low to reduce MTM variance per trade.
- **1.5:1 TP:SL is the sweet spot.** 1:1 (v69_1to1) pushed win rate to 56% but spread ate small wins. 2:1 (v69_2to1) dropped win to 39%.
- **24-bar trend gate is load-bearing.** 48-bar too short (42.7% win), 144-bar too long (46%). 96-bar optimal.
- **Z/RSI/Williams %R in OR** beats any single oscillator or 2-of-3 AND voting. Independent pullback detectors catch different slices.
- **Cross-asset "any 1 of 3" confirm** (gold, DXY, Nikkei, 4-bar lookback, 5bps threshold) is load-bearing. Removing it drops quality 4pp (v47_no_xasset).
- **2-bar strict pullback persistence** beats 1-bar (noisy) and 3-bar (over-strict). Loose 2-of-3 also worse.
- **Adaptive Z-threshold** scaled by current/200-bar vol ratio, clamped [0.9, 1.8]. More sensitive in calm, stricter in active regimes.
- **Parkinson vol spike skip** at 90th percentile of 500-bar distribution. Trades through vol spikes are net-negative.
- **Top-decile win-rate filters** (swing confirm 51.5%, TNX regime 51.57%, gold corr gate 47.5%) all had **real quality lift** but collapsed trade count too much to help Sharpe.

### What didn't work (dead categories)

- **Session gating is dead.** Tokyo-only -4.13, skip-NY -2.30, London-ORB -3.98, London v47 -2.74, Tue-Thu -2.89. No hour-of-day has isolable edge on current OANDA data.
- **Historic v14 replay fails.** Pure z>1.5 MR vote got 39% win (below 40% random breakeven). Inverted (v98 momentum) also 39%. Raw z-score extremes have **no directional edge** in current regime.
- **Pin bars signal continuation, not reversal.** v95 got 36% win — rejection candles continue the move on 15m USD/JPY.
- **Opening range breakouts dead both ways.** London ORB breakout 38.6%, fade 39.8%. No info in the range.
- **VWAP session fade** too noisy (42% win, many trades).
- **DXY-JPY cointegration spread** — neither fade (44.4%) nor momentum (45.8%) has directional edge.
- **ML / HistGBM / logistic regression** all overfit warmup regime (histgbm_longh: -27% return, 38.6% win). Warmup → eval regime shift is too strong.
- **Hour-of-day seasonality** doesn't generalize warmup→eval.
- **Calm-bar filter** (v99) kills best reversal entries — reversal bars are usually large-range.
- **Pin/swing OR xasset** dilutes quality (v93 Sharpe -1.85).
- **Multi-horizon trend vote** (48/96/288): horizon mixing dilutes the 96-bar edge.
- **SPY swap for gold** in xasset confirm — worse anchor.
- **TNX as 4th OR** — adds marginal trades but drops win rate 49.14→48.91.
- **VIX regime direction flip** — high-vol regime doesn't flip the edge.
- **Strong-trend filter** (trend magnitude ≥ 20bps required) — MR needs weak drift, strong trends break it.
- **Meta-labeling logistic filter** trained on warmup — kills 70% of trades and drops quality.
- **Shock fade / shock ride** — 2-bar >2σ shocks have no directional edge either way.
- **Wick rejection signal** — 44.8% win at random baseline.
- **Volume filter** (v36): mild lift in isolation but doesn't stack with v80 structure.

### Binding constraint: per-bar MTM variance

- Every strategy that clears ~45% win rate is **Sharpe-negative despite positive return**. v80 earns +3.60% over 170 days but per-bar return stream is choppy enough to kill annualized Sharpe.
- Signal-quality lifts (swing 51.5%, TNX 51.57%, gold corr 47.5%) **do not translate to Sharpe** because they cut trade count faster than they cut variance.
- ATR-adaptive brackets mean each trade spans ~10–30 × 15m bars of MTM oscillation. Holding period is the dominant Sharpe term, not per-trade edge.

---

### What I could try next (train.py side, current data)

- **DXY-beta residual MR.** Regress USDJPY returns on DXY returns (60-bar rolling), z-score the residual, fade when residual is stretched. Statistical arbitrage cleaner than current "any 1 of 3" OR.
- **Kalman-filter trend baseline.** Replace the 96-bar endpoint trend with a Kalman smoother — reduces whipsaw in trend classification.
- **HMM 2-state regime model.** Fit on warmup, switch between MR and flat (not MR vs momentum — both momentum variants failed). Gate entries to the MR state only.
- **Pairs with EUR/JPY or GBP/JPY.** If OANDA data could be extended, JPY-basket cointegration gives a cleaner anchor than DXY. (Pitch territory.)
- **Online learning with tiny refit budget.** Logistic filter refit every 1000 bars on the last 2000 bars — track recent regime, not fit-once-on-warmup.
- **Ensemble of v80 + DXY residual + HMM gate** with 2-of-3 vote — diversify the signal sources rather than stack more filters on one MR stem.
- **Structural pause-after-loss.** Maintain module-level "skip next K bars after an SL" counter. Can't observe trade outcomes from `trade()` directly, but can infer from position_was_open→now_flat transitions and price-level checks.

### What would unlock new strategy classes (prepare.py pitches)

- **Populate the economic calendar.** `prices["_CALENDAR"]` currently has **0 rows** despite program.md advertising US + JP high/medium events from Finnhub. Without it I cannot:
  - Blackout ±30 min around NFP / CPI / BoJ / FOMC (biggest single source of MTM variance spikes)
  - Pre-position on scheduled events based on historical hit rates
  - Condition regime on "next 2h contains high-impact event" vs not
  This is likely the highest-leverage single fix — it directly attacks the MTM-variance ceiling.

- **5-minute bars.** 15m bracket holds span hours of real time, which is the mechanical cause of the MTM-variance ceiling. On 5m I could:
  - Use same MR structure with tighter brackets (4–6p TP / 3–4p SL)
  - Resolve trades in minutes, dramatically reducing per-bar variance per trade
  - 3× the trade count → approach the user's 10+ trades/day target
  - Keep 15m data loaded as a coarser trend context

- **Live / historical NLP news feed.** Finnhub or similar headline stream with sentiment scores. Enables:
  - Intraday sentiment regime detection (risk-on/off) beyond equity-linked CFDs
  - Event-driven entries (Fed speak, BoJ commentary, surprise index moves)
  - A genuine feature set for ML filters that doesn't overfit to price-action patterns

- **Additional macro tickers.**
  - **EUR/JPY, GBP/JPY** — JPY-basket cointegration, the cleanest MR anchor for a yen strategy
  - **US 2Y yield** or **2s10s spread** — short-end is the real yield-differential driver for USDJPY (current `^TNX` / USB10Y is too far out the curve)
  - **JPY basis swap or MOF intervention flow proxy** — direct read on BoJ / MoF positioning
  - **CFTC COT speculative positioning** (weekly) — crowded-trade reversal signal

- **Microstructure data: bid/ask spread, tick volume, order book imbalance.** OANDA practice API exposes some of this. Enables:
  - Execution-aware entries (skip when spread widens)
  - Genuine volume-based regime filters (current tick volume is ~noise)
  - Order-flow imbalance as a leading indicator

- **Economic surprise index** (Citi ESI or similar, daily). A scalar regime variable encoding "recent data > expectations" — complements raw event filtering.

- **Transaction-cost realism pitch:** current 0.8p is already modeled; keep it. But pair-dependent slippage during news would improve the realism of any news-gated strategy.
