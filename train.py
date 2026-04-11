#!/usr/bin/env python3
"""
train.py — v31-meta-labeling
============================
Hypothesis: v24 champion has real edge (+0.72%, 43.96% win) but fires
marginally. Train a logistic regression on warmup v24 signals to predict
TP-vs-SL using features not already in v24 (raw VIX, hour, |z|, DXY 1h,
gold 1h). Only take live signals the meta-model thinks will win. Stacked
structural filter — novel vs pure rule-based or pure ML baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TP_PIPS = 15.0
SL_PIPS = 10.0
PIP = 0.01

Z_LB = 20
Z_ENTRY = 1.2
RSI_LB = 14
RSI_LOW = 32.0
RSI_HIGH = 68.0
WR_LB = 14
WR_LOW = -85.0
WR_HIGH = -15.0
TREND_LB = 96
LB_SHORT = 4
MIN_MOVE = 0.0005

META_THRESH = 0.5

_META: dict = {"fit": False, "model": None, "scaler": None}


# ---------- oscillators ----------

def _rsi_arr(closes: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    if len(closes) < n + 1:
        return out
    diffs = np.diff(closes)
    ups = np.maximum(diffs, 0.0)
    downs = np.maximum(-diffs, 0.0)
    avg_up = np.full(len(closes), np.nan)
    avg_down = np.full(len(closes), np.nan)
    avg_up[n] = ups[:n].mean()
    avg_down[n] = downs[:n].mean()
    for i in range(n + 1, len(closes)):
        avg_up[i] = (avg_up[i - 1] * (n - 1) + ups[i - 1]) / n
        avg_down[i] = (avg_down[i - 1] * (n - 1) + downs[i - 1]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_down > 0, avg_up / avg_down, np.inf)
        out = 100.0 - 100.0 / (1.0 + rs)
    out[avg_down <= 0] = np.where(avg_up[avg_down <= 0] > 0, 100.0, 50.0)
    return out


def _wr_arr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    for i in range(n - 1, len(closes)):
        hi = highs[i - n + 1 : i + 1].max()
        lo = lows[i - n + 1 : i + 1].min()
        if hi - lo > 0:
            out[i] = -100.0 * (hi - closes[i]) / (hi - lo)
    return out


def _zscore_arr(closes: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    for i in range(n - 1, len(closes)):
        w = closes[i - n + 1 : i + 1]
        sd = w.std(ddof=1)
        if sd > 0:
            out[i] = (closes[i] - w.mean()) / sd
    return out


# ---------- v24 signal (vectorised) ----------

def _v24_signal_idx(pair: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (idx_of_trigger_bars, direction_of_that_bar) over entire pair."""
    closes = pair["close"].values.astype(float)
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    n = len(closes)
    if n < TREND_LB + 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    z = _zscore_arr(closes, Z_LB)
    rsi = _rsi_arr(closes, RSI_LB)
    wr = _wr_arr(highs, lows, closes, WR_LB)
    trend = np.full(n, np.nan)
    trend[TREND_LB:] = closes[TREND_LB:] / closes[:-TREND_LB] - 1.0

    long_pull = (
        (z < -Z_ENTRY)
        | ((~np.isnan(rsi)) & (rsi < RSI_LOW))
        | ((~np.isnan(wr)) & (wr < WR_LOW))
    )
    short_pull = (
        (z > Z_ENTRY)
        | ((~np.isnan(rsi)) & (rsi > RSI_HIGH))
        | ((~np.isnan(wr)) & (wr > WR_HIGH))
    )

    sig = np.zeros(n, dtype=int)
    sig[(trend > 0) & long_pull] = 1
    sig[(trend < 0) & short_pull] = -1
    idx = np.where(sig != 0)[0]
    return idx, sig[idx]


# ---------- cross-asset confirm at an index ----------

def _aligned_series(pair_idx: pd.DatetimeIndex, other: pd.DataFrame | None) -> np.ndarray | None:
    if other is None:
        return None
    s = other["close"].dropna()
    if len(s) == 0:
        return None
    try:
        return s.reindex(pair_idx, method="ffill").values.astype(float)
    except Exception:
        return None


def _confirms_at(i: int, want_long: bool, pair_closes: np.ndarray,
                 gold: np.ndarray | None, dxy: np.ndarray | None,
                 nk: np.ndarray | None) -> bool:
    if i - LB_SHORT < 0:
        return False
    p_now = pair_closes[i]
    p_prev = pair_closes[i - LB_SHORT]
    if p_prev <= 0:
        return False
    jr = np.log(p_now / p_prev)

    def _r(arr):
        if arr is None:
            return np.nan
        a = arr[i]; b = arr[i - LB_SHORT]
        if np.isnan(a) or np.isnan(b) or b <= 0:
            return np.nan
        return np.log(a / b)

    gr = _r(gold); dr = _r(dxy); nr = _r(nk)

    if want_long:
        if not np.isnan(gr) and jr < -MIN_MOVE and gr < -MIN_MOVE:
            return True
        if not np.isnan(dr) and jr < -MIN_MOVE and dr > MIN_MOVE:
            return True
        if not np.isnan(nr) and jr < -MIN_MOVE and nr > MIN_MOVE:
            return True
        return False
    else:
        if not np.isnan(gr) and jr > MIN_MOVE and gr > MIN_MOVE:
            return True
        if not np.isnan(dr) and jr > MIN_MOVE and dr < -MIN_MOVE:
            return True
        if not np.isnan(nr) and jr > MIN_MOVE and nr < -MIN_MOVE:
            return True
        return False


# ---------- outcome sim on warmup bars ----------

def _outcome(i: int, direction: int, highs: np.ndarray, lows: np.ndarray,
             closes: np.ndarray, max_ahead: int = 96) -> int | None:
    """Return 1 if TP hits first, 0 if SL first, None if neither within horizon."""
    entry = closes[i]
    if direction == 1:
        tp = entry + TP_PIPS * PIP
        sl = entry - SL_PIPS * PIP
    else:
        tp = entry - TP_PIPS * PIP
        sl = entry + SL_PIPS * PIP
    end = min(len(closes), i + 1 + max_ahead)
    for j in range(i + 1, end):
        h = highs[j]; l = lows[j]
        if direction == 1:
            hit_sl = l <= sl
            hit_tp = h >= tp
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp
        if hit_sl and hit_tp:
            return 0  # conservative
        if hit_sl:
            return 0
        if hit_tp:
            return 1
    return None


# ---------- features at an index ----------

def _feat_at(i: int, pair_closes: np.ndarray, z: np.ndarray,
             gold: np.ndarray | None, dxy: np.ndarray | None,
             vix: np.ndarray | None, hour_of_day: np.ndarray,
             direction: int) -> np.ndarray | None:
    if i - LB_SHORT < 0:
        return None
    p_now = pair_closes[i]
    p_prev = pair_closes[i - LB_SHORT]
    if p_prev <= 0:
        return None
    jr = np.log(p_now / p_prev)

    def _r(arr):
        if arr is None:
            return 0.0
        a = arr[i]; b = arr[i - LB_SHORT]
        if np.isnan(a) or np.isnan(b) or b <= 0:
            return 0.0
        return np.log(a / b)

    gr = _r(gold); dr = _r(dxy)
    v_raw = vix[i] if (vix is not None and not np.isnan(vix[i])) else 0.0
    zv = z[i] if not np.isnan(z[i]) else 0.0
    h = hour_of_day[i]

    # signed by direction so long and short share the same "favourable" direction
    s = float(direction)
    return np.array([
        abs(zv),
        s * jr,
        s * -dr,  # long-favourable = DXY down
        s * -gr,  # long-favourable = gold down (risk-off signal)
        v_raw,
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
    ], dtype=float)


# ---------- fit meta model on full pair warmup ----------

def _fit_meta(prices: dict) -> None:
    if _META["fit"]:
        return
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < 5000:
        return
    closes = pair["close"].values.astype(float)
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    z = _zscore_arr(closes, Z_LB)
    gold = _aligned_series(pair.index, prices.get("GC=F"))
    dxy = _aligned_series(pair.index, prices.get("DX-Y.NYB"))
    nk = _aligned_series(pair.index, prices.get("^N225"))
    vix = _aligned_series(pair.index, prices.get("^VIX"))
    hour = np.array([ts.hour for ts in pair.index])

    idx_arr, dir_arr = _v24_signal_idx(pair)
    if len(idx_arr) == 0:
        return

    X_list = []; y_list = []
    for i, d in zip(idx_arr, dir_arr):
        # must be far enough from end to have outcome labels
        if i >= len(closes) - 100:
            continue
        if not _confirms_at(int(i), d == 1, closes, gold, dxy, nk):
            continue
        out = _outcome(int(i), int(d), highs, lows, closes)
        if out is None:
            continue
        f = _feat_at(int(i), closes, z, gold, dxy, vix, hour, int(d))
        if f is None:
            continue
        if not np.all(np.isfinite(f)):
            continue
        X_list.append(f); y_list.append(out)

    if len(X_list) < 50 or len(set(y_list)) < 2:
        _META["fit"] = True
        return

    X = np.array(X_list); y = np.array(y_list)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(max_iter=500, C=0.5)
    model.fit(Xs, y)
    _META["fit"] = True
    _META["model"] = model
    _META["scaler"] = scaler
    _META["n_train"] = len(y)
    _META["train_pos_rate"] = float(y.mean())


# ---------- live trade ----------

def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    _fit_meta(prices)

    n = len(pair)
    if n < TREND_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # evaluate v24 at the last bar only
    closes = pair["close"].values.astype(float)
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    last = closes[-1]
    prev = closes[-1 - TREND_LB]
    if prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    trend = last / prev - 1.0

    # single-bar oscillators
    w = closes[-Z_LB:]
    sd = float(w.std(ddof=1)); z_last = (last - float(w.mean())) / sd if sd > 0 else 0.0
    rsi_tail = _rsi_arr(closes[-(RSI_LB + 5):], RSI_LB)
    rsi = rsi_tail[-1] if len(rsi_tail) else float("nan")
    wr_tail = _wr_arr(highs[-(WR_LB + 2):], lows[-(WR_LB + 2):], closes[-(WR_LB + 2):], WR_LB)
    wr = wr_tail[-1] if len(wr_tail) else float("nan")

    long_pull = (z_last < -Z_ENTRY) or (not np.isnan(rsi) and rsi < RSI_LOW) or (not np.isnan(wr) and wr < WR_LOW)
    short_pull = (z_last > Z_ENTRY) or (not np.isnan(rsi) and rsi > RSI_HIGH) or (not np.isnan(wr) and wr > WR_HIGH)

    d = 0
    if trend > 0 and long_pull:
        d = 1
    elif trend < 0 and short_pull:
        d = -1
    if d == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # cross-asset confirm at last bar
    gold = _aligned_series(pair.index[-300:], prices.get("GC=F"))
    dxy = _aligned_series(pair.index[-300:], prices.get("DX-Y.NYB"))
    nk = _aligned_series(pair.index[-300:], prices.get("^N225"))

    # local confirm on tail arrays (index -1 → LB_SHORT lookback)
    def _r_tail(arr):
        if arr is None or len(arr) < LB_SHORT + 1:
            return float("nan")
        a = arr[-1]; b = arr[-1 - LB_SHORT]
        if np.isnan(a) or np.isnan(b) or b <= 0:
            return float("nan")
        return float(np.log(a / b))

    p_tail_prev = closes[-1 - LB_SHORT]
    jr = float(np.log(last / p_tail_prev)) if p_tail_prev > 0 else 0.0
    gr = _r_tail(gold); dr = _r_tail(dxy); nr = _r_tail(nk)

    if d == 1:
        confirmed = (
            (not np.isnan(gr) and jr < -MIN_MOVE and gr < -MIN_MOVE)
            or (not np.isnan(dr) and jr < -MIN_MOVE and dr > MIN_MOVE)
            or (not np.isnan(nr) and jr < -MIN_MOVE and nr > MIN_MOVE)
        )
    else:
        confirmed = (
            (not np.isnan(gr) and jr > MIN_MOVE and gr > MIN_MOVE)
            or (not np.isnan(dr) and jr > MIN_MOVE and dr < -MIN_MOVE)
            or (not np.isnan(nr) and jr > MIN_MOVE and nr < -MIN_MOVE)
        )
    if not confirmed:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # meta filter
    model = _META.get("model"); scaler = _META.get("scaler")
    if model is not None and scaler is not None:
        vix_s = _aligned_series(pair.index[-5:], prices.get("^VIX"))
        v_raw = vix_s[-1] if (vix_s is not None and not np.isnan(vix_s[-1])) else 0.0
        h = pair.index[-1].hour
        gr_ = gr if not np.isnan(gr) else 0.0
        dr_ = dr if not np.isnan(dr) else 0.0
        s = float(d)
        feat = np.array([[
            abs(z_last),
            s * jr,
            s * -dr_,
            s * -gr_,
            v_raw,
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
        ]], dtype=float)
        if np.all(np.isfinite(feat)):
            p_tp = float(model.predict_proba(scaler.transform(feat))[0, 1])
            if p_tp < META_THRESH:
                return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    return {"direction": d, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
