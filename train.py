"""
Autoresearch-trader training script. Single-GPU, single-file.

Cross-sectional momentum with regime-adaptive sizing and calibrated stops.
Long top-ranked momentum stocks, short bottom-ranked. Market-neutral-ish
design for robustness across bull/bear/sideways regimes.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import numpy as np
import torch

from prepare import O, H, L, C, evaluate

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Strategy API: build_strategy + generate_orders
# ---------------------------------------------------------------------------

def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]
    D = ohlcv.shape[0]

    # Find VIX index
    vix_idx = None
    for j, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_idx = j
            break

    # Compute historical stats for stop/target calibration
    opens = ohlcv[:, tradeable_idx, O]
    closes = ohlcv[:, tradeable_idx, C]
    highs = ohlcv[:, tradeable_idx, H]
    lows = ohlcv[:, tradeable_idx, L]

    tr = torch.zeros(D, len(tickers))
    tr[1:] = torch.max(
        torch.max(highs[1:] - lows[1:], (highs[1:] - closes[:-1]).abs()),
        (lows[1:] - closes[:-1]).abs()
    )

    # Optimize stop/target multipliers on training data
    lookback_opt = min(500, D - 21)
    best_sharpe = -999
    best_stop_mult = 1.5
    best_target_mult = 1.5

    if lookback_opt >= 50:
        for stop_m in [0.8, 1.0, 1.2, 1.5, 2.0]:
            for target_m in [0.8, 1.0, 1.5, 2.0, 2.5]:
                sim_rets = []
                for d in range(D - lookback_opt, D):
                    if d < 21:
                        continue
                    atr_14 = tr[max(1, d-14):d].mean(dim=0)
                    atr_pct = (atr_14 / opens[d].clamp(min=1e-8)).clamp(0.003, 0.15)

                    close_10 = closes[d-10]
                    mom = (closes[d-1] - close_10) / close_10.clamp(min=1e-8)
                    ranks = mom.argsort(descending=True)
                    top_3 = ranks[:3]
                    bot_3 = ranks[-3:]

                    day_ret = 0.0
                    w = 1.0 / 6.0

                    for idx in top_3:
                        op = float(opens[d, idx])
                        hi = float(highs[d, idx])
                        lo = float(lows[d, idx])
                        cl = float(closes[d, idx])
                        ap = float(atr_pct[idx])
                        sl = op * (1 - ap * stop_m)
                        tp = op * (1 + ap * target_m)
                        s_hit = lo <= sl
                        l_hit = hi >= tp
                        if s_hit and l_hit:
                            ex = tp if cl > op else sl
                        elif l_hit:
                            ex = tp
                        elif s_hit:
                            ex = sl
                        else:
                            ex = cl
                        day_ret += w * ((ex - op) / op - 0.0002)

                    for idx in bot_3:
                        op = float(opens[d, idx])
                        hi = float(highs[d, idx])
                        lo = float(lows[d, idx])
                        cl = float(closes[d, idx])
                        ap = float(atr_pct[idx])
                        sl = op * (1 + ap * stop_m)
                        tp = op * (1 - ap * target_m)
                        s_hit = hi >= sl
                        l_hit = lo <= tp
                        if s_hit and l_hit:
                            ex = tp if cl < op else sl
                        elif l_hit:
                            ex = tp
                        elif s_hit:
                            ex = sl
                        else:
                            ex = cl
                        day_ret += w * ((op - ex) / op - 0.0002)

                    sim_rets.append(day_ret)

                sim_rets = np.array(sim_rets)
                if len(sim_rets) > 10 and sim_rets.std() > 1e-10:
                    sharpe = np.sqrt(252) * sim_rets.mean() / sim_rets.std()
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_stop_mult = stop_m
                        best_target_mult = target_m

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "vix_idx": vix_idx,
        "stop_mult": best_stop_mult,
        "target_mult": best_target_mult,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    vix_idx = strategy["vix_idx"]
    stop_mult = strategy["stop_mult"]
    target_mult = strategy["target_mult"]

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]

    # ATR over last 14 days
    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_atr = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs - lows, (highs - closes_atr).abs()),
        (lows - closes_atr).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.15)

    # 10-day momentum
    close_10 = ohlcv[day_idx - 10, tradeable_idx, C]
    mom_10d = (prev_close - close_10) / close_10.clamp(min=1e-8)

    # 20-day momentum
    close_20 = ohlcv[day_idx - 20, tradeable_idx, C]
    mom_20d = (prev_close - close_20) / close_20.clamp(min=1e-8)

    # Combined signal
    signal = 0.6 * mom_10d + 0.4 * mom_20d

    # Cross-sectional rank
    ranks = signal.argsort(descending=True)

    # VIX regime scaling
    if vix_idx is not None:
        vix = float(ohlcv[day_idx - 1, vix_idx, C])
        if vix > 35:
            scale = 0.5
        elif vix > 28:
            scale = 0.7
        else:
            scale = 1.0
    else:
        scale = 1.0

    n_each = 3
    top_n = ranks[:n_each]
    bot_n = ranks[-n_each:]
    weight_each = scale / (2 * n_each)

    orders = []

    for i in range(n_each):
        idx = int(top_n[i])
        op = float(today_open[idx])
        if op <= 0 or np.isnan(op):
            continue
        ap = float(atr_pct[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": weight_each,
            "stop_loss": op * (1 - ap * stop_mult),
            "take_profit": op * (1 + ap * target_mult),
        })

    for i in range(n_each):
        idx = int(bot_n[i])
        op = float(today_open[idx])
        if op <= 0 or np.isnan(op):
            continue
        ap = float(atr_pct[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "short",
            "weight": weight_each,
            "stop_loss": op * (1 + ap * stop_mult),
            "take_profit": op * (1 - ap * target_mult),
        })

    return orders


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")

    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"sharpe_ratio:     {results['sharpe_ratio']:.4f}")
    print(f"total_return:     {results['total_return']*100:.2f}%")
    print(f"max_drawdown:     {results['max_drawdown']*100:.2f}%")
    print(f"win_rate:         {results['win_rate']*100:.1f}%")
    print(f"avg_daily_trades: {results['avg_daily_trades']:.1f}")
    print(f"num_slices:       {results['num_slices']}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
