"""
v14-hybrid-ls-ml: L/S mean-reversion base + ML weight tilting.

Core insight: v6 equal-weight L/S = sharpe 12.72, the pure MR signal works.
ML alone fails (<50% WR). Solution: use MR for direction, train ML to predict
which MR trades are strongest. ML adjusts weights, not directions.

Additionally, train the target on actual per-stock MR profitability during
the training period — this is the most directly useful signal.

Features focus on what makes MR work better or worse:
- How extreme was the move (signal strength)
- Volatility regime (MR works better in normal vol)
- Volume context (low-vol moves revert more)
- Cross-sectional positioning
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import O, H, L, C, V, evaluate

t_start = time.time()


class MRQualityModel(nn.Module):
    """Predicts quality of a mean-reversion trade (higher = better)."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    midx = train_data["macro_indices"].numpy()
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    closes = ohlcv[:, tidx, C]
    opens_all = ohlcv[:, tidx, O]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    vols = ohlcv[:, tidx, V]
    bar_range_pct = (highs - lows) / np.maximum(closes, 1e-8)
    median_range = np.nanmedian(bar_range_pct, axis=0)

    print(f"  Median 15m bar range: {np.mean(median_range)*100:.3f}%")

    # Build training data: for each bar's MR trade, compute features + MR outcome
    X_list = []
    y_list = []

    for t in range(5, T - 1):
        # MR signal: 2-bar momentum
        if t < 3:
            continue
        momentum = (closes[t-1] - closes[t-3]) / np.maximum(np.abs(closes[t-3]), 1e-8)

        # Next bar: how much did the MR trade profit?
        next_open = ohlcv[t, tidx, O]
        next_close = ohlcv[t, tidx, C]
        next_high = ohlcv[t, tidx, H]
        next_low = ohlcv[t, tidx, L]

        for si in range(n_stocks):
            if np.isnan(momentum[si]) or next_open[si] <= 0:
                continue

            op = next_open[si]
            mr = median_range[si]
            sd = max(mr * 0.5, 0.001) * op

            # MR direction
            if momentum[si] > 0:
                # Short: TP below, SL above
                tp_hit = next_low[si] <= op - sd
                sl_hit = next_high[si] >= op + sd
            else:
                # Long: TP above, SL below
                tp_hit = next_high[si] >= op + sd
                sl_hit = next_low[si] <= op - sd

            # Target: 1 if TP hit (MR worked), 0 if SL hit
            if tp_hit and not sl_hit:
                target = 1.0
            elif sl_hit and not tp_hit:
                target = 0.0
            elif tp_hit and sl_hit:
                # Both hit — use close direction
                if momentum[si] > 0:
                    target = 1.0 if next_close[si] < op else 0.0
                else:
                    target = 1.0 if next_close[si] > op else 0.0
            else:
                # Neither hit — small signal, use close
                if momentum[si] > 0:
                    target = 1.0 if next_close[si] < op else 0.0
                else:
                    target = 1.0 if next_close[si] > op else 0.0

            # Features about the quality of this MR opportunity
            feats = []
            # 1. Absolute momentum magnitude (how extreme was the move)
            feats.append(abs(float(momentum[si])))
            # 2. Momentum magnitude relative to stock's vol
            if t >= 6:
                rets = np.diff(closes[t-6:t, si]) / np.maximum(np.abs(closes[t-6:t-1, si]), 1e-8)
                stock_vol = np.std(rets)
            else:
                stock_vol = 0.01
            feats.append(abs(float(momentum[si])) / max(stock_vol, 1e-8))
            # 3. Volume ratio
            if t >= 10:
                avg_v = np.mean(vols[t-10:t, si])
                feats.append(float(vols[t-1, si]) / max(avg_v, 1e-8))
            else:
                feats.append(1.0)
            # 4. Close position within bar (did it close near its low/high?)
            br = highs[t-1, si] - lows[t-1, si]
            feats.append(float((closes[t-1, si] - lows[t-1, si]) / max(br, 1e-8)))
            # 5. Cross-sectional rank of this stock's momentum
            mom_rank = float(np.sum(np.abs(momentum) < abs(momentum[si]))) / n_stocks
            feats.append(mom_rank)
            # 6. SPY momentum (market context)
            feats.append(float(momentum[0]))
            # 7. Bar range relative to median (is vol normal?)
            feats.append(float(bar_range_pct[t-1, si]) / max(float(median_range[si]), 1e-8))

            X_list.append(feats)
            y_list.append(target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    n_features = X.shape[1]
    print(f"  Training: {len(X)} samples, {n_features} features")
    print(f"  MR success rate: {y.mean():.3f}")

    # Normalize
    feat_mean = np.mean(X, axis=0)
    feat_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - feat_mean) / feat_std

    # Train
    model = MRQualityModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    n = len(X_t)
    best_loss = float('inf')
    patience = 0

    for epoch in range(150):
        perm = torch.randperm(n)
        total_loss = 0
        batches = 0
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            raw = model(X_t[idx])
            loss = criterion(raw, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg = total_loss / batches
        if avg < best_loss - 1e-5:
            best_loss = avg
            patience = 0
        else:
            patience += 1
            if patience > 15:
                break

    model.eval()
    with torch.no_grad():
        raw_pred = model(X_t).numpy().flatten()
        pred = (raw_pred > 0).astype(float)
        acc = (pred == y).mean()
    print(f"  Train acc: {acc:.3f}, epochs: {epoch+1}")

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "median_range": median_range,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_stocks": n_stocks,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 5:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    median_range = strategy["median_range"]
    model = strategy["model"]
    n_stocks = strategy["n_stocks"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]
    highs = ohlcv_np[:bar_idx, tidx_np, H]
    lows = ohlcv_np[:bar_idx, tidx_np, L]
    vols = ohlcv_np[:bar_idx, tidx_np, V]

    if bar_idx < 3:
        return []

    momentum = (closes[-1] - closes[-3]) / np.maximum(np.abs(closes[-3]), 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) < 4:
        return []

    # Compute ML features for each valid stock
    feats_list = []
    for si in valid:
        feats = []
        feats.append(abs(float(momentum[si])))
        if bar_idx >= 6:
            rets = np.diff(closes[-6:, si]) / np.maximum(np.abs(closes[-6:-1, si]), 1e-8)
            stock_vol = float(np.std(rets))
        else:
            stock_vol = 0.01
        feats.append(abs(float(momentum[si])) / max(stock_vol, 1e-8))
        if bar_idx >= 10:
            avg_v = float(np.mean(vols[-10:, si]))
            feats.append(float(vols[-1, si]) / max(avg_v, 1e-8))
        else:
            feats.append(1.0)
        br = float(highs[-1, si] - lows[-1, si])
        feats.append(float(closes[-1, si] - lows[-1, si]) / max(br, 1e-8))
        mom_rank = float(np.sum(np.abs(momentum) < abs(momentum[si]))) / n_stocks
        feats.append(mom_rank)
        feats.append(float(momentum[0]))
        bar_range_now = br / max(float(closes[-1, si]), 1e-8)
        feats.append(bar_range_now / max(float(median_range[si]), 1e-8))
        feats_list.append(feats)

    feats_arr = np.array(feats_list, dtype=np.float32)
    feats_norm = (feats_arr - strategy["feat_mean"]) / strategy["feat_std"]

    with torch.no_grad():
        quality_scores = model(torch.from_numpy(feats_norm)).numpy().flatten()

    # L/S market-neutral: rank by momentum, long the losers, short the winners
    # But weight by ML quality score (higher quality = more weight)
    ranked = valid[np.argsort(momentum[valid])]
    n = len(ranked)
    half = n // 2
    long_ids = ranked[:half]
    short_ids = ranked[half:]

    # Map valid index -> quality score
    valid_to_score = {int(v): float(quality_scores[i]) for i, v in enumerate(valid)}

    # Compute quality-weighted positions
    def get_weights(ids):
        scores = np.array([max(valid_to_score.get(int(i), 0), 0.1) for i in ids])
        return scores / scores.sum() * 0.5  # each side gets 50%

    long_w = get_weights(long_ids)
    short_w = get_weights(short_ids)

    orders = []
    for i, idx in enumerate(long_ids):
        op = float(opens[idx])
        mr = float(median_range[idx])
        sd = max(mr * 0.5, 0.001) * op
        orders.append({
            "ticker": tickers[idx], "direction": "long",
            "weight": float(long_w[i]),
            "stop_loss": op - sd, "take_profit": op + sd,
        })

    for i, idx in enumerate(short_ids):
        op = float(opens[idx])
        mr = float(median_range[idx])
        sd = max(mr * 0.5, 0.001) * op
        orders.append({
            "ticker": tickers[idx], "direction": "short",
            "weight": float(short_w[i]),
            "stop_loss": op + sd, "take_profit": op - sd,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
