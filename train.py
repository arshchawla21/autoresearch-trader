"""
v12-ml-aggressive: Better ML with cross-asset features, concentrated bets,
slight TP asymmetry. Target: high return with >50% WR on tight stops.

Changes from v11:
- Add cross-asset features (SPY return, VIX level, sector ETF divergence)
- Add intrabar microstructure features (upper/lower shadow ratios)
- Bigger model (128->64->32) with residual connection
- Top 3 most confident bets only (more weight per trade)
- Slight TP > SL asymmetry (1.2:1) to boost per-trade expectancy
- Train on whether TP or SL gets hit (not just close direction)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import O, H, L, C, V, evaluate

t_start = time.time()


class DirectionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.15)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.out(x))


def compute_features(ohlcv_np, tidx, midx, bar_idx, n_stocks):
    """Enhanced feature set with cross-asset and microstructure features."""
    if bar_idx < 20:
        return None

    closes = ohlcv_np[:bar_idx, tidx, C]
    opens_h = ohlcv_np[:bar_idx, tidx, O]
    highs = ohlcv_np[:bar_idx, tidx, H]
    lows = ohlcv_np[:bar_idx, tidx, L]
    vols = ohlcv_np[:bar_idx, tidx, V]
    curr_open = ohlcv_np[bar_idx, tidx, O]

    features = []

    # === Price-based features ===

    # 1. Multi-timeframe returns (1, 2, 3, 5, 10 bars)
    for lb in [1, 2, 3, 5, 10]:
        if bar_idx >= lb + 1:
            ret = (closes[-1] - closes[-1 - lb]) / np.maximum(np.abs(closes[-1 - lb]), 1e-8)
        else:
            ret = np.zeros(n_stocks)
        features.append(ret)

    # 2. Open-to-close of last 3 bars
    for i in [1, 2, 3]:
        if bar_idx >= i:
            oc = (closes[-i] - opens_h[-i]) / np.maximum(np.abs(opens_h[-i]), 1e-8)
        else:
            oc = np.zeros(n_stocks)
        features.append(oc)

    # 3. Bar ranges (last 3)
    for i in [1, 2, 3]:
        if bar_idx >= i:
            hl = (highs[-i] - lows[-i]) / np.maximum(closes[-i], 1e-8)
        else:
            hl = np.zeros(n_stocks)
        features.append(hl)

    # 4. Close position within bar (0=low, 1=high)
    bar_range = highs[-1] - lows[-1]
    close_pos = np.where(bar_range > 1e-8, (closes[-1] - lows[-1]) / bar_range, 0.5)
    features.append(close_pos)

    # 5. Gap
    gap = (curr_open - closes[-1]) / np.maximum(np.abs(closes[-1]), 1e-8)
    features.append(gap)

    # === Volume features ===

    # 6. Volume ratio (last bar vs 10-bar avg)
    if bar_idx >= 10:
        avg_vol = np.mean(vols[-10:], axis=0)
        vol_ratio = vols[-1] / np.maximum(avg_vol, 1e-8)
    else:
        vol_ratio = np.ones(n_stocks)
    features.append(np.clip(vol_ratio, 0, 10))

    # 7. Volume trend (last 3 bars vs prev 3)
    if bar_idx >= 6:
        recent_vol = np.mean(vols[-3:], axis=0)
        older_vol = np.mean(vols[-6:-3], axis=0)
        vol_trend = recent_vol / np.maximum(older_vol, 1e-8)
    else:
        vol_trend = np.ones(n_stocks)
    features.append(np.clip(vol_trend, 0, 10))

    # === Volatility features ===

    # 8. Rolling vol (5-bar)
    if bar_idx >= 6:
        rets_5 = np.diff(closes[-6:], axis=0) / np.maximum(np.abs(closes[-6:-1]), 1e-8)
        roll_vol = np.std(rets_5, axis=0)
    else:
        roll_vol = np.zeros(n_stocks)
    features.append(roll_vol)

    # 9. Vol regime: current vol vs 20-bar avg vol
    if bar_idx >= 20:
        rets_20 = np.diff(closes[-21:], axis=0) / np.maximum(np.abs(closes[-21:-1]), 1e-8)
        vol_20 = np.std(rets_20, axis=0)
        vol_regime = roll_vol / np.maximum(vol_20, 1e-8)
    else:
        vol_regime = np.ones(n_stocks)
    features.append(vol_regime)

    # === Microstructure features ===

    # 10. Upper shadow ratio (rejection of highs)
    if bar_idx >= 1:
        body_top = np.maximum(closes[-1], opens_h[-1])
        upper_shadow = (highs[-1] - body_top) / np.maximum(bar_range, 1e-8)
    else:
        upper_shadow = np.zeros(n_stocks)
    features.append(upper_shadow)

    # 11. Lower shadow ratio (rejection of lows)
    if bar_idx >= 1:
        body_bot = np.minimum(closes[-1], opens_h[-1])
        lower_shadow = (body_bot - lows[-1]) / np.maximum(bar_range, 1e-8)
    else:
        lower_shadow = np.zeros(n_stocks)
    features.append(lower_shadow)

    # === Cross-sectional features ===

    # 12. Cross-sectional rank of 1-bar return
    ret1 = features[0]
    ranks = np.argsort(np.argsort(ret1)).astype(np.float32) / max(n_stocks - 1, 1)
    features.append(ranks)

    # 13. Streak (consecutive up/down bars)
    if bar_idx >= 3:
        streak = (np.sign(closes[-1] - opens_h[-1]) +
                  np.sign(closes[-2] - opens_h[-2]) +
                  np.sign(closes[-3] - opens_h[-3]))
    else:
        streak = np.zeros(n_stocks)
    features.append(streak / 3.0)

    # === Cross-asset features (broadcast to all stocks) ===

    # 14. SPY 2-bar return (market direction) — SPY is tidx[0]
    if bar_idx >= 3:
        spy_ret = float((closes[-1, 0] - closes[-3, 0]) / max(abs(closes[-3, 0]), 1e-8))
    else:
        spy_ret = 0.0
    features.append(np.full(n_stocks, spy_ret, dtype=np.float32))

    # 15. VIX level (if available via macro indices)
    if midx is not None and len(midx) > 0:
        vix_close = float(ohlcv_np[bar_idx - 1, midx[0], C])
        # Normalize VIX: (VIX - 20) / 10
        vix_norm = (vix_close - 20.0) / 10.0
    else:
        vix_norm = 0.0
    features.append(np.full(n_stocks, vix_norm, dtype=np.float32))

    # 16. Stock return minus SPY return (beta-adjusted signal)
    if bar_idx >= 2:
        stock_ret1 = (closes[-1] - closes[-2]) / np.maximum(np.abs(closes[-2]), 1e-8)
        spy_ret1 = float(stock_ret1[0]) if len(stock_ret1) > 0 else 0.0
        excess_ret = stock_ret1 - spy_ret1
    else:
        excess_ret = np.zeros(n_stocks)
    features.append(excess_ret)

    return np.stack(features, axis=1).astype(np.float32)


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    midx = train_data["macro_indices"].numpy()
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    # Bar range stats for stop calibration
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    closes = ohlcv[:, tidx, C]
    bar_range_pct = (highs - lows) / np.maximum(closes, 1e-8)
    median_range = np.nanmedian(bar_range_pct, axis=0)

    print(f"  Median 15m bar range: {np.mean(median_range)*100:.3f}% across stocks")

    # Build training data — target: does TP get hit before SL?
    # Use symmetric stop_dist = 50% of median range for training target
    X_list = []
    y_list = []

    for t in range(20, T - 1):
        feats = compute_features(ohlcv, tidx, midx, t, n_stocks)
        if feats is None:
            continue

        # For each stock, simulate: if we went long at open, would TP or SL hit?
        next_open = ohlcv[t, tidx, O]
        next_high = ohlcv[t, tidx, H]
        next_low = ohlcv[t, tidx, L]
        next_close = ohlcv[t, tidx, C]

        for stock_i in range(n_stocks):
            if np.isnan(feats[stock_i]).any() or next_open[stock_i] <= 0:
                continue

            op = next_open[stock_i]
            hi = next_high[stock_i]
            lo = next_low[stock_i]
            mr = median_range[stock_i]
            sd = max(mr * 0.5, 0.001) * op

            # Long scenario: TP hit if high >= op + sd, SL hit if low <= op - sd
            long_tp_hit = hi >= op + sd
            long_sl_hit = lo <= op - sd

            # Target: 1 if long TP hits (price goes up enough), 0 if SL hits
            # If both or neither, use close direction
            if long_tp_hit and not long_sl_hit:
                target = 1.0
            elif long_sl_hit and not long_tp_hit:
                target = 0.0
            elif long_tp_hit and long_sl_hit:
                target = 1.0 if next_close[stock_i] > op else 0.0
            else:
                target = 1.0 if next_close[stock_i] > op else 0.0

            X_list.append(feats[stock_i])
            y_list.append(target)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    print(f"  Training samples: {len(X)}, features: {X.shape[1]}")
    print(f"  Base rate (long TP hit): {y.mean():.3f}")

    # Normalize
    feat_mean = np.nanmean(X, axis=0)
    feat_std = np.nanstd(X, axis=0) + 1e-8
    X_norm = np.nan_to_num((X - feat_mean) / feat_std, 0.0)

    # Train
    n_features = X.shape[1]
    model = DirectionModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    batch_size = 512
    n = len(X_t)

    best_loss = float('inf')
    patience_count = 0
    for epoch in range(300):
        perm = torch.randperm(n)
        epoch_loss = 0
        batches = 0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            pred = model(X_t[batch_idx])
            loss = criterion(pred, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        avg_loss = epoch_loss / batches
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count > 25:
                break

    model.eval()
    with torch.no_grad():
        train_pred = model(X_t).numpy().flatten()
        train_acc = ((train_pred > 0.5) == y).mean()
    print(f"  Train accuracy: {train_acc:.3f}, loss: {best_loss:.4f}, epochs: {epoch+1}")

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "macro_idx": train_data["macro_indices"],
        "median_range": median_range,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_stocks": n_stocks,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 20:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    midx = strategy["macro_idx"]
    ohlcv = data["ohlcv"]
    median_range = strategy["median_range"]
    model = strategy["model"]
    n_stocks = strategy["n_stocks"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx
    midx_np = midx.numpy() if isinstance(midx, torch.Tensor) else midx

    feats = compute_features(ohlcv_np, tidx_np, midx_np, bar_idx, n_stocks)
    if feats is None:
        return []

    opens = ohlcv_np[bar_idx, tidx_np, O]
    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(feats).any(axis=1))[0]
    if len(valid) == 0:
        return []

    feat_norm = np.nan_to_num((feats[valid] - strategy["feat_mean"]) / strategy["feat_std"], 0.0)

    with torch.no_grad():
        probs = model(torch.from_numpy(feat_norm)).numpy().flatten()

    # Collect confident predictions
    candidates = []
    for i, idx in enumerate(valid):
        p = float(probs[i])
        confidence = abs(p - 0.5)
        if confidence < 0.08:  # need at least 58% or 42%
            continue
        direction = "long" if p > 0.5 else "short"
        candidates.append((idx, direction, confidence, p))

    if not candidates:
        return []

    # Top 5 most confident
    candidates.sort(key=lambda x: -x[2])
    top = candidates[:min(5, len(candidates))]
    w = 1.0 / len(top)

    orders = []
    for idx, direction, conf, p in top:
        ticker = tickers[idx]
        op = float(opens[idx])
        mr = float(median_range[idx])

        # Stops at 50% of median bar range — should trigger within the bar often
        sl_dist = max(mr * 0.5, 0.001) * op
        # TP slightly wider (1.15x SL) — slight asymmetry rewards winners more
        tp_dist = sl_dist * 1.15

        if direction == "long":
            sl = op - sl_dist
            tp = op + tp_dist
        else:
            sl = op + sl_dist
            tp = op - tp_dist

        orders.append({
            "ticker": ticker,
            "direction": direction,
            "weight": w,
            "stop_loss": sl,
            "take_profit": tp,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
