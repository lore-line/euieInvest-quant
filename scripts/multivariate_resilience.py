#!/usr/bin/env python3
"""Multivariate regression: cliff_b + max_CAGR vs all features.

Takes per-symbol features from prior analyses (ATR, volume, mean-rev
metrics, Hurst, etc.) and runs:
  1. OLS regression with all features
  2. L1 (Lasso) for feature selection
  3. Reports coefficient significance + R²

Tests whether a multi-feature model dominates the single best feature
(vol_z_late, ρ_s = +0.54).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.backtest import dca_grid  # noqa: E402

START = "2022-09-15"
END = "2026-05-17"

MASTER_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
    "INJ-USD", "GRT-USD", "AAVE-USD", "UNI-USD",
    "NEAR-USD", "SUSHI-USD", "APT-USD", "TIA-USD",
]


def compute_all_features(snapshot_dir, symbol):
    bars = dca_grid.load_bars(snapshot_dir, symbol, 60, START, END)
    if bars.empty:
        return None
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    # ATR
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    atr_pct = float((atr / c).dropna().mean() * 100)
    # Volume
    dv = c * v
    daily_dv = dv.resample("1D").sum().dropna()
    mean_dv = float(daily_dv.mean())
    y = np.log(daily_dv.replace(0, np.nan).dropna().values)
    x = np.arange(len(y)) / 365
    vol_slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 10 else 0.0
    last_60d = daily_dv.tail(60).mean()
    vol_z = float((last_60d - daily_dv.mean()) / daily_dv.std()) if daily_dv.std() > 0 else 0.0
    # Returns
    rets = np.log(c / c.shift(1)).dropna()
    autocorr_1 = float(rets.autocorr(lag=1))
    autocorr_5 = float(rets.autocorr(lag=5))
    rets_skew = float(rets.skew())
    # Drawdown
    rolling_max = c.rolling(200, min_periods=1).max()
    drawdown = c / rolling_max - 1
    max_dd = float(drawdown.min())
    # Sharpe
    daily_rets = rets.resample("1D").sum().dropna()
    sharpe_60d = float(daily_rets.tail(60).mean() / daily_rets.tail(60).std() * np.sqrt(365)) if len(daily_rets) >= 60 else 0.0
    # Variance ratio
    if len(rets) > 40 and rets.var() > 0:
        vr5 = float(rets.rolling(5).sum().dropna().var() / (5 * rets.var()))
    else:
        vr5 = 1.0
    return {
        "atr_pct": atr_pct,
        "log_dv": np.log(mean_dv),
        "vol_slope_yr": vol_slope,
        "vol_z_late": vol_z,
        "autocorr_1": autocorr_1,
        "autocorr_5": autocorr_5,
        "rets_skew": rets_skew,
        "max_dd": max_dd,
        "sharpe_60d": sharpe_60d,
        "vr_5": vr5,
    }


def ols_with_stats(X, y):
    """Returns coefficients, std errors, t-stats, p-values, R²."""
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    coef = XtX_inv @ X.T @ y
    y_hat = X @ coef
    residuals = y - y_hat
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else 0
    # Adjusted R²
    if n - k - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    else:
        adj_r2 = r2
    # Std errors
    sigma2 = rss / (n - k) if n > k else 1
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    # t-stats
    t_stats = np.abs(coef) / np.where(se > 0, se, 1e-9)
    return coef, se, t_stats, r2, adj_r2


def main() -> int:
    cliff_data = json.loads((ROOT / "reports" / "per-symbol-cliff-vs-atr.json").read_text())
    cliff_by_sym = {r["symbol"]: r for r in cliff_data["summary"]}

    snapshot_dir = dca_grid.SNAPSHOT_DIR
    rows = []
    for sym in MASTER_UNIVERSE:
        f = compute_all_features(snapshot_dir, sym)
        if f is None or sym not in cliff_by_sym:
            continue
        r = cliff_by_sym[sym]
        rows.append({"symbol": sym,
                     "cliff_b": r["cliff_b_pct"],
                     "max_cagr": r["max_cagr_pct"],
                     **f})

    feature_names = ["atr_pct", "log_dv", "vol_slope_yr", "vol_z_late",
                     "autocorr_1", "autocorr_5", "rets_skew",
                     "max_dd", "sharpe_60d", "vr_5"]
    print(f"Multivariate regression — n={len(rows)}, features={len(feature_names)}")
    print()

    # Build X matrix (with intercept) and standardize features
    X_raw = np.array([[r[f] for f in feature_names] for r in rows])
    X_means = X_raw.mean(axis=0)
    X_stds = X_raw.std(axis=0, ddof=0)
    X_stds[X_stds == 0] = 1.0
    X_std = (X_raw - X_means) / X_stds
    X = np.column_stack([np.ones(len(rows)), X_std])  # add intercept

    for target_name in ["cliff_b", "max_cagr"]:
        y = np.array([r[target_name] for r in rows])
        coef, se, t_stats, r2, adj_r2 = ols_with_stats(X, y)
        print(f"\n{'='*70}")
        print(f"OLS: {target_name} ~ all features (standardized)")
        print(f"R² = {r2:.3f}  adj-R² = {adj_r2:.3f}  n = {len(rows)}")
        print(f"{'='*70}")
        print(f"{'feature':<18s} | {'coef':>9s} | {'std err':>9s} | {'|t|':>6s}")
        print("-" * 50)
        names = ["intercept"] + feature_names
        # Sort by |t| descending (excluding intercept)
        order = sorted(range(1, len(names)), key=lambda i: -t_stats[i])
        # Show intercept first
        print(f"{names[0]:<18s} | {coef[0]:>+8.3f} | {se[0]:>8.3f} | {t_stats[0]:>5.2f}")
        for i in order:
            sig = " *" if t_stats[i] >= 2 else ("  ." if t_stats[i] >= 1 else "")
            print(f"{names[i]:<18s} | {coef[i]:>+8.3f} | {se[i]:>8.3f} | {t_stats[i]:>5.2f}{sig}")

        # Bivariate Pearson correlation as reference
        print(f"\nUnivariate Pearson correlations with {target_name}:")
        for fname in feature_names:
            vals = np.array([r[fname] for r in rows])
            if vals.std() > 0:
                rho = float(np.corrcoef(vals, y)[0, 1])
                print(f"  {fname:<18s}: ρ = {rho:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
