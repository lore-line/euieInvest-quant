#!/usr/bin/env python3
"""Per-symbol resilience metrics v2 — drowning the bimodal cliff in
classical mean-reversion + drawdown-recovery measures.

Recovery-quality v1 found weak correlations (max ρ ≈ 0.4 from
sharpe_60d_terminal, recency-biased). This script adds:

  1. Hurst exponent (R/S, log-rescaled-range) — classical mean-rev measure
     - H < 0.5: mean-reverting (good for DCA)
     - H = 0.5: random walk
     - H > 0.5: trending (bad for DCA)
  2. Variance ratio at multiple lags
     - VR < 1: mean-reverting at that horizon
     - VR > 1: trending
  3. Conditional recovery probability
     - P(price returns to entry within 20 bars | dropped -5%)
     - Captures the EXACT thing DCA strategy needs
  4. P(catastrophic continuation)
     - P(price drops -30% from entry | first dropped -10%)
     - High value = the strategy gets stranded often
  5. Half-life of mean-reversion
     - OU process fit: half-life of return to mean

These are price-derived (no simulation needed) so they can be screener
features for the harvestability scorer (doctrine §2.7 T2/T3 roadmap).
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


def hurst_rs(series: pd.Series, max_lag: int = 100) -> float:
    """R/S Hurst exponent. Returns H in [0,1]. H=0.5 is random walk.

    Loglog regression of (R/S) vs lag, slope = H.
    """
    lags = np.unique(np.geomspace(2, max_lag, num=15).astype(int))
    rs = []
    valid_lags = []
    for lag in lags:
        if len(series) < lag * 2:
            continue
        # Split into non-overlapping windows of size `lag`
        n_wins = len(series) // lag
        if n_wins < 2:
            continue
        rs_vals = []
        for w in range(n_wins):
            seg = series.iloc[w*lag:(w+1)*lag].values
            mean = seg.mean()
            dev = seg - mean
            cumdev = dev.cumsum()
            R = cumdev.max() - cumdev.min()
            S = seg.std()
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs.append(np.mean(rs_vals))
            valid_lags.append(lag)
    if len(rs) < 3:
        return 0.5
    slope = np.polyfit(np.log(valid_lags), np.log(rs), 1)[0]
    return float(slope)


def variance_ratio(returns: pd.Series, lag: int) -> float:
    """VR = var(k-period rets) / (k × var(1-period rets)).
    VR < 1: mean-reverting. VR = 1: random walk. VR > 1: trending."""
    if len(returns) < lag * 2:
        return 1.0
    var_1 = returns.var()
    if var_1 == 0:
        return 1.0
    k_rets = returns.rolling(lag).sum().dropna()
    var_k = k_rets.var()
    return float(var_k / (lag * var_1))


def conditional_recovery(close: pd.Series, drop_pct: float = 0.05,
                          recovery_bars: int = 20) -> float:
    """P(close returns to entry within `recovery_bars` | dropped by `drop_pct`).

    For each bar t, simulate an entry. Find first bar t' > t where price
    crossed (1 - drop_pct) × close[t]. From t' on, check if price returns
    to close[t] within `recovery_bars`. Average across all such episodes.
    """
    c = close.values
    n = len(c)
    successes = 0
    attempts = 0
    for t in range(n - recovery_bars * 5):
        entry = c[t]
        threshold = entry * (1 - drop_pct)
        # Find first drop below threshold within reasonable window
        future = c[t+1:t+1+recovery_bars*5]
        below = np.where(future <= threshold)[0]
        if len(below) == 0:
            continue
        drop_idx = below[0] + t + 1
        recovery_window = c[drop_idx:drop_idx + recovery_bars]
        if len(recovery_window) == 0:
            continue
        attempts += 1
        if recovery_window.max() >= entry:
            successes += 1
    return successes / attempts if attempts > 0 else 0.0


def catastrophic_continuation(close: pd.Series,
                              first_drop: float = 0.10,
                              cata_drop: float = 0.30,
                              lookback_bars: int = 200) -> float:
    """P(price drops -cata_drop% | first dropped first_drop% from entry).

    For each bar t with a -first_drop drawdown from some prior peak:
    does it continue to -cata_drop within `lookback_bars`? Higher value
    means strategy is more likely to be stranded on its DCA ladder.
    """
    c = close.values
    n = len(c)
    rolling_max = pd.Series(c).rolling(window=lookback_bars, min_periods=1).max().values
    drawdown = c / rolling_max - 1
    # Bars in first-drop state
    first_drop_idx = np.where((drawdown <= -first_drop) & (drawdown > -cata_drop))[0]
    if len(first_drop_idx) == 0:
        return 0.0
    cata_count = 0
    total = 0
    for t in first_drop_idx:
        # Did we hit cata_drop within next `lookback_bars`?
        future_dd = drawdown[t:t + lookback_bars]
        if len(future_dd) == 0:
            continue
        total += 1
        if future_dd.min() <= -cata_drop:
            cata_count += 1
    return cata_count / total if total > 0 else 0.0


def ou_half_life(close: pd.Series) -> float:
    """Half-life of mean reversion via OU process fit on log-price.

    Fits Δlog(p)_t = -θ · (log(p)_{t-1} - μ) + ε. Half-life = ln(2) / θ.
    Returns inf if non-mean-reverting (θ ≤ 0).
    """
    lp = np.log(close.values)
    dlp = np.diff(lp)
    lp_lag = lp[:-1]
    # Regress dlp ~ lp_lag (OLS)
    X = np.column_stack([np.ones(len(lp_lag)), lp_lag])
    try:
        coef, *_ = np.linalg.lstsq(X, dlp, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    theta = -coef[1]
    if theta <= 0:
        return float("inf")
    return float(np.log(2) / theta)


def main() -> int:
    cliff_path = ROOT / "reports" / "per-symbol-cliff-vs-atr.json"
    cliff_data = json.loads(cliff_path.read_text())
    cliff_by_sym = {r["symbol"]: r for r in cliff_data["summary"]}

    print(f"Resilience metrics v2 — testing classical mean-reversion measures")
    print(f"Source cliff data: {len(cliff_by_sym)} symbols")
    print()

    snapshot_dir = dca_grid.SNAPSHOT_DIR
    print(f"{'symbol':<10s} | {'cliff b%':>9s} | {'Hurst':>6s} | "
          f"{'VR(5)':>6s} | {'VR(20)':>6s} | {'P_rec_5':>8s} | {'P_cata':>7s} | "
          f"{'half_life':>10s}")
    print("-" * 90)

    rows = []
    for sym in MASTER_UNIVERSE:
        bars = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
        if bars.empty:
            continue
        c = bars["close"]
        rets = np.log(c / c.shift(1)).dropna()

        h = hurst_rs(c.tail(5000))  # tail-window for speed
        vr5 = variance_ratio(rets, 5)
        vr20 = variance_ratio(rets, 20)
        p_rec = conditional_recovery(c, drop_pct=0.05, recovery_bars=20)
        p_cata = catastrophic_continuation(c, first_drop=0.10, cata_drop=0.30)
        hl = ou_half_life(c)
        hl_str = f"{hl:.1f}" if np.isfinite(hl) else "inf"

        cliff = cliff_by_sym.get(sym, {})
        cliff_b = cliff.get("cliff_b_pct", None)
        cliff_str = f"{cliff_b:.2f}%" if cliff_b is not None else "?"

        print(f"{sym:<10s} | {cliff_str:>9s} | {h:>5.3f} | {vr5:>5.3f} | "
              f"{vr20:>5.3f} | {p_rec:>7.3f} | {p_cata:>6.3f} | {hl_str:>10s}")
        rows.append({
            "symbol": sym, "cliff_b": cliff_b,
            "hurst": h, "vr_5": vr5, "vr_20": vr20,
            "p_recovery_5pct": p_rec, "p_catastrophic_30pct": p_cata,
            "ou_half_life": hl if np.isfinite(hl) else None,
        })

    valid = [r for r in rows if r["cliff_b"] is not None]
    if len(valid) >= 5:
        cliff_arr = np.array([r["cliff_b"] for r in valid])
        print(f"\n{'='*70}")
        print("CORRELATIONS — cliff_b vs each metric")
        print(f"{'='*70}")
        for metric in ["hurst", "vr_5", "vr_20", "p_recovery_5pct",
                       "p_catastrophic_30pct"]:
            vals = np.array([r[metric] for r in valid])
            if vals.std() == 0:
                continue
            r_pearson = float(np.corrcoef(cliff_arr, vals)[0, 1])
            sign = "+" if r_pearson >= 0 else ""
            print(f"  cliff_b vs {metric:<24s}: ρ = {sign}{r_pearson:.3f}")

        # Spearman (rank) correlation — robust to outliers / non-linear
        print(f"\nSpearman (rank) correlations (more robust):")
        cliff_ranks = pd.Series(cliff_arr).rank().values
        for metric in ["hurst", "vr_5", "vr_20", "p_recovery_5pct",
                       "p_catastrophic_30pct"]:
            vals = np.array([r[metric] for r in valid])
            if vals.std() == 0:
                continue
            metric_ranks = pd.Series(vals).rank().values
            r_spearman = float(np.corrcoef(cliff_ranks, metric_ranks)[0, 1])
            sign = "+" if r_spearman >= 0 else ""
            print(f"  cliff_b vs {metric:<24s}: ρ_s = {sign}{r_spearman:.3f}")

        # High-cliff vs low-cliff comparison
        sorted_by_cliff = sorted(valid, key=lambda r: r["cliff_b"], reverse=True)
        n_half = len(sorted_by_cliff) // 2
        high = sorted_by_cliff[:n_half]
        low = sorted_by_cliff[n_half:]
        print(f"\nHigh-cliff (n={len(high)}) vs Low-cliff (n={len(low)}) means:")
        print(f"{'metric':<24s} | {'high':>8s} | {'low':>8s} | {'Δ':>7s}")
        print("-" * 60)
        for metric in ["hurst", "vr_5", "vr_20", "p_recovery_5pct",
                       "p_catastrophic_30pct"]:
            hi = np.mean([r[metric] for r in high])
            lo = np.mean([r[metric] for r in low])
            print(f"  {metric:<22s} | {hi:>+7.3f} | {lo:>+7.3f} | "
                  f"{hi-lo:>+6.3f}")

    out = ROOT / "reports" / "resilience-v2.json"
    out.write_text(json.dumps({"rows": rows}, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
