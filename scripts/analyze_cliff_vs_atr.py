#!/usr/bin/env python3
"""Cliff position vs universe mean ATR%.

Doctrine §2.7 cliff rule: b_cliff × N ≈ 6%. But the empirical table shows
the constant varies — small-N universes (BTC+ETH) tolerate b×N up to 20%,
while large-N (with thin alts mixed in) tighten to ~5-6%.

Hypothesis: the cliff constant depends on the MEAN ATR% of the universe.
Higher ATR → larger swings → SO ladders fill faster → cliff at lower b.

This script:
  1. Loads 60m bars for the 20-symbol universe
  2. Computes per-symbol mean ATR% over the backtest window
  3. For each N ∈ {2,4,6,8,10,12,16,20}, computes the mean ATR of the top-N
  4. Reads the cliff position from cliff-at-500k-friction.json
  5. Tabulates: N | mean_ATR% | cliff_b% | b×N | implied K = (b×N) × mean_ATR
  6. Fits: log(b_cliff) = α + β·log(N) + γ·log(mean_ATR)
"""

from __future__ import annotations

import argparse
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

# Same ordering as the cliff sweep — symbols added in this order as N grows.
MASTER_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
    "INJ-USD", "GRT-USD", "AAVE-USD", "UNI-USD",
    "NEAR-USD", "SUSHI-USD", "APT-USD", "TIA-USD",
]


def mean_atr_pct(bars: pd.DataFrame, period: int = 14) -> float:
    """ATR% as fraction of close, averaged over the window.

    True range = max(high - low, |high - prev_close|, |low - prev_close|).
    Then EMA-smoothed over `period` bars, divided by close, averaged.
    """
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    atr_pct = (atr / c) * 100
    return float(atr_pct.dropna().mean())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--cliff-json",
                   default="reports/cliff-at-500k-friction.json")
    args = p.parse_args()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR

    print(f"Loading 60m bars for ATR computation...")
    atr_per_sym: dict[str, float] = {}
    for sym in MASTER_UNIVERSE:
        bars = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
        if bars.empty:
            print(f"  [warn] no 60m bars for {sym}, skipping")
            continue
        atr_per_sym[sym] = mean_atr_pct(bars)
        print(f"  {sym}: mean ATR(14) = {atr_per_sym[sym]:.2f}%")

    cliff_path = ROOT / args.cliff_json
    if not cliff_path.exists():
        print(f"\nNo cliff JSON at {cliff_path} — need v1-only cliff sweep results.")
        return 1
    cliff_data = json.loads(cliff_path.read_text())

    # Extract cliff position per N from the v1-only sweep
    n_to_cliff: dict[int, dict] = {}
    for r in cliff_data["results"]:
        N = r["N"]
        positive = [c for c in r["cells"] if c["cagr_pct"] > 0]
        if not positive:
            continue
        cliff = max(positive, key=lambda c: c["base_pct"])
        max_cagr = max(r["cells"], key=lambda c: c["cagr_pct"])
        n_to_cliff[N] = {
            "cliff_b_pct": cliff["base_pct"] * 100,
            "max_cagr_pct": max_cagr["cagr_pct"],
            "b_at_max_cagr_pct": max_cagr["base_pct"] * 100,
        }

    print(f"\n{'='*88}")
    print("CLIFF vs MEAN ATR")
    print(f"{'='*88}")
    print(f"{'N':>3s} | {'symbols added':<30s} | {'mean ATR%':>9s} | "
          f"{'cliff b%':>9s} | {'b × N':>7s} | {'(b×N) × ATR':>11s}")
    print("-" * 88)

    rows = []
    for N in sorted(n_to_cliff.keys()):
        symbols = MASTER_UNIVERSE[:N]
        added = MASTER_UNIVERSE[max(0, N-2):N] if N > 2 else symbols
        atrs = [atr_per_sym[s] for s in symbols if s in atr_per_sym]
        if not atrs:
            continue
        mean_atr = float(np.mean(atrs))
        cliff = n_to_cliff[N]
        b_x_n = cliff["cliff_b_pct"] * N
        b_x_n_x_atr = b_x_n * mean_atr
        rows.append({
            "N": N,
            "mean_atr": mean_atr,
            "cliff_b": cliff["cliff_b_pct"],
            "b_x_n": b_x_n,
            "b_x_n_x_atr": b_x_n_x_atr,
        })
        added_str = ",".join(added).replace("-USD", "")
        print(f"{N:>3d} | {added_str:<30s} | {mean_atr:>8.2f}% | "
              f"{cliff['cliff_b_pct']:>8.3f}% | {b_x_n:>6.2f}% | "
              f"{b_x_n_x_atr:>10.2f}")

    if len(rows) >= 4:
        # Fit log-log regression: log(cliff_b) = α + β·log(N) + γ·log(mean_atr)
        n_arr = np.log(np.array([r["N"] for r in rows]))
        atr_arr = np.log(np.array([r["mean_atr"] for r in rows]))
        cliff_arr = np.log(np.array([r["cliff_b"] for r in rows]))
        # Solve via least squares
        X = np.column_stack([np.ones(len(rows)), n_arr, atr_arr])
        coef, *_ = np.linalg.lstsq(X, cliff_arr, rcond=None)
        alpha, beta, gamma = coef
        print(f"\nLog-log regression: cliff_b ≈ exp({alpha:.3f}) × N^({beta:.3f}) × mean_ATR^({gamma:.3f})")
        K = float(np.exp(alpha))
        print(f"  K = {K:.3f}, β (N exponent) = {beta:.3f}, γ (ATR exponent) = {gamma:.3f}")
        if abs(beta + 1) < 0.3:
            print(f"  β ≈ -1 confirms the inverse-N relationship from doctrine §2.7")
        if gamma < -0.2:
            print(f"  γ < 0 confirms HIGHER ATR → LOWER cliff (hypothesis supported)")
        elif gamma > 0.2:
            print(f"  γ > 0 means HIGHER ATR → HIGHER cliff (hypothesis CONTRADICTED)")
        else:
            print(f"  γ ≈ 0 means ATR doesn't materially affect cliff position")

        # Also compute residuals for the simple b×N=K rule (β=-1, γ=0)
        simple_K = np.array([r["cliff_b"] * r["N"] for r in rows])
        print(f"\nSimple b × N rule values: {simple_K}")
        print(f"  Mean: {simple_K.mean():.2f}%, Std: {simple_K.std():.2f}%")

        # And b × N × ATR
        bna = np.array([r["b_x_n_x_atr"] for r in rows])
        print(f"\nb × N × ATR values:       {bna}")
        print(f"  Mean: {bna.mean():.2f}, Std: {bna.std():.2f}")
        print(f"  Coefficient of variation: {bna.std()/bna.mean()*100:.1f}% "
              f"(vs {simple_K.std()/simple_K.mean()*100:.1f}% for simple b×N)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
