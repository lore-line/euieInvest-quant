#!/usr/bin/env python3
"""Per-symbol volume vs cliff position.

Tests whether dollar-volume (or volume slope, recent volume z-score)
predicts the bimodal cliff pattern in per-symbol-cliff-vs-atr.json.

Hypothesis: liquid symbols (high volume) execute the DCA strategy
cleanly → high cliff tolerance. Low-volume symbols experience
slippage / partial fills / wider spreads → degraded execution → low
cliff. Or: declining-volume symbols are losing adoption → trended-
down regimes → low cliff.
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


def main() -> int:
    cliff_path = ROOT / "reports" / "per-symbol-cliff-vs-atr.json"
    cliff_data = json.loads(cliff_path.read_text())
    cliff_by_sym = {r["symbol"]: r for r in cliff_data["summary"]}

    snapshot_dir = dca_grid.SNAPSHOT_DIR
    hdr_sym = "symbol"
    hdr_b = "cliff b%"
    hdr_dv = "mean $vol/day"
    hdr_slope = "vol_slope/yr"
    hdr_z = "vol_z_late"
    print(f"{hdr_sym:<10s} | {hdr_b:>9s} | {hdr_dv:>15s} | {hdr_slope:>13s} | {hdr_z:>11s}")
    print("-" * 80)
    rows = []
    for sym in cliff_by_sym:
        bars = dca_grid.load_bars(snapshot_dir, sym, 60, "2022-09-15", "2026-05-17")
        if bars.empty:
            continue
        dv = bars["close"] * bars["volume"]
        daily_dv = dv.resample("1D").sum().dropna()
        mean_dv = float(daily_dv.mean())
        y = np.log(daily_dv.replace(0, np.nan).dropna().values)
        x = np.arange(len(y)) / 365
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 10 else float("nan")
        last_60d_mean = daily_dv.tail(60).mean()
        z = (last_60d_mean - daily_dv.mean()) / daily_dv.std() if daily_dv.std() > 0 else 0
        cliff_b = cliff_by_sym[sym].get("cliff_b_pct", 0)
        print(f"{sym:<10s} | {cliff_b:>8.2f}% | ${mean_dv/1e6:>13.1f}M | "
              f"{slope:>+12.2f} | {z:>+10.2f}")
        rows.append({"symbol": sym, "cliff_b": cliff_b,
                     "mean_dv": mean_dv, "vol_slope_yr": slope,
                     "vol_z_late": z})

    cliff = np.array([r["cliff_b"] for r in rows])
    dv = np.array([r["mean_dv"] for r in rows])
    log_dv = np.log(dv)
    slope = np.array([r["vol_slope_yr"] for r in rows])
    z = np.array([r["vol_z_late"] for r in rows])

    def pearson(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = pd.Series(a).rank().values
        rb = pd.Series(b).rank().values
        if ra.std() == 0 or rb.std() == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\n=== Pearson correlations vs cliff_b ===")
    print(f"  mean_dv:        rho = {pearson(cliff, dv):+.3f}")
    print(f"  log(mean_dv):   rho = {pearson(cliff, log_dv):+.3f}")
    print(f"  vol_slope_yr:   rho = {pearson(cliff, slope):+.3f}")
    print(f"  vol_z_late:     rho = {pearson(cliff, z):+.3f}")

    print(f"\n=== Spearman (rank) correlations vs cliff_b ===")
    print(f"  mean_dv:        rho_s = {spearman(cliff, dv):+.3f}")
    print(f"  log(mean_dv):   rho_s = {spearman(cliff, log_dv):+.3f}")
    print(f"  vol_slope_yr:   rho_s = {spearman(cliff, slope):+.3f}")
    print(f"  vol_z_late:     rho_s = {spearman(cliff, z):+.3f}")

    sorted_by_cliff = sorted(rows, key=lambda r: r["cliff_b"], reverse=True)
    n_half = len(sorted_by_cliff) // 2
    high = sorted_by_cliff[:n_half]
    low = sorted_by_cliff[n_half:]
    print(f"\n=== High-cliff (n={len(high)}) vs Low-cliff (n={len(low)}) means ===")
    hi_dv = np.mean([r["mean_dv"] for r in high]) / 1e6
    lo_dv = np.mean([r["mean_dv"] for r in low]) / 1e6
    print(f"  mean $vol/day:  high=${hi_dv:.0f}M  low=${lo_dv:.0f}M")
    print(f"  vol_slope_yr:   high={np.mean([r['vol_slope_yr'] for r in high]):+.2f}  "
          f"low={np.mean([r['vol_slope_yr'] for r in low]):+.2f}")
    print(f"  vol_z_late:     high={np.mean([r['vol_z_late'] for r in high]):+.2f}  "
          f"low={np.mean([r['vol_z_late'] for r in low]):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
