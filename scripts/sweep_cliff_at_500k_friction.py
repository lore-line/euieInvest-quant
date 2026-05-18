#!/usr/bin/env python3
"""Cliff topology at $500K-tier friction (0.16% RT, 0.08% maker).

Doctrine §2.7 cliff numbers were measured at $0-tier friction (the
backtest default — rolling 30d volume starts at zero). Real deployment
at Stage 1+ (monthly notional ≥ $500K) operates at 4-5× better per-deal
economics. Question: does the cliff topology (b_cliff × N ≈ 6%, N=10
sweet spot, plateau qualification) hold at deployment-scale friction,
or does lower friction shift where it's economic to deploy capital?

Hypothesis: cliff POSITION is friction-independent (it's a
capital-deployment + correlation phenomenon — friction is fractions of
percent per fill, dwarfed by drawdown depth at the cliff). But max CAGR
at and below the cliff should scale upward with lower friction.

Uses the simulator's `fixed_friction_vol_30d` param to pin every fee
lookup at $500K-tier from t=0, isolating the topology question from
the volume-buildup transient.

Sweep: N ∈ {2, 4, 6, 8, 10, 12, 16, 20} × base_pct grid covering each N's
expected cliff position.

Output: reports/cliff-at-500k-friction.json + side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo-root resolution: this script lives at <repo>/scripts/, simulator at
# <repo>/src/quant/backtest/dca_grid.py. Add src/ to path so the module
# imports without needing `uv run` to set PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.backtest import dca_grid  # noqa: E402

START = "2022-09-15"
END = "2026-05-17"
STARTING_CAPITAL = 3000.0
FIXED_FRICTION_VOL_30D = 500_000.0  # pins fee lookups at $500K tier

# Master universe sorted roughly by liquidity tier (matches prior sweeps
# on euieInvest's `reports/cliff-low-N.json` + `cliff-by-universe-size.json`).
MASTER_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
    "INJ-USD", "GRT-USD", "AAVE-USD", "UNI-USD",
    "NEAR-USD", "SUSHI-USD", "APT-USD", "TIA-USD",
]

# v1 Optuna best params (from euieInvest:reports/crypto-dca-v1-100trials.json).
# Pinned here so the sweep is self-contained on the euieInvest-quant side.
# so_volume_scale overridden to 2.30 per N=10 sweet-spot finding.
V1_OPTUNA_PARAMS = {
    "n_safety_orders": 9,
    "first_so_step_pct": 0.025747011371995105,  # 2.57%
    "so_step_scale": 1.6942997249142477,
    "so_volume_scale": 2.30,
    "strand_ban_days": 122,
    "is_taker": False,
    "early_sl_pct": None,
    "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
}


def load_signals_once(snapshot_dir: Path, symbols: list[str], version: int = 1):
    bars_per_sv: dict = {}
    signals_per_sv: dict = {}
    tf_min = dca_grid.get_native_tf_min(version)
    for sym in symbols:
        bars_1h = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
        bars = dca_grid.load_bars(snapshot_dir, sym, tf_min, START, END)
        if bars.empty or bars_1h.empty:
            continue
        sigs = dca_grid.generate_entry_signals(bars, version, bars_1h)
        bars_per_sv[(sym, version)] = bars
        signals_per_sv[(sym, version)] = sigs
    return bars_per_sv, signals_per_sv


def run_sweep_at_N(snapshot_dir: Path, N: int, base_pcts: list[float],
                   base_params: dict) -> dict:
    symbols = MASTER_UNIVERSE[:N]
    print(f"\n{'='*72}")
    print(f"N={N}: {symbols}")
    print(f"{'='*72}")

    bars_per_sv, signals_per_sv = load_signals_once(snapshot_dir, symbols, version=1)
    n_active = len(bars_per_sv)
    print(f"Loaded {n_active} (sym, ver) pairs\n")

    print(f"{'base_pct':>9s} | {'deals':>6s} | {'eod':>4s} | {'CAGR':>8s} | "
          f"{'final $':>9s} | {'peak %':>7s} | {'b×N':>7s}")
    print("-" * 75)

    cells = []
    for bp in base_pcts:
        params = {**base_params,
                  "base_order_usd": 0,
                  "base_pct_of_equity": bp}
        result = dca_grid.simulate_portfolio(
            signals_per_sv, bars_per_sv, params, STARTING_CAPITAL)
        m = dca_grid.compute_metrics(result)
        closed = result["closed_deals"]
        events = []
        for d in closed:
            events.append((d.deal.opened_at, d.deal.cumulative_cost_usd))
            events.append((d.close_ts, -d.deal.cumulative_cost_usd))
        events.sort(key=lambda x: x[0])
        conc = 0.0
        peak = 0.0
        for _, delta in events:
            conc += delta
            peak = max(peak, conc)

        ec = m.get("exit_breakdown", {})
        print(f"{bp*100:>8.2f}% | {m['n_deals']:>6d} | {ec.get('end_of_data', 0):>4d} | "
              f"{m.get('cagr_pct', 0):>+7.2f}% | "
              f"${m['final_equity']:>8.0f} | "
              f"{peak/STARTING_CAPITAL*100:>6.0f}% | "
              f"{bp*100*N:>6.2f}%")

        cells.append({
            "base_pct": bp,
            "cagr_pct": m.get("cagr_pct", 0),
            "final_equity": m["final_equity"],
            "n_deals": m["n_deals"],
            "peak_concurrent": peak,
        })

    positive = [c for c in cells if c["cagr_pct"] > 0]
    if positive:
        cliff_cell = max(positive, key=lambda c: c["base_pct"])
        max_cagr_cell = max(cells, key=lambda c: c["cagr_pct"])
        print(f"\n  Highest positive-CAGR: {cliff_cell['base_pct']*100:.2f}% "
              f"(CAGR {cliff_cell['cagr_pct']:+.2f}%)  →  b × N = {cliff_cell['base_pct']*100*N:.2f}%")
        print(f"  Max-CAGR:              {max_cagr_cell['base_pct']*100:.2f}% "
              f"(CAGR {max_cagr_cell['cagr_pct']:+.2f}%)  →  b × N = {max_cagr_cell['base_pct']*100*N:.2f}%")
    else:
        print(f"\n  No positive-CAGR cells at N={N}")

    return {"N": N, "n_active": n_active, "cells": cells}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot-dir", default=None,
                   help="Dir with intraday_{N}m.parquet files. Default: <repo>/data/snapshots/.")
    p.add_argument("--out", default="reports/cliff-at-500k-friction.json",
                   help="Output JSON path (relative to repo root).")
    args = p.parse_args()

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_params = dict(V1_OPTUNA_PARAMS)
    print(f"v1 cliff sweep — friction PINNED at $500K tier (0.16% RT)")
    print(f"Held constant: vol_scale={base_params['so_volume_scale']:.2f}, "
          f"n_SOs={base_params['n_safety_orders']}, "
          f"first_step={base_params['first_so_step_pct']*100:.2f}%, "
          f"step_scale={base_params['so_step_scale']:.2f}")
    print(f"Fee override: fixed_friction_vol_30d = ${FIXED_FRICTION_VOL_30D:,.0f}")
    print(f"Snapshot dir: {snapshot_dir}")

    base_pcts = [0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
                 0.0100, 0.0150, 0.0200, 0.0300, 0.0400, 0.0500]
    universe_sizes = [2, 4, 6, 8, 10, 12, 16, 20]

    all_results = []
    for N in universe_sizes:
        if N > len(MASTER_UNIVERSE):
            print(f"\nSkipping N={N} — only {len(MASTER_UNIVERSE)} symbols available")
            continue
        r = run_sweep_at_N(snapshot_dir, N, base_pcts, base_params)
        all_results.append(r)

    print(f"\n{'='*75}")
    print("CLIFF CURVE AT $500K-TIER FRICTION")
    print(f"{'='*75}")
    print(f"{'N':>3s} | {'measured cliff':>15s} | {'max CAGR':>10s} | "
          f"{'b × N':>7s} | {'predicted (6%/N)':>17s}")
    print("-" * 75)
    for r in all_results:
        positive = [c for c in r["cells"] if c["cagr_pct"] > 0]
        if positive:
            cliff = max(positive, key=lambda c: c["base_pct"])
            max_cagr = max(r["cells"], key=lambda c: c["cagr_pct"])
            print(f"{r['N']:>3d} | {cliff['base_pct']*100:>13.3f}% | "
                  f"{max_cagr['cagr_pct']:>+8.2f}% | "
                  f"{cliff['base_pct']*100*r['N']:>5.2f}% | "
                  f"{6.0/r['N']:>15.3f}%")
        else:
            print(f"{r['N']:>3d} | {'(no positive)':>15s} | — | — | {6.0/r['N']:>15.3f}%")

    out = {
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "vol_scale": base_params["so_volume_scale"],
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "friction_round_trip_pct": 0.16,
        "results": [
            {"N": r["N"], "n_active": r["n_active"], "cells": r["cells"]}
            for r in all_results
        ],
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
