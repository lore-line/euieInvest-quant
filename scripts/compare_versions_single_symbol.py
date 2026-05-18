#!/usr/bin/env python3
"""Single-symbol version-set comparison sweep.

Isolates the version-contribution question from cross-symbol effects:
does adding v0.5/v4/v5 to v1+v2+v3 measurably improve CAGR on a single
symbol's price history?

Approach: load ALL versions' bars + signals once for the chosen symbol,
then run the simulator with each version-set selection independently.
Same base_pct grid, $500K friction. Side-by-side comparison.

Usage:
    uv run scripts/compare_versions_single_symbol.py --symbol BTC-USD

Compares (default): v1+v2+v3 vs v0.5+v1+v2+v3+v4+v5. Add --version-sets
to override.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.backtest import dca_grid  # noqa: E402

START = "2022-09-15"
END = "2026-05-17"
STARTING_CAPITAL = 3000.0
FIXED_FRICTION_VOL_30D = 500_000.0

ALL_VERSIONS = [0.5, 1, 2, 3, 4, 5]

BASE_PARAMS = {
    "n_safety_orders": 9,
    "first_so_step_pct": 0.025747011371995105,
    "so_step_scale": 1.6942997249142477,
    "so_volume_scale": 2.30,
    "strand_ban_days": 122,
    "is_taker": False,
    "early_sl_pct": None,
    "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
}

_ALL_BARS: dict = {}
_ALL_SIGNALS: dict = {}
_ACTIVE_VERSIONS: list = []


def load_all_versions(snapshot_dir: Path, symbol: str, versions: list):
    bars, signals = {}, {}
    bars_1h = dca_grid.load_bars(snapshot_dir, symbol, 60, START, END)
    if bars_1h.empty:
        raise RuntimeError(f"No 60m bars for {symbol}")
    for ver in versions:
        tf_min = dca_grid.get_native_tf_min(ver)
        bars_v = dca_grid.load_bars(snapshot_dir, symbol, tf_min, START, END)
        if bars_v.empty:
            print(f"  [warn] no bars at {tf_min}m for {symbol} v{ver}, skipping")
            continue
        sigs = dca_grid.generate_entry_signals(bars_v, ver, bars_1h)
        bars[(symbol, ver)] = bars_v
        signals[(symbol, ver)] = sigs
    return bars, signals


def _simulate_cell(args: tuple) -> dict:
    bp, base_params, starting_capital = args
    # Build per-version subset of the cached globals
    bars_subset = {k: v for k, v in _ALL_BARS.items() if k[1] in _ACTIVE_VERSIONS}
    signals_subset = {k: v for k, v in _ALL_SIGNALS.items() if k[1] in _ACTIVE_VERSIONS}
    params = {**base_params,
              "base_order_usd": 0,
              "base_pct_of_equity": bp}
    result = dca_grid.simulate_portfolio(
        signals_subset, bars_subset, params, starting_capital)
    m = dca_grid.compute_metrics(result)
    closed = result["closed_deals"]
    by_ver: dict = {}
    for d in closed:
        by_ver[d.deal.version] = by_ver.get(d.deal.version, 0) + 1
    return {
        "base_pct": bp,
        "cagr_pct": m.get("cagr_pct", 0),
        "final_equity": m["final_equity"],
        "n_deals": m["n_deals"],
        "deals_by_version": {f"v{v}": by_ver.get(v, 0) for v in _ACTIVE_VERSIONS},
        "eod": m.get("exit_breakdown", {}).get("end_of_data", 0),
    }


def run_version_set(versions: list, base_pcts: list, pool_size: int) -> list:
    global _ACTIVE_VERSIONS
    _ACTIVE_VERSIONS = list(versions)
    args = [(bp, BASE_PARAMS, STARTING_CAPITAL) for bp in base_pcts]
    ctx = mp.get_context("fork")
    with ctx.Pool(pool_size) as pool:
        cells = pool.map(_simulate_cell, args)
    cells.sort(key=lambda c: c["base_pct"])
    return cells


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--version-sets", nargs="*",
                   default=["1,2,3", "0.5,1,2,3,4,5"],
                   help="Comma-separated version lists to compare (default: "
                        "'1,2,3' vs '0.5,1,2,3,4,5').")
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR

    version_sets = []
    for vs in args.version_sets:
        parsed = [float(v) if "." in v else int(v) for v in vs.split(",")]
        version_sets.append(parsed)
    print(f"Single-symbol version-set comparison")
    print(f"Symbol: {args.symbol}")
    print(f"Version sets: {version_sets}")
    print(f"Workers: {args.workers}")
    print(f"Snapshot dir: {snapshot_dir}")
    print()

    # Load union of all needed versions ONCE
    union_versions = sorted({v for vs in version_sets for v in vs})
    print(f"Loading bars + signals for union of versions: {union_versions}")
    global _ALL_BARS, _ALL_SIGNALS
    _ALL_BARS, _ALL_SIGNALS = load_all_versions(
        snapshot_dir, args.symbol, union_versions)
    print(f"Loaded {len(_ALL_BARS)} (sym, ver) pairs")
    sig_counts = {ver: 0 for ver in union_versions}
    for (_, ver), sigs in _ALL_SIGNALS.items():
        sig_counts[ver] += int(sigs.sum())
    print(f"Signal counts: " + " | ".join(f"v{v}={c}" for v, c in sig_counts.items()))
    print()

    base_pcts = [0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
                 0.0100, 0.0150, 0.0200, 0.0300, 0.0400, 0.0500, 0.0700,
                 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.4000, 0.5000]

    results = []
    for versions in version_sets:
        label = "+".join(f"v{v}" for v in versions)
        print(f"\n{'='*72}")
        print(f"Running {label}")
        print(f"{'='*72}")
        cells = run_version_set(versions, base_pcts, args.workers)
        results.append({"versions": versions, "label": label, "cells": cells})

        print(f"{'b%':>5s} | {'deals':>6s} | "
              f"{'CAGR':>8s} | {'final $':>9s}")
        print("-" * 50)
        for c in cells:
            print(f"{c['base_pct']*100:>4.2f}% | {c['n_deals']:>6d} | "
                  f"{c['cagr_pct']:>+7.2f}% | "
                  f"${c['final_equity']:>8.0f}")
        mx = max(cells, key=lambda c: c["cagr_pct"])
        print(f"  Peak: {mx['cagr_pct']:+.2f}% @ b={mx['base_pct']*100:.2f}% "
              f"(deals: {mx['n_deals']}, by-version: {mx['deals_by_version']})")

    # Side-by-side delta
    print(f"\n{'='*72}")
    print(f"SIDE-BY-SIDE: {args.symbol} — peak CAGR per version set")
    print(f"{'='*72}")
    print(f"{'b%':>5s} | " + " | ".join(f"{r['label']:>20s}" for r in results) + " | Δ")
    print("-" * (8 + 23 * len(results) + 8))
    for i, bp in enumerate(base_pcts):
        row = [f"{bp*100:>4.2f}%"]
        cagrs = []
        for r in results:
            c = r["cells"][i]
            row.append(f"{c['cagr_pct']:>+18.2f}%")
            cagrs.append(c["cagr_pct"])
        delta = cagrs[-1] - cagrs[0] if len(cagrs) >= 2 else 0
        row.append(f"{'+' if delta>=0 else ''}{delta:>5.2f}pp")
        print(" | ".join(row))

    out_path = ROOT / (args.out or f"reports/compare-versions-{args.symbol.replace('-', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "symbol": args.symbol,
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "signal_counts": {f"v{v}": c for v, c in sig_counts.items()},
        "results": results,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
