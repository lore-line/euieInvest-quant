#!/usr/bin/env python3
"""Multi-version × N sweep — v1+v2+v3 concurrent across N ∈ {2,4,6,8,10,12,16,20}.

Companion to scripts/sweep_multi_version_n10.py, but iterates over the
full N range to find the new sweet spot under multi-version dynamics.

Phase 2a (N=10 only) showed v1+v2+v3 concurrent produced +41.57% CAGR
at b=0.50% per-version vs v1-only +34.57% at b=0.70%/N=10. The cliff
sharpened nearly 2× and shifted left.

Hypothesis for this sweep: the multi-version sweet spot may shift from
N=10 to a smaller N (~6-8?) because each (sym, ver) entry consumes
deployment budget — wider universes hit the cliff faster under
multi-version. Alternatively it may stay at N=10-12.

Output: reports/multi-version-by-N-500k.json + side-by-side comparison
vs the v1-only N table from cliff-at-500k-friction.json.
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
VERSIONS = [1, 2, 3]

MASTER_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
    "INJ-USD", "GRT-USD", "AAVE-USD", "UNI-USD",
    "NEAR-USD", "SUSHI-USD", "APT-USD", "TIA-USD",
]

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

_BARS_PER_SV: dict = {}
_SIGNALS_PER_SV: dict = {}


def load_all_versions(snapshot_dir: Path, symbols: list[str], versions: list[int]):
    bars: dict = {}
    signals: dict = {}
    for ver in versions:
        tf_min = dca_grid.get_native_tf_min(ver)
        for sym in symbols:
            bars_1h = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
            bars_v = dca_grid.load_bars(snapshot_dir, sym, tf_min, START, END)
            if bars_v.empty or bars_1h.empty:
                continue
            sigs = dca_grid.generate_entry_signals(bars_v, ver, bars_1h)
            bars[(sym, ver)] = bars_v
            signals[(sym, ver)] = sigs
    return bars, signals


def _simulate_cell(args: tuple) -> dict:
    N, bp, base_params, starting_capital = args
    params = {**base_params,
              "base_order_usd": 0,
              "base_pct_of_equity": bp}
    result = dca_grid.simulate_portfolio(
        _SIGNALS_PER_SV, _BARS_PER_SV, params, starting_capital)
    m = dca_grid.compute_metrics(result)
    closed = result["closed_deals"]
    events = []
    for d in closed:
        events.append((d.deal.opened_at, d.deal.cumulative_cost_usd))
        events.append((d.close_ts, -d.deal.cumulative_cost_usd))
    events.sort(key=lambda x: x[0])
    conc, peak = 0.0, 0.0
    for _, delta in events:
        conc += delta
        peak = max(peak, conc)
    by_ver = {v: 0 for v in VERSIONS}
    for d in closed:
        by_ver[d.deal.version] = by_ver.get(d.deal.version, 0) + 1
    return {
        "N": N,
        "base_pct": bp,
        "cagr_pct": m.get("cagr_pct", 0),
        "final_equity": m["final_equity"],
        "n_deals": m["n_deals"],
        **{f"deals_v{v}": by_ver.get(v, 0) for v in VERSIONS},
        "peak_concurrent": peak,
        "eod": m.get("exit_breakdown", {}).get("end_of_data", 0),
    }


def run_sweep_at_N(snapshot_dir: Path, N: int, base_pcts: list[float],
                   pool_size: int) -> dict:
    global _BARS_PER_SV, _SIGNALS_PER_SV
    symbols = MASTER_UNIVERSE[:N]
    print(f"\n{'='*72}")
    print(f"N={N}: {symbols}")
    print(f"{'='*72}")

    _BARS_PER_SV, _SIGNALS_PER_SV = load_all_versions(
        snapshot_dir, symbols, VERSIONS)
    n_active = len(_BARS_PER_SV)
    sig_counts = {ver: 0 for ver in VERSIONS}
    for (_, ver), sigs in _SIGNALS_PER_SV.items():
        sig_counts[ver] += int(sigs.sum())
    print(f"Loaded {n_active} (sym, ver) pairs. "
          f"Signals: " + " | ".join(f"v{v}={c}" for v, c in sig_counts.items()))

    args_list = [(N, bp, BASE_PARAMS, STARTING_CAPITAL) for bp in base_pcts]
    ctx = mp.get_context("fork")
    with ctx.Pool(pool_size) as pool:
        cells = pool.map(_simulate_cell, args_list)
    cells.sort(key=lambda c: c["base_pct"])

    print(f"\n{'b%':>5s} | {'deals':>6s} | {'v1':>5s} | {'v2':>5s} | {'v3':>5s} | "
          f"{'CAGR':>8s} | {'final $':>9s} | {'peak %':>7s}")
    print("-" * 80)
    for c in cells:
        print(f"{c['base_pct']*100:>4.2f}% | {c['n_deals']:>6d} | "
              f"{c['deals_v1']:>5d} | {c['deals_v2']:>5d} | {c['deals_v3']:>5d} | "
              f"{c['cagr_pct']:>+7.2f}% | "
              f"${c['final_equity']:>8.0f} | "
              f"{c['peak_concurrent']/STARTING_CAPITAL*100:>6.0f}%")

    positive = [c for c in cells if c["cagr_pct"] > 0]
    if positive:
        cliff_cell = max(positive, key=lambda c: c["base_pct"])
        max_cagr_cell = max(cells, key=lambda c: c["cagr_pct"])
        print(f"\n  Cliff (highest pos-CAGR): {cliff_cell['base_pct']*100:.2f}% "
              f"(CAGR {cliff_cell['cagr_pct']:+.2f}%)")
        print(f"  Peak CAGR: {max_cagr_cell['cagr_pct']:+.2f}% @ b={max_cagr_cell['base_pct']*100:.2f}%")
    return {"N": N, "n_active": n_active, "signal_counts": sig_counts,
            "cells": cells}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--out", default="reports/multi-version-by-N-500k.json")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Multi-version × N sweep — v1+v2+v3 concurrent at $500K friction")
    print(f"Versions: {VERSIONS}")
    print(f"Workers: {args.workers}")
    print(f"Snapshot dir: {snapshot_dir}")

    base_pcts = [0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
                 0.0100, 0.0150, 0.0200, 0.0300, 0.0400, 0.0500]
    universe_sizes = [2, 4, 6, 8, 10, 12, 16, 20]

    all_results = []
    for N in universe_sizes:
        if N > len(MASTER_UNIVERSE):
            print(f"\nSkipping N={N} — only {len(MASTER_UNIVERSE)} symbols available")
            continue
        r = run_sweep_at_N(snapshot_dir, N, base_pcts, args.workers)
        all_results.append(r)

    print(f"\n{'='*75}")
    print("MULTI-VERSION CLIFF CURVE AT $500K-TIER FRICTION")
    print(f"{'='*75}")
    print(f"{'N':>3s} | {'peak CAGR':>10s} | {'@ b%':>6s} | {'cliff b%':>8s} | "
          f"{'v1 baseline':>12s} | {'Δ pp':>6s}")
    print("-" * 75)
    # v1-only baselines from cliff-at-500k-friction.json (doctrine §2.7)
    v1_baseline = {2: 7.06, 4: 18.28, 6: 18.24, 8: 22.87, 10: 34.57,
                   12: 34.96, 16: 27.71, 20: 27.52}
    for r in all_results:
        positive = [c for c in r["cells"] if c["cagr_pct"] > 0]
        if positive:
            cliff = max(positive, key=lambda c: c["base_pct"])
            mx = max(r["cells"], key=lambda c: c["cagr_pct"])
            base = v1_baseline.get(r["N"], 0)
            delta = mx["cagr_pct"] - base
            print(f"{r['N']:>3d} | {mx['cagr_pct']:>+8.2f}% | "
                  f"{mx['base_pct']*100:>5.2f}% | "
                  f"{cliff['base_pct']*100:>7.3f}% | "
                  f"{base:>+10.2f}% | "
                  f"{'+' if delta>=0 else ''}{delta:>5.2f}")

    out = {
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "versions": VERSIONS,
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "v1_only_baseline_peak_cagr_by_N": v1_baseline,
        "results": [
            {"N": r["N"], "n_active": r["n_active"],
             "signal_counts": r["signal_counts"], "cells": r["cells"]}
            for r in all_results
        ],
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
