#!/usr/bin/env python3
"""Multi-version concurrent sweep — v1 + v2 + v3 firing on N=10 at $500K friction.

Tests the regime-orthogonal harvest hypothesis from doctrine §2.7 / the
queue Job 2 in https://github.com/lore-line/euieInvest-quant/issues/24:

  Does running v1 (1% TP, 5m TF, ATR≥1%) + v2 (2% TP, 15m TF, ATR≥2%) +
  v3 (3% TP, 1h TF, ATR 4-8%) CONCURRENTLY on the same symbols harvest
  more CAGR than v1 alone?

The simulator already supports concurrent versions per symbol — open_deals
is keyed by (symbol, version) so each version holds an independent deal.
Cash is shared globally. Per-version base_pct applies independently, so
total per-symbol deployment = base_pct × n_active_versions when all
versions' signals fire simultaneously (rare; ATR gates are non-overlapping
in most regimes).

Comparison reference (v1-only at N=10, $500K friction, doctrine §2.7):
  base_pct=0.70% → max CAGR +34.57%

Sweep grid is the same as v1-only so we can compare cell-by-cell. The
per-version base_pct is what gets swept; total per-symbol deployment
varies with how many versions fire simultaneously in each regime.

Output: reports/multi-version-n10-500k.json + comparison vs v1 baseline.
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
N = 10  # sweet-spot from baseline cliff sweep
VERSIONS = [1, 2, 3]  # v1=1% TP, v2=2% TP, v3=3% TP

# Same 10-symbol universe as the cliff sweep's N=10 row (validated set).
UNIVERSE_N10 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "ATOM-USD", "RUNE-USD", "FET-USD",
]

# v1 Optuna best params, vol_scale overridden to 2.30 per N=10 cliff sweet spot.
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
    """Load bars + signals for every (symbol, version) combination."""
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
    bp, base_params, starting_capital = args
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
    # Per-version deal breakdown
    by_ver = {1: 0, 2: 0, 3: 0}
    for d in closed:
        by_ver[d.deal.version] = by_ver.get(d.deal.version, 0) + 1
    return {
        "base_pct": bp,
        "cagr_pct": m.get("cagr_pct", 0),
        "final_equity": m["final_equity"],
        "n_deals": m["n_deals"],
        "deals_v1": by_ver[1],
        "deals_v2": by_ver[2],
        "deals_v3": by_ver[3],
        "peak_concurrent": peak,
        "eod": m.get("exit_breakdown", {}).get("end_of_data", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--out", default="reports/multi-version-n10-500k.json")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global _BARS_PER_SV, _SIGNALS_PER_SV
    print(f"Multi-version sweep — v1+v2+v3 concurrent on N={N} at $500K friction")
    print(f"Universe: {UNIVERSE_N10}")
    print(f"Versions: {VERSIONS} (1% / 2% / 3% TP)")
    print(f"Workers: {args.workers}")
    print(f"Snapshot dir: {snapshot_dir}")
    print()
    print("Loading bars + signals for all (sym, ver) combinations...")
    _BARS_PER_SV, _SIGNALS_PER_SV = load_all_versions(
        snapshot_dir, UNIVERSE_N10, VERSIONS)
    n_active = len(_BARS_PER_SV)
    print(f"Loaded {n_active} (sym, ver) pairs "
          f"({len(UNIVERSE_N10)} symbols × {len(VERSIONS)} versions)")

    # Signal counts by version for sanity check
    sig_counts = {ver: 0 for ver in VERSIONS}
    for (sym, ver), sigs in _SIGNALS_PER_SV.items():
        sig_counts[ver] += int(sigs.sum())
    print(f"Total entry signals: " +
          " | ".join(f"v{v}={c}" for v, c in sig_counts.items()))
    print()

    # Same base_pct grid as cliff sweep for direct cell comparison
    base_pcts = [0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
                 0.0100, 0.0150, 0.0200, 0.0300, 0.0400, 0.0500]

    args_list = [(bp, BASE_PARAMS, STARTING_CAPITAL) for bp in base_pcts]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        cells = pool.map(_simulate_cell, args_list)
    cells.sort(key=lambda c: c["base_pct"])

    print(f"{'b%':>5s} | {'deals':>5s} | {'v1':>5s} | {'v2':>5s} | {'v3':>5s} | "
          f"{'eod':>4s} | {'CAGR':>8s} | {'final $':>9s} | {'peak %':>7s}")
    print("-" * 90)
    for c in cells:
        print(f"{c['base_pct']*100:>4.2f}% | {c['n_deals']:>5d} | "
              f"{c['deals_v1']:>5d} | {c['deals_v2']:>5d} | {c['deals_v3']:>5d} | "
              f"{c['eod']:>4d} | "
              f"{c['cagr_pct']:>+7.2f}% | "
              f"${c['final_equity']:>8.0f} | "
              f"{c['peak_concurrent']/STARTING_CAPITAL*100:>6.0f}%")

    positive = [c for c in cells if c["cagr_pct"] > 0]
    if positive:
        max_cagr = max(cells, key=lambda c: c["cagr_pct"])
        print(f"\n  Max CAGR: {max_cagr['cagr_pct']:+.2f}% @ b={max_cagr['base_pct']*100:.2f}%")
        # Compare to v1-only N=10 baseline at $500K friction (from cliff sweep): +34.57% @ b=0.70%
        delta = max_cagr['cagr_pct'] - 34.57
        print(f"  vs v1-only baseline (+34.57%): "
              f"{'+' if delta>=0 else ''}{delta:.2f}pp ({'+' if delta>=0 else ''}{delta/34.57*100:.1f}% relative)")

    out = {
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "universe": UNIVERSE_N10,
        "versions": VERSIONS,
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "signal_counts_by_version": sig_counts,
        "cells": cells,
        "v1_only_baseline_max_cagr": 34.57,
        "v1_only_baseline_base_pct": 0.0070,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
