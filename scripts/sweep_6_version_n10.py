#!/usr/bin/env python3
"""6-version concurrent sweep — v0.5 + v1 + v2 + v3 + v4 + v5 on N=10 at $500K friction.

Phase 2b extension of the multi-version concurrent finding (Phase 2a:
v1+v2+v3 → +41.57% CAGR at b=0.50%, doctrine §2.7). Tests whether
extending the TP-tier ladder to v0.5 (low-vol catcher) and v4/v5
(extreme-vol catchers) adds further CAGR via wider regime coverage.

Version configs (per VERSION_CONFIG in src/quant/backtest/dca_grid.py):
  v0.5: 0.5% TP, 1m TF, ATR ∈ [0.5%, 1%)
  v1:   1% TP,   5m TF, ATR ≥ 1%
  v2:   2% TP,   15m TF, ATR ≥ 2%
  v3:   3% TP,   60m TF, ATR ∈ [4%, 8%]
  v4:   4% TP,   4h TF,  ATR ≥ 8% (resampled from 60m)
  v5:   5% TP,   1d TF,  ATR ≥ 10% (resampled from 60m)

Comparison reference (Phase 2a, v1+v2+v3 only):
  Peak +41.57% CAGR at b=0.50%

Comparison reference (Job 1, v1-only):
  Peak +34.57% CAGR at b=0.70%

If v0.5/v4/v5 add edge: predicts further compounding of regime-orthogonal
harvest hypothesis. If they don't: TP-tier extension hits diminishing
returns at the v1/v2/v3 boundary.
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
N = 10
VERSIONS = [0.5, 1, 2, 3, 4, 5]

UNIVERSE_N10 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "ATOM-USD", "RUNE-USD", "FET-USD",
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


def load_all_versions(snapshot_dir: Path, symbols: list, versions: list):
    bars, signals = {}, {}
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
    by_ver = {v: 0 for v in VERSIONS}
    for d in closed:
        by_ver[d.deal.version] = by_ver.get(d.deal.version, 0) + 1
    return {
        "base_pct": bp,
        "cagr_pct": m.get("cagr_pct", 0),
        "final_equity": m["final_equity"],
        "n_deals": m["n_deals"],
        **{f"deals_v{v}": by_ver.get(v, 0) for v in VERSIONS},
        "peak_concurrent": peak,
        "eod": m.get("exit_breakdown", {}).get("end_of_data", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--out", default="reports/6-version-n10-500k.json")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global _BARS_PER_SV, _SIGNALS_PER_SV
    print(f"6-version sweep — v{VERSIONS} concurrent on N={N} at $500K friction")
    print(f"Universe: {UNIVERSE_N10}")
    print(f"Workers: {args.workers}")
    print(f"Snapshot dir: {snapshot_dir}")
    print()
    print("Loading bars + signals for all (sym, ver) combinations...")
    _BARS_PER_SV, _SIGNALS_PER_SV = load_all_versions(
        snapshot_dir, UNIVERSE_N10, VERSIONS)
    n_active = len(_BARS_PER_SV)
    print(f"Loaded {n_active} (sym, ver) pairs "
          f"({len(UNIVERSE_N10)} symbols × {len(VERSIONS)} versions)")

    sig_counts = {ver: 0 for ver in VERSIONS}
    for (sym, ver), sigs in _SIGNALS_PER_SV.items():
        sig_counts[ver] += int(sigs.sum())
    print(f"Total entry signals: " +
          " | ".join(f"v{v}={c}" for v, c in sig_counts.items()))
    print()

    base_pcts = [0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
                 0.0100, 0.0150, 0.0200, 0.0300, 0.0400, 0.0500]
    args_list = [(bp, BASE_PARAMS, STARTING_CAPITAL) for bp in base_pcts]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        cells = pool.map(_simulate_cell, args_list)
    cells.sort(key=lambda c: c["base_pct"])

    headers = ["b%", "deals"] + [f"v{v}" for v in VERSIONS] + ["eod", "CAGR", "final $", "peak %"]
    print(" | ".join(f"{h:>6s}" if i > 0 else f"{h:>5s}" for i, h in enumerate(headers)))
    print("-" * 110)
    for c in cells:
        row = [f"{c['base_pct']*100:>4.2f}%", f"{c['n_deals']:>6d}"]
        for v in VERSIONS:
            row.append(f"{c[f'deals_v{v}']:>6d}")
        row.append(f"{c['eod']:>6d}")
        row.append(f"{c['cagr_pct']:>+5.2f}%")
        row.append(f"${c['final_equity']:>6.0f}")
        row.append(f"{c['peak_concurrent']/STARTING_CAPITAL*100:>5.0f}%")
        print(" | ".join(row))

    positive = [c for c in cells if c["cagr_pct"] > 0]
    if positive:
        mx = max(cells, key=lambda c: c["cagr_pct"])
        print(f"\n  Peak CAGR: {mx['cagr_pct']:+.2f}% @ b={mx['base_pct']*100:.2f}%")
        print(f"  vs v1-only N=10/$500K baseline (+34.57%): "
              f"{'+' if mx['cagr_pct']-34.57>=0 else ''}{mx['cagr_pct']-34.57:.2f}pp")
        print(f"  vs v1+v2+v3 N=10/$500K (+41.57%): "
              f"{'+' if mx['cagr_pct']-41.57>=0 else ''}{mx['cagr_pct']-41.57:.2f}pp")

    out = {
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "universe": UNIVERSE_N10,
        "versions": VERSIONS,
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "signal_counts_by_version": {f"v{v}": c for v, c in sig_counts.items()},
        "cells": cells,
        "v1_only_baseline_max_cagr": 34.57,
        "v1_v2_v3_baseline_max_cagr": 41.57,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
