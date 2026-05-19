#!/usr/bin/env python3
"""Regime-gated multi-version sweep — v1+v2+v3 × N=12 with P1 regime labels.

Tests the highest-leverage Stream 2 hypothesis (doctrine §9.5):
  "go aggressive in healthy regimes, hold cash in declining regimes"

Uses consumer-team P1 v0.4 regime labels (data/quant_publish/regime_labels_v1.parquet,
4 classes: bear_trend / choppy_recovery / sideways_range / steady_bull).

Recommended gating per consumer-team comment on issue #20:
  bear_trend       = 0.0   (pause new entries — strategy bleeds in trended-down regime)
  choppy_recovery  = 1.0   (chop is the strategy's home regime)
  sideways_range   = 1.0   (also harvestable)
  steady_bull      = 1.0   (or 1.5 for aggressive variant)
  unknown          = 1.0   (default for unlabeled days)

Compares against un-gated baseline:
  multi-version v1+v2+v3 × N=12 / $500K friction / b=0.50% → +45.19% CAGR
  (from doctrine §2.7, reports/multi-version-by-N-500k.json)

Two gating profiles tested:
  - Conservative: bear=0, others=1 (pause in bear only)
  - Aggressive: bear=0, chop=1, sideways=1, steady=1.5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.backtest import dca_grid  # noqa: E402

START = "2022-09-15"
END = "2026-05-17"
STARTING_CAPITAL = 3000.0
FIXED_FRICTION_VOL_30D = 500_000.0
N = 12
VERSIONS = [1, 2, 3]

UNIVERSE_N12 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "ATOM-USD", "RUNE-USD", "FET-USD",
    "DOGE-USD", "XRP-USD",
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

GATING_PROFILES = {
    "ungated": None,  # baseline — sim's regime gate disabled
    "conservative": {
        "bear_trend": 0.0,
        "choppy_recovery": 1.0,
        "sideways_range": 1.0,
        "steady_bull": 1.0,
        "unknown": 1.0,
    },
    "aggressive": {
        "bear_trend": 0.0,
        "choppy_recovery": 1.0,
        "sideways_range": 1.0,
        "steady_bull": 1.5,
        "unknown": 1.0,
    },
    "ultra_aggressive": {
        "bear_trend": 0.0,
        "choppy_recovery": 1.5,
        "sideways_range": 1.0,
        "steady_bull": 2.0,
        "unknown": 1.0,
    },
}

_BARS_PER_SV: dict = {}
_SIGNALS_PER_SV: dict = {}
_REGIME_LOOKUP: dict = {}


def load_regime_labels(parquet_path: Path) -> dict:
    """date (normalized midnight) → regime_label."""
    df = pl.read_parquet(parquet_path).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return {row["date"].normalize(): row["regime_label"]
            for _, row in df.iterrows()}


def load_all_versions(snapshot_dir: Path, symbols, versions):
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


def _simulate_cell(args):
    bp, profile_name, regime_mults, base_params, starting_capital = args
    params = {**base_params,
              "base_order_usd": 0,
              "base_pct_of_equity": bp}
    if regime_mults is not None:
        params["regime_multipliers"] = regime_mults
        params["regime_lookup"] = _REGIME_LOOKUP
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
        "profile": profile_name,
        "base_pct": bp,
        "cagr_pct": m.get("cagr_pct", 0),
        "final_equity": m["final_equity"],
        "n_deals": m["n_deals"],
        **{f"deals_v{v}": by_ver.get(v, 0) for v in VERSIONS},
        "peak_concurrent": peak,
        "eod": m.get("exit_breakdown", {}).get("end_of_data", 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--regime-parquet",
                   default="data/quant_publish/regime_labels_v1.parquet")
    p.add_argument("--out", default="reports/regime-gated-n12.json")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 16))
    args = p.parse_args()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global _BARS_PER_SV, _SIGNALS_PER_SV, _REGIME_LOOKUP
    print(f"Regime-gated multi-version sweep — N={N}, $500K friction, v1+v2+v3")
    print(f"Workers: {args.workers} (reduced to share CPU with other sweep)")

    print("Loading regime labels...")
    regime_path = ROOT / args.regime_parquet
    _REGIME_LOOKUP = load_regime_labels(regime_path)
    print(f"  {len(_REGIME_LOOKUP)} day-labels loaded "
          f"({min(_REGIME_LOOKUP.keys()).date()} → {max(_REGIME_LOOKUP.keys()).date()})")
    label_counts = {}
    for lbl in _REGIME_LOOKUP.values():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"  Distribution: {label_counts}")

    print("\nLoading bars + signals for all (sym, ver) combinations...")
    _BARS_PER_SV, _SIGNALS_PER_SV = load_all_versions(
        snapshot_dir, UNIVERSE_N12, VERSIONS)
    print(f"  Loaded {len(_BARS_PER_SV)} (sym, ver) pairs")

    base_pcts = [0.0010, 0.0030, 0.0050, 0.0070, 0.0100, 0.0150,
                 0.0200, 0.0300, 0.0500]

    all_cells = []
    for profile_name, regime_mults in GATING_PROFILES.items():
        print(f"\n{'='*72}")
        print(f"Profile: {profile_name}")
        if regime_mults:
            print(f"  Multipliers: {regime_mults}")
        print(f"{'='*72}")
        args_list = [(bp, profile_name, regime_mults, BASE_PARAMS, STARTING_CAPITAL)
                     for bp in base_pcts]
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            cells = pool.map(_simulate_cell, args_list)
        cells.sort(key=lambda c: c["base_pct"])
        print(f"{'b%':>5s} | {'deals':>5s} | {'CAGR':>8s} | {'final $':>9s}")
        print("-" * 45)
        for c in cells:
            print(f"{c['base_pct']*100:>4.2f}% | {c['n_deals']:>5d} | "
                  f"{c['cagr_pct']:>+7.2f}% | ${c['final_equity']:>8.0f}")
        mx = max(cells, key=lambda c: c["cagr_pct"])
        print(f"\n  Peak: {mx['cagr_pct']:+.2f}% @ b={mx['base_pct']*100:.2f}%")
        all_cells.extend(cells)

    # Comparison table
    print(f"\n{'='*72}")
    print(f"COMPARISON: Peak CAGR by profile (vs un-gated baseline)")
    print(f"{'='*72}")
    print(f"{'profile':<20s} | {'peak CAGR':>10s} | {'@ b%':>6s} | {'vs ungated':>11s}")
    print("-" * 60)
    ungated_peak = max(
        (c for c in all_cells if c["profile"] == "ungated"),
        key=lambda c: c["cagr_pct"]
    )
    for profile_name in GATING_PROFILES.keys():
        mx = max((c for c in all_cells if c["profile"] == profile_name),
                 key=lambda c: c["cagr_pct"])
        delta = mx["cagr_pct"] - ungated_peak["cagr_pct"]
        sign = "+" if delta >= 0 else ""
        print(f"{profile_name:<20s} | {mx['cagr_pct']:>+9.2f}% | "
              f"{mx['base_pct']*100:>5.2f}% | {sign}{delta:>+9.2f}pp")

    out = {
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "universe": UNIVERSE_N12,
        "versions": VERSIONS,
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "regime_label_counts": label_counts,
        "gating_profiles": {k: v for k, v in GATING_PROFILES.items()},
        "cells": all_cells,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
