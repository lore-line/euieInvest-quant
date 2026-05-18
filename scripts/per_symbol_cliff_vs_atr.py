#!/usr/bin/env python3
"""Per-symbol cliff position vs ATR%.

Isolates the symbol-intrinsic cliff effect from the universe-level
correlation confound that hit `analyze_cliff_vs_atr.py`.

For each symbol in the 20-symbol universe:
  1. Run v1-only simulator at the symbol alone (N=1 per simulation)
  2. Sweep base_pct over a wide grid {0.05%, ..., 50%}
  3. Find the symbol-specific b_cliff (highest b with positive CAGR)
  4. Pair with that symbol's mean ATR(14)

Then regress: log(b_cliff) = α + β · log(ATR)

Hypothesis: β ≈ -1 (b_cliff × ATR = constant). Higher ATR → deeper
drawdowns → SO ladder eats more cash → lower b_cliff.

Output: reports/per-symbol-cliff-vs-atr.json + ASCII plot in the log.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.backtest import dca_grid  # noqa: E402

START = "2022-09-15"
END = "2026-05-17"
STARTING_CAPITAL = 3000.0
FIXED_FRICTION_VOL_30D = 500_000.0

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

BASE_PCTS = [0.0005, 0.0010, 0.0020, 0.0030, 0.0050, 0.0070,
             0.0100, 0.0150, 0.0200, 0.0300, 0.0500, 0.0700,
             0.1000, 0.1500, 0.2000, 0.3000, 0.5000]

_BARS: dict = {}
_SIGNALS: dict = {}


def mean_atr_pct(bars: pd.DataFrame, period: int = 14) -> float:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return float((atr / c).dropna().mean() * 100)


def _simulate_cell(args: tuple) -> dict:
    symbol, bp = args
    params = {**BASE_PARAMS,
              "base_order_usd": 0,
              "base_pct_of_equity": bp}
    bars_dict = {(symbol, 1): _BARS[symbol]}
    signals_dict = {(symbol, 1): _SIGNALS[symbol]}
    result = dca_grid.simulate_portfolio(
        signals_dict, bars_dict, params, STARTING_CAPITAL)
    m = dca_grid.compute_metrics(result)
    return {
        "symbol": symbol,
        "base_pct": bp,
        "cagr_pct": m.get("cagr_pct", 0),
        "final_equity": m["final_equity"],
        "n_deals": m["n_deals"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--out", default="reports/per-symbol-cliff-vs-atr.json")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR

    print(f"Per-symbol cliff vs ATR analysis")
    print(f"Universe: {len(MASTER_UNIVERSE)} symbols × {len(BASE_PCTS)} base_pcts = "
          f"{len(MASTER_UNIVERSE) * len(BASE_PCTS)} cells")
    print(f"Workers: {args.workers}")

    print(f"\nLoading bars + signals + computing ATR per symbol...")
    global _BARS, _SIGNALS
    atr_per_sym: dict[str, float] = {}
    for sym in MASTER_UNIVERSE:
        bars_1h = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
        bars_5m = dca_grid.load_bars(snapshot_dir, sym, 5, START, END)
        if bars_1h.empty or bars_5m.empty:
            print(f"  [warn] missing bars for {sym}, skipping")
            continue
        atr_per_sym[sym] = mean_atr_pct(bars_1h)
        sigs = dca_grid.generate_entry_signals(bars_5m, 1, bars_1h)
        _BARS[sym] = bars_5m
        _SIGNALS[sym] = sigs
        print(f"  {sym}: ATR={atr_per_sym[sym]:.2f}%, signals={int(sigs.sum())}")

    print(f"\nDispatching {len(_BARS) * len(BASE_PCTS)} cells to pool...")
    cell_args = [(sym, bp) for sym in _BARS for bp in BASE_PCTS]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        cells = pool.map(_simulate_cell, cell_args)

    # Group cells by symbol, identify cliff b per symbol
    per_sym: dict[str, list] = {sym: [] for sym in _BARS}
    for c in cells:
        per_sym[c["symbol"]].append(c)
    for sym in per_sym:
        per_sym[sym].sort(key=lambda c: c["base_pct"])

    print(f"\n{'='*90}")
    print(f"PER-SYMBOL CLIFF POSITIONS")
    print(f"{'='*90}")
    print(f"{'symbol':<10s} | {'ATR%':>6s} | {'cliff b%':>9s} | {'max-CAGR b%':>12s} | "
          f"{'max CAGR':>9s} | {'b × ATR':>8s}")
    print("-" * 90)

    rows = []
    for sym in MASTER_UNIVERSE:
        if sym not in per_sym:
            continue
        cells_s = per_sym[sym]
        positive = [c for c in cells_s if c["cagr_pct"] > 0]
        if not positive:
            print(f"{sym:<10s} | {atr_per_sym[sym]:>5.2f}% | {'NO POS':>9s} | {'—':>12s} | "
                  f"{'—':>9s} | {'—':>8s}")
            continue
        cliff = max(positive, key=lambda c: c["base_pct"])
        mx = max(cells_s, key=lambda c: c["cagr_pct"])
        b_x_atr = cliff["base_pct"] * 100 * atr_per_sym[sym]
        rows.append({
            "symbol": sym,
            "atr_pct": atr_per_sym[sym],
            "cliff_b_pct": cliff["base_pct"] * 100,
            "max_cagr_b_pct": mx["base_pct"] * 100,
            "max_cagr_pct": mx["cagr_pct"],
            "b_x_atr": b_x_atr,
        })
        print(f"{sym:<10s} | {atr_per_sym[sym]:>5.2f}% | {cliff['base_pct']*100:>8.2f}% | "
              f"{mx['base_pct']*100:>11.2f}% | {mx['cagr_pct']:>+8.2f}% | "
              f"{b_x_atr:>7.3f}")

    if len(rows) >= 5:
        atr_arr = np.log(np.array([r["atr_pct"] for r in rows]))
        cliff_arr = np.log(np.array([r["cliff_b_pct"] for r in rows]))
        max_cagr_b_arr = np.log(np.array([r["max_cagr_b_pct"] for r in rows]))
        X = np.column_stack([np.ones(len(rows)), atr_arr])
        coef_cliff, *_ = np.linalg.lstsq(X, cliff_arr, rcond=None)
        coef_max, *_ = np.linalg.lstsq(X, max_cagr_b_arr, rcond=None)
        print(f"\n{'='*60}")
        print("REGRESSIONS")
        print(f"{'='*60}")
        print(f"cliff_b ≈ {np.exp(coef_cliff[0]):.3f} × ATR^({coef_cliff[1]:.3f})")
        print(f"max_cagr_b ≈ {np.exp(coef_max[0]):.3f} × ATR^({coef_max[1]:.3f})")
        bxa = np.array([r["b_x_atr"] for r in rows])
        bxa_mc = np.array([r["max_cagr_b_pct"] * r["atr_pct"] for r in rows])
        print(f"\nb_cliff × ATR values: mean={bxa.mean():.2f}, std={bxa.std():.2f}, "
              f"CoV={bxa.std()/bxa.mean()*100:.1f}%")
        print(f"b_max_cagr × ATR values: mean={bxa_mc.mean():.2f}, std={bxa_mc.std():.2f}, "
              f"CoV={bxa_mc.std()/bxa_mc.mean()*100:.1f}%")

        # Compare to invariants
        bxa_a2 = np.array([r["cliff_b_pct"] * r["atr_pct"]**2 for r in rows])
        bxa_a05 = np.array([r["cliff_b_pct"] * r["atr_pct"]**0.5 for r in rows])
        print(f"\nAlternative exponents on ATR (tighter = better):")
        for exp_val, label in [(0.5, "ATR^0.5"), (1.0, "ATR^1.0"),
                                (1.5, "ATR^1.5"), (2.0, "ATR^2.0")]:
            vals = np.array([r["cliff_b_pct"] * r["atr_pct"]**exp_val for r in rows])
            print(f"  b_cliff × {label}: mean={vals.mean():.2f}, "
                  f"CoV={vals.std()/vals.mean()*100:.1f}%")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "window": [START, END],
        "starting_capital": STARTING_CAPITAL,
        "fixed_friction_vol_30d": FIXED_FRICTION_VOL_30D,
        "base_pcts": BASE_PCTS,
        "per_symbol": {sym: {
            "atr_pct": atr_per_sym.get(sym),
            "cells": per_sym.get(sym, []),
        } for sym in MASTER_UNIVERSE},
        "summary": rows,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
