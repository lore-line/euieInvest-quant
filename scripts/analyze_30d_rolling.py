#!/usr/bin/env python3
"""Rolling-30d P&L distribution for the best multi-version config.

Re-runs the simulator once on N=12 v1+v2+v3 at b=0.50% (the doctrine §2.7
sweet spot at $500K friction) and walks `closed_deals` to compute:
  - per-day net P&L (sum of all deal closes that day)
  - 30-day rolling sum
  - peak / trough / median / quartiles of the rolling 30d series

Quick risk-shape sanity check: does the +45.19% CAGR come from a few
concentrated months, or a smooth grind? Big 30d peak = concentration
risk = sizing should respect drawdown of an equivalent magnitude.
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
STARTING_CAPITAL = 3000.0
N = 12
VERSIONS = [1, 2, 3]
BASE_PCT = 0.0050
UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
]

BASE_PARAMS = {
    "n_safety_orders": 9,
    "first_so_step_pct": 0.025747011371995105,
    "so_step_scale": 1.6942997249142477,
    "so_volume_scale": 2.30,
    "strand_ban_days": 122,
    "is_taker": False,
    "early_sl_pct": None,
    "fixed_friction_vol_30d": 500_000.0,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", default=None)
    args = p.parse_args()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else dca_grid.SNAPSHOT_DIR

    print(f"Config: N={N}, versions={VERSIONS}, b={BASE_PCT*100:.2f}%, "
          f"friction=$500K, window={START}→{END}")
    print(f"Loading bars + signals...")
    bars, signals = {}, {}
    for ver in VERSIONS:
        tf_min = dca_grid.get_native_tf_min(ver)
        for sym in UNIVERSE:
            bars_1h = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
            bars_v = dca_grid.load_bars(snapshot_dir, sym, tf_min, START, END)
            if bars_v.empty or bars_1h.empty:
                continue
            sigs = dca_grid.generate_entry_signals(bars_v, ver, bars_1h)
            bars[(sym, ver)] = bars_v
            signals[(sym, ver)] = sigs
    print(f"Loaded {len(bars)} (sym, ver) pairs")

    params = {**BASE_PARAMS, "base_order_usd": 0, "base_pct_of_equity": BASE_PCT}
    print(f"Simulating...")
    result = dca_grid.simulate_portfolio(signals, bars, params, STARTING_CAPITAL)
    closed = result["closed_deals"]
    m = dca_grid.compute_metrics(result)
    print(f"  CAGR: {m['cagr_pct']:+.2f}%, deals: {m['n_deals']}, "
          f"final $: ${m['final_equity']:.0f}")

    # Per-day net P&L from deal closes
    rows = []
    for d in closed:
        rows.append({
            "close_date": d.close_ts.normalize(),
            "net_pnl_usd": d.realized_pnl_usd,
            "version": d.deal.version,
        })
    df = pd.DataFrame(rows)
    daily = df.groupby("close_date")["net_pnl_usd"].sum().sort_index()

    # Fill missing days with 0 P&L so the rolling-30d window is calendar-based
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    rolling_30d = daily.rolling("30D").sum()

    # Convert to % of equity-at-start-of-window (rough — uses running equity)
    # Approx: start with STARTING_CAPITAL, add running sum of daily P&L
    eq = STARTING_CAPITAL + daily.cumsum()
    eq_30d_ago = eq.shift(30).fillna(STARTING_CAPITAL)
    rolling_30d_pct = (rolling_30d / eq_30d_ago) * 100

    print()
    print("=== Rolling 30-day P&L distribution ===")
    print(f"{'metric':<25} {'$ value':>10s}  {'% equity':>10s}")
    print("-" * 50)
    for label, q in [("min (worst 30d)", rolling_30d.min()),
                     ("p10", rolling_30d.quantile(0.10)),
                     ("p25", rolling_30d.quantile(0.25)),
                     ("median", rolling_30d.median()),
                     ("mean", rolling_30d.mean()),
                     ("p75", rolling_30d.quantile(0.75)),
                     ("p90", rolling_30d.quantile(0.90)),
                     ("p95", rolling_30d.quantile(0.95)),
                     ("p99", rolling_30d.quantile(0.99)),
                     ("max (best 30d)", rolling_30d.max())]:
        q_pct = (q / STARTING_CAPITAL) * 100  # rough — % of starting capital
        print(f"{label:<25} ${q:>+9.0f}  {q_pct:>+9.2f}%")

    # Best + worst window dates for context
    best_end = rolling_30d.idxmax()
    worst_end = rolling_30d.idxmin()
    print()
    print(f"Best 30d window ends: {best_end.date()} "
          f"(${rolling_30d.max():+.0f}, ~{(rolling_30d.max()/STARTING_CAPITAL)*100:+.1f}% of starting capital)")
    print(f"Worst 30d window ends: {worst_end.date()} "
          f"(${rolling_30d.min():+.0f}, ~{(rolling_30d.min()/STARTING_CAPITAL)*100:+.1f}% of starting capital)")

    # % equity using running equity (more honest for compounding strategy)
    print()
    print(f"Note: % equity uses STARTING capital ($3K) as denominator (conservative).")
    print(f"Late-window % values are misleadingly large because equity grew "
          f"to ${eq.iloc[-1]:.0f}.")
    print(f"Median 30d P&L as fraction of contemporaneous equity (rolling): "
          f"{(rolling_30d / eq).median()*100:+.2f}%")
    print(f"Max 30d P&L as fraction of contemporaneous equity (rolling): "
          f"{(rolling_30d / eq).max()*100:+.2f}%")
    print(f"Min 30d P&L as fraction of contemporaneous equity (rolling): "
          f"{(rolling_30d / eq).min()*100:+.2f}%")

    out = {
        "config": {
            "N": N, "versions": VERSIONS, "base_pct": BASE_PCT,
            "friction_vol_30d": BASE_PARAMS["fixed_friction_vol_30d"],
            "window": [START, END],
            "starting_capital": STARTING_CAPITAL,
        },
        "summary": {
            "cagr_pct": m["cagr_pct"],
            "n_deals": m["n_deals"],
            "final_equity": m["final_equity"],
        },
        "rolling_30d_pnl_usd": {
            "min": float(rolling_30d.min()),
            "p10": float(rolling_30d.quantile(0.10)),
            "p25": float(rolling_30d.quantile(0.25)),
            "median": float(rolling_30d.median()),
            "mean": float(rolling_30d.mean()),
            "p75": float(rolling_30d.quantile(0.75)),
            "p90": float(rolling_30d.quantile(0.90)),
            "p95": float(rolling_30d.quantile(0.95)),
            "p99": float(rolling_30d.quantile(0.99)),
            "max": float(rolling_30d.max()),
        },
        "rolling_30d_pct_of_equity_contemporaneous": {
            "min": float((rolling_30d / eq).min()) * 100,
            "median": float((rolling_30d / eq).median()) * 100,
            "max": float((rolling_30d / eq).max()) * 100,
        },
        "best_30d_window_end_date": str(best_end.date()),
        "worst_30d_window_end_date": str(worst_end.date()),
    }
    out_path = ROOT / "reports" / "rolling-30d-n12-mv-b050.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
