#!/usr/bin/env python3
"""T2 scorer test — adds volume slope to T1.6.

Existing scorer (T1.6, doctrine §2.7):
  score_t16 = ATR^5 × clip(volume/$5M, 1.0) × friction^-1

Proposed T2 (adds volume health):
  score_t2 = score_t16 × vol_health_factor(vol_slope_yr)

Where vol_health_factor:
  - Linear:    max(0.1, 1 + clip(vol_slope_yr, -0.5, +0.5))
  - Hard cut:  0 if vol_slope_yr < -0.1, 1 otherwise
  - Sigmoid:   1 / (1 + exp(-vol_slope_yr × 4))

Compares both scorers against per-symbol max-CAGR (the true
harvestability target) AND cliff_b (deployment capacity proxy).
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

START = "2022-09-15"
END = "2026-05-17"
VOLUME_FLOOR_USD = 5_000_000


def symbol_features(snapshot_dir: Path, symbol: str) -> dict | None:
    bars = dca_grid.load_bars(snapshot_dir, symbol, 60, START, END)
    if bars.empty:
        return None
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    # ATR(14)% on 60m bars
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    atr_pct = float((atr / c).dropna().mean() * 100)
    # Mean daily dollar volume
    dv_60m = c * v
    daily_dv = dv_60m.resample("1D").sum().dropna()
    mean_dv = float(daily_dv.mean())
    # Volume slope (log-DV per year)
    y = np.log(daily_dv.replace(0, np.nan).dropna().values)
    x = np.arange(len(y)) / 365
    vol_slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 10 else 0.0
    # Volume z-score (last 60d vs full window)
    last_60d = daily_dv.tail(60).mean()
    vol_z = float((last_60d - daily_dv.mean()) / daily_dv.std()) if daily_dv.std() > 0 else 0.0
    return {
        "atr_pct": atr_pct,
        "mean_dv": mean_dv,
        "vol_slope_yr": vol_slope,
        "vol_z_late": vol_z,
    }


def t16_score(f: dict, friction_rt: float = 0.0016) -> float:
    """T1.6: ATR^5 × clip(volume/floor, 1.0) × friction^-1."""
    vol_gate = min(f["mean_dv"] / VOLUME_FLOOR_USD, 1.0)
    return (f["atr_pct"] ** 5) * vol_gate / friction_rt


def vol_health_linear(slope: float) -> float:
    return max(0.1, 1 + max(-0.5, min(0.5, slope)))


def vol_health_hard(slope: float) -> float:
    return 0.0 if slope < -0.1 else 1.0


def vol_health_sigmoid(slope: float) -> float:
    return float(1 / (1 + np.exp(-slope * 4)))


def t2_score(f: dict, health_fn=vol_health_linear, friction_rt: float = 0.0016) -> float:
    base = t16_score(f, friction_rt)
    return base * health_fn(f["vol_slope_yr"])


def rank_corr(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    cliff_data = json.loads((ROOT / "reports" / "per-symbol-cliff-vs-atr.json").read_text())
    cliff_by_sym = {r["symbol"]: r for r in cliff_data["summary"]}

    snapshot_dir = dca_grid.SNAPSHOT_DIR
    print(f"Computing T1.6 + T2 scores for {len(cliff_by_sym)} symbols...")
    rows = []
    for sym in cliff_by_sym:
        f = symbol_features(snapshot_dir, sym)
        if f is None:
            continue
        r = cliff_by_sym[sym]
        rows.append({
            "symbol": sym,
            "cliff_b": r.get("cliff_b_pct"),
            "max_cagr": r.get("max_cagr_pct"),
            "max_cagr_b": r.get("max_cagr_b_pct"),
            "atr_pct": f["atr_pct"],
            "mean_dv_M": f["mean_dv"] / 1e6,
            "vol_slope_yr": f["vol_slope_yr"],
            "vol_z_late": f["vol_z_late"],
            "t16": t16_score(f),
            "t2_linear": t2_score(f, vol_health_linear),
            "t2_hard": t2_score(f, vol_health_hard),
            "t2_sigmoid": t2_score(f, vol_health_sigmoid),
        })

    print(f"\n{'symbol':<10s} | {'cliff%':>6s} | {'max_CAGR':>8s} | "
          f"{'T1.6':>10s} | {'T2_lin':>10s} | {'T2_hard':>10s} | {'T2_sig':>10s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['symbol']:<10s} | {r['cliff_b']:>5.1f}% | "
              f"{r['max_cagr']:>+7.2f}% | "
              f"{r['t16']:>9.2e} | "
              f"{r['t2_linear']:>9.2e} | "
              f"{r['t2_hard']:>9.2e} | "
              f"{r['t2_sigmoid']:>9.2e}")

    # IC tests
    cliff = np.array([r["cliff_b"] for r in rows])
    max_cagr = np.array([r["max_cagr"] for r in rows])
    print(f"\n{'='*70}")
    print(f"SCORER IC (Spearman rank correlation)")
    print(f"{'='*70}")
    print(f"{'scorer':<15s} | {'vs cliff_b':>12s} | {'vs max_CAGR':>13s}")
    print("-" * 50)
    for scorer_name in ["t16", "t2_linear", "t2_hard", "t2_sigmoid"]:
        scores = np.array([r[scorer_name] for r in rows])
        ic_cliff = rank_corr(scores, cliff)
        ic_cagr = rank_corr(scores, max_cagr)
        print(f"{scorer_name:<15s} | {ic_cliff:>+10.3f}   | {ic_cagr:>+11.3f}")

    # Top-5 and Bottom-5 by each scorer — does T2 pick better?
    print(f"\n{'='*70}")
    print("TOP-5 PICKS BY EACH SCORER (and the realized max_CAGR)")
    print(f"{'='*70}")
    for scorer_name in ["t16", "t2_linear", "t2_hard"]:
        ranked = sorted(rows, key=lambda r: r[scorer_name], reverse=True)
        top5 = ranked[:5]
        print(f"\n{scorer_name}:")
        for r in top5:
            print(f"  {r['symbol']:<10s}  score={r[scorer_name]:.2e}  "
                  f"cliff={r['cliff_b']:.1f}%  max_CAGR={r['max_cagr']:+.2f}%  "
                  f"vol_slope={r['vol_slope_yr']:+.2f}/yr")
        mean_top5_cagr = np.mean([r["max_cagr"] for r in top5])
        print(f"  → mean max_CAGR of top-5: {mean_top5_cagr:+.2f}%")

    print(f"\n{'='*70}")
    print("BOTTOM-5 PICKS BY EACH SCORER (would-be-skipped symbols)")
    print(f"{'='*70}")
    for scorer_name in ["t16", "t2_linear", "t2_hard"]:
        ranked = sorted(rows, key=lambda r: r[scorer_name])
        bot5 = ranked[:5]
        print(f"\n{scorer_name}:")
        for r in bot5:
            print(f"  {r['symbol']:<10s}  score={r[scorer_name]:.2e}  "
                  f"cliff={r['cliff_b']:.1f}%  max_CAGR={r['max_cagr']:+.2f}%  "
                  f"vol_slope={r['vol_slope_yr']:+.2f}/yr")
        mean_bot5_cagr = np.mean([r["max_cagr"] for r in bot5])
        print(f"  → mean max_CAGR of bottom-5: {mean_bot5_cagr:+.2f}%")

    out = ROOT / "reports" / "t2-scorer-test.json"
    out.write_text(json.dumps({"rows": rows}, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
