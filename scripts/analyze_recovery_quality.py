#!/usr/bin/env python3
"""Per-symbol recovery quality / mean-reversion analysis.

The per-symbol cliff vs ATR analysis found cliff position is BIMODAL —
some symbols tolerate b up to 50%, others fail at <3%. ATR doesn't
discriminate (CoV 80% in regression).

Hypothesis: the real driver is mean-reversion quality. Symbols whose
drawdowns recover cleanly tolerate high b (DCA ladders fill, TPs eventually
hit). Symbols whose drawdowns drift further down break the strategy.

This script computes several mean-reversion metrics per symbol and
correlates against the cliff position from per-symbol-cliff-vs-atr.json:

  1. lag-1 autocorrelation of log-returns
     - negative = mean-reverting (good)
     - positive = trending (bad)
  2. Max-drawdown-from-200d-high duration
     - shorter = faster recovery (good)
  3. P(drawdown > 30%) over 60-day rolling windows
     - lower = more stable (good)
  4. Skewness of returns
     - negative skew = downside-heavy (bad for DCA buyers)

Output: ranking + regression of cliff_b against each metric.
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

MASTER_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
    "INJ-USD", "GRT-USD", "AAVE-USD", "UNI-USD",
    "NEAR-USD", "SUSHI-USD", "APT-USD", "TIA-USD",
]


def compute_metrics(bars: pd.DataFrame) -> dict:
    c = bars["close"]
    rets = np.log(c / c.shift(1)).dropna()

    # 1. Lag-1 autocorrelation of returns
    autocorr_1 = float(rets.autocorr(lag=1))
    autocorr_5 = float(rets.autocorr(lag=5))   # also check 5-bar autocorr
    autocorr_60 = float(rets.autocorr(lag=60)) # daily-ish autocorr (60×60m)

    # 2. Drawdown analysis from rolling 200-bar (≈8d at 60m) high
    rolling_max = c.rolling(window=200, min_periods=1).max()
    drawdown = (c / rolling_max) - 1
    max_dd = float(drawdown.min())
    p_dd_30 = float((drawdown < -0.30).mean())  # fraction of bars in deep DD

    # 3. Mean recovery time from -10% drawdown
    underwater = drawdown < -0.10
    underwater_runs = []
    cur_run = 0
    for u in underwater:
        if u:
            cur_run += 1
        elif cur_run > 0:
            underwater_runs.append(cur_run)
            cur_run = 0
    if cur_run > 0:
        underwater_runs.append(cur_run)
    mean_recovery_bars = float(np.mean(underwater_runs)) if underwater_runs else 0.0

    # 4. Skewness of returns
    rets_skew = float(rets.skew())

    # 5. Realized 60-day Sharpe (proxy for trend vs chop)
    daily_rets = rets.resample("1D").sum().dropna()
    sharpe_60d = float(daily_rets.tail(60).mean() / daily_rets.tail(60).std() * np.sqrt(365)) if len(daily_rets) >= 60 else 0.0

    return {
        "autocorr_1": autocorr_1,
        "autocorr_5": autocorr_5,
        "autocorr_60": autocorr_60,
        "max_drawdown": max_dd,
        "p_drawdown_30pct": p_dd_30,
        "mean_recovery_bars": mean_recovery_bars,
        "returns_skew": rets_skew,
        "sharpe_60d_terminal": sharpe_60d,
    }


def main() -> int:
    cliff_path = ROOT / "reports" / "per-symbol-cliff-vs-atr.json"
    cliff_data = json.loads(cliff_path.read_text())
    cliff_by_sym = {r["symbol"]: r for r in cliff_data["summary"]}

    print(f"Per-symbol recovery quality analysis")
    print(f"Source: per-symbol-cliff-vs-atr.json ({len(cliff_by_sym)} symbols)")
    print()

    snapshot_dir = dca_grid.SNAPSHOT_DIR
    print(f"Computing metrics on 60m bars...\n")
    print(f"{'symbol':<10s} | {'cliff b%':>9s} | {'autocorr_1':>11s} | "
          f"{'max_DD':>8s} | {'%DD>30':>7s} | {'mean_rec_bars':>14s} | {'rets_skew':>10s}")
    print("-" * 95)

    rows = []
    for sym in MASTER_UNIVERSE:
        bars = dca_grid.load_bars(snapshot_dir, sym, 60, START, END)
        if bars.empty:
            continue
        m = compute_metrics(bars)
        cliff = cliff_by_sym.get(sym, {})
        cliff_b = cliff.get("cliff_b_pct", None)
        cliff_str = f"{cliff_b:.2f}%" if cliff_b is not None else "?"
        print(f"{sym:<10s} | {cliff_str:>9s} | {m['autocorr_1']:>+10.4f} | "
              f"{m['max_drawdown']*100:>+7.1f}% | {m['p_drawdown_30pct']*100:>6.1f}% | "
              f"{m['mean_recovery_bars']:>13.0f} | {m['returns_skew']:>+9.3f}")
        rows.append({"symbol": sym, "cliff_b": cliff_b, **m})

    # Correlation of cliff_b against each metric
    valid = [r for r in rows if r["cliff_b"] is not None]
    if len(valid) >= 5:
        cliff_arr = np.array([r["cliff_b"] for r in valid])
        print(f"\n{'='*60}")
        print(f"CORRELATION of cliff_b vs each metric")
        print(f"{'='*60}")
        for metric in ["autocorr_1", "autocorr_5", "autocorr_60",
                       "max_drawdown", "p_drawdown_30pct",
                       "mean_recovery_bars", "returns_skew",
                       "sharpe_60d_terminal"]:
            vals = np.array([r[metric] for r in valid])
            if vals.std() == 0:
                continue
            r_pearson = float(np.corrcoef(cliff_arr, vals)[0, 1])
            print(f"  {metric:<24s}: ρ = {r_pearson:+.3f}")

        # Rank cliff_b high vs low, see if the metrics differ
        sorted_by_cliff = sorted(valid, key=lambda r: r["cliff_b"], reverse=True)
        high_cliff = sorted_by_cliff[:len(sorted_by_cliff)//2]
        low_cliff = sorted_by_cliff[len(sorted_by_cliff)//2:]
        print(f"\n{'='*60}")
        print(f"HIGH-CLIFF (top half) vs LOW-CLIFF (bottom half) mean metrics")
        print(f"{'='*60}")
        print(f"{'metric':<24s} | {'high-cliff':>12s} | {'low-cliff':>12s} | "
              f"{'spread':>9s}")
        print("-" * 70)
        for metric in ["autocorr_1", "autocorr_5", "max_drawdown",
                       "p_drawdown_30pct", "mean_recovery_bars",
                       "returns_skew", "sharpe_60d_terminal"]:
            hi = np.mean([r[metric] for r in high_cliff])
            lo = np.mean([r[metric] for r in low_cliff])
            print(f"  {metric:<22s} | {hi:>+11.4f} | {lo:>+11.4f} | "
                  f"{hi-lo:>+8.4f}")

    out_path = ROOT / "reports" / "recovery-quality.json"
    out_path.write_text(json.dumps({
        "window": [START, END],
        "rows": rows,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
