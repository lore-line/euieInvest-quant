#!/usr/bin/env python3
"""Heuristic regime gate (option C from issue #22 thread) — naive BTC-based rules
mapping to the consumer-team's 4-class regime taxonomy.

Builds a parquet matching the schema of regime_labels_v1.parquet so it can drop
into the multi-strategy harness via --regime-labels-path without changes.

Features (all derived from BTC-USD daily closes via 1h resample):
  - r_30d, r_60d: trailing returns
  - drawdown_60d: distance from trailing-60d high (negative when below)
  - sma_50, sma_200: simple moving averages
  - sma_ratio = sma_50 / sma_200 (>1 = uptrend)
  - vol_30d, vol_60d: stddev of daily returns
  - vol_ratio = vol_30d / vol_60d (>1 = recent vol elevated)

Labels (rules; one per row; argmax by condition-satisfaction count):
  - bear_trend: drawdown_60d ≤ -20% AND r_30d ≤ -5% AND sma_ratio < 1.0
  - choppy_recovery: drawdown_60d ≤ -15% AND r_30d ≥ +5% AND vol_ratio > 1.0
  - steady_bull: sma_ratio ≥ 1.05 AND r_60d ≥ +20% AND vol_ratio < 1.0 AND drawdown_60d > -10%
  - sideways_range: catch-all (|r_30d| < 10% AND |sma_ratio - 1| < 10% AND drawdown_60d > -15%)

Confidence = fraction of sub-conditions met for the chosen label (in [0, 1]).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "euieinvest.db"


def load_btc_daily(start: str, end: str) -> pd.DataFrame:
    """Resample BTC-USD 1h bars to daily close."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        """SELECT timestamp, close FROM intraday_history
           WHERE symbol = 'BTC-USD' AND interval_min = 60
             AND timestamp >= ? AND timestamp <= ?
           ORDER BY timestamp ASC""",
        (start, end),
    ).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    daily = df["close"].resample("1D").last().dropna()
    return daily.to_frame("close")


def compute_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["daily_ret"] = df["close"].pct_change()
    df["r_30d"] = df["close"].pct_change(30)
    df["r_60d"] = df["close"].pct_change(60)
    df["max_60d"] = df["close"].rolling(60).max()
    df["drawdown_60d"] = (df["close"] - df["max_60d"]) / df["max_60d"]
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["sma_ratio"] = df["sma_50"] / df["sma_200"]
    df["vol_30d"] = df["daily_ret"].rolling(30).std()
    df["vol_60d"] = df["daily_ret"].rolling(60).std()
    df["vol_ratio"] = df["vol_30d"] / df["vol_60d"]
    return df


def _sig(x: float, scale: float = 10.0) -> float:
    """Logistic sigmoid for soft constraint scoring. x=0 → 0.5, large x → 1."""
    return 1.0 / (1.0 + np.exp(-x * scale))


def heuristic_label(row, strict: bool = False, continuous: bool = False) -> tuple[str, float]:
    """Returns (label, confidence) for a single day's features.

    `strict=True`: tighter thresholds; steady_bull fires ~5% of days.
    `continuous=True`: confidence is a continuous logistic-sigmoid score over
        the constraint slacks instead of a discrete fraction-of-conditions.
        Allows fine-grained threshold tuning (replaces the discrete
        0.5/0.75/1.0 buckets in the boolean version).
    """
    dd60 = row["drawdown_60d"]
    r30 = row["r_30d"]
    r60 = row["r_60d"]
    sma_r = row["sma_ratio"]
    vol_r = row["vol_ratio"]

    if any(pd.isna(x) for x in [dd60, r30, r60, sma_r, vol_r]):
        return "sideways_range", 0.5

    if strict:
        bear_thresh   = {"dd60": -0.25, "r30": -0.08, "sma": 0.95}
        chop_thresh   = {"dd60": -0.20, "r30": 0.10,  "vol": 1.2}
        steady_thresh = {"sma": 1.15,   "r60": 0.30,  "vol": 0.85, "dd60": -0.05, "r30": 0.10}
    else:
        bear_thresh   = {"dd60": -0.20, "r30": -0.05, "sma": 1.0}
        chop_thresh   = {"dd60": -0.15, "r30": 0.05,  "vol": 1.0}
        steady_thresh = {"sma": 1.05,   "r60": 0.20,  "vol": 1.0,  "dd60": -0.10}

    if continuous:
        # Each sub-condition contributes a soft score in (0, 1). Average → label fit.
        bear_score = np.mean([
            _sig(bear_thresh["dd60"] - dd60, scale=10),   # large when dd60 is far below threshold
            _sig(bear_thresh["r30"]  - r30,  scale=20),
            _sig(bear_thresh["sma"]  - sma_r, scale=20),
        ])
        chop_score = np.mean([
            _sig(chop_thresh["dd60"] - dd60, scale=10),
            _sig(r30 - chop_thresh["r30"],   scale=20),
            _sig(vol_r - chop_thresh["vol"], scale=10),
        ])
        steady_conds = [
            _sig(sma_r - steady_thresh["sma"], scale=20),
            _sig(r60   - steady_thresh["r60"], scale=10),
            _sig(steady_thresh["vol"] - vol_r, scale=10),
            _sig(dd60  - steady_thresh["dd60"], scale=20),
        ]
        if "r30" in steady_thresh:
            steady_conds.append(_sig(r30 - steady_thresh["r30"], scale=20))
        steady_score = np.mean(steady_conds)
        sideways_score = 1.0 - max(bear_score, chop_score, steady_score)
        fits = {
            "bear_trend":      float(bear_score),
            "choppy_recovery": float(chop_score),
            "steady_bull":     float(steady_score),
            "sideways_range":  float(max(0.0, sideways_score)),
        }
    else:
        if strict:
            bear_conds = [dd60 <= -0.25, r30 <= -0.08, sma_r < 0.95]
            chop_conds = [dd60 <= -0.20, r30 >= 0.10, vol_r > 1.2]
            steady_conds = [sma_r >= 1.15, r60 >= 0.30, vol_r < 0.85, dd60 > -0.05, r30 >= 0.10]
            sideways_conds = [abs(r30) < 0.05, 0.97 < sma_r < 1.05, dd60 > -0.10]
        else:
            bear_conds = [dd60 <= -0.20, r30 <= -0.05, sma_r < 1.0]
            chop_conds = [dd60 <= -0.15, r30 >= 0.05, vol_r > 1.0]
            steady_conds = [sma_r >= 1.05, r60 >= 0.20, vol_r < 1.0, dd60 > -0.10]
            sideways_conds = [abs(r30) < 0.10, 0.95 < sma_r < 1.10, dd60 > -0.15]
        fits = {
            "bear_trend":      sum(bear_conds) / len(bear_conds),
            "choppy_recovery": sum(chop_conds) / len(chop_conds),
            "steady_bull":     sum(steady_conds) / len(steady_conds),
            "sideways_range":  sum(sideways_conds) / len(sideways_conds),
        }
    label = max(fits, key=fits.get)
    return label, fits[label]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-05-17T23:59:59Z")
    p.add_argument("--out", default=str(REPO / "data" / "heuristic_regime_labels.parquet"))
    p.add_argument("--strict", action="store_true",
                   help="Tighten thresholds so steady_bull fires ~5% of days "
                        "(matches consumer P1 v0.4 selectivity)")
    p.add_argument("--continuous", action="store_true",
                   help="Use continuous sigmoid-based confidence scoring "
                        "instead of discrete fraction-of-conditions (allows "
                        "fine-grained threshold tuning).")
    args = p.parse_args()

    print(f"[heuristic-regime] loading BTC daily {args.start}→{args.end}")
    daily = load_btc_daily(args.start, args.end)
    print(f"[heuristic-regime] loaded {len(daily)} daily bars")

    df = compute_features(daily)
    labels = df.apply(
        lambda row: heuristic_label(row, strict=args.strict, continuous=args.continuous),
        axis=1, result_type="expand",
    )
    df["regime_label"] = labels[0]
    df["regime_confidence"] = labels[1]
    df["rule_label"] = df["regime_label"]   # for schema compatibility

    out = df.reset_index()[["timestamp", "regime_label", "regime_confidence",
                            "rule_label"]].copy()
    out.columns = ["date", "regime_label", "regime_confidence", "rule_label"]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if out["date"].dt.tz is None:
        out["date"] = out["date"].dt.tz_localize("UTC")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[heuristic-regime] wrote {len(out)} rows → {out_path}")
    print(f"\nLabel distribution:")
    print(out["regime_label"].value_counts().to_string())
    print(f"\nConfidence stats: mean={out['regime_confidence'].mean():.3f}  "
          f"min={out['regime_confidence'].min():.3f}  "
          f"max={out['regime_confidence'].max():.3f}")


if __name__ == "__main__":
    main()
