"""Regenerate heuristic regime labels via sidecar (no local SQLite).

Mirrors scripts/build-heuristic-regime-labels.py but:
  * loads BTC daily from the server sidecar's /api/v1/intraday endpoint
    (per server-team snippet on issue #22 issuecomment-4485..., 14:38Z)
  * parameterizes heuristic_label() to accept threshold dicts directly,
    so the same code path serves both the canonical regenerator and a
    LOO threshold sweep.

When called with --canonical-strict, produces the same labels as
server team's data/server_research_labels/heuristic_strict_continuous_regime_labels.parquet
(to within BTC-source rounding differences near label-boundary days).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SIDECAR_INTRADAY_URL = (
    "http://100.68.86.56:8443/api/v1/intraday?symbol=BTC-USD&interval_min=60"
)


# Canonical thresholds (strict variant) — straight copy from
# scripts/build-heuristic-regime-labels.py heuristic_label()
CANONICAL_STRICT = {
    "bear":   {"dd60": -0.25, "r30": -0.08, "sma": 0.95},
    "chop":   {"dd60": -0.20, "r30": 0.10,  "vol": 1.2},
    "steady": {"sma": 1.15,   "r60": 0.30,  "vol": 0.85, "dd60": -0.05, "r30": 0.10},
}


def load_btc_daily_sidecar(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch BTC-USD 1h bars from sidecar, resample to daily close.

    Note (from server team issue #22, 14:26Z): the /intraday endpoint
    currently ignores start/end query params and returns all rows for
    the symbol — so we client-side filter.
    """
    print(f"[sidecar] GET {SIDECAR_INTRADAY_URL}", file=sys.stderr)
    data = urllib.request.urlopen(SIDECAR_INTRADAY_URL, timeout=60).read()
    df = pd.read_parquet(io.BytesIO(data))
    # Sidecar /intraday ignores the symbol query param and returns ALL symbols
    # (16 symbols, ~1.6M rows). Must filter client-side. See issue #22 14:26Z.
    df = df[df["symbol"] == "BTC-USD"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    print(f"[sidecar] got {len(df)} BTC intraday rows (after symbol filter), "
          f"range {df['timestamp'].min()} -> {df['timestamp'].max()}", file=sys.stderr)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    daily = df.set_index("timestamp")["close"].resample("1D").last().dropna().to_frame("close")
    print(f"[sidecar] resampled to {len(daily)} daily bars", file=sys.stderr)
    return daily


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
    return 1.0 / (1.0 + np.exp(-x * scale))


def heuristic_label_continuous(row, thresholds: dict) -> tuple[str, float]:
    """Continuous-confidence variant, parameterized by threshold dict.

    `thresholds` shape:
        {
          "bear":   {"dd60": float, "r30": float, "sma": float},
          "chop":   {"dd60": float, "r30": float, "vol": float},
          "steady": {"sma": float, "r60": float, "vol": float, "dd60": float, "r30": float?},
        }
    """
    dd60 = row["drawdown_60d"]
    r30 = row["r_30d"]
    r60 = row["r_60d"]
    sma_r = row["sma_ratio"]
    vol_r = row["vol_ratio"]

    if any(pd.isna(x) for x in [dd60, r30, r60, sma_r, vol_r]):
        return "sideways_range", 0.5

    bear_t = thresholds["bear"]
    chop_t = thresholds["chop"]
    steady_t = thresholds["steady"]

    bear_score = float(np.mean([
        _sig(bear_t["dd60"] - dd60, scale=10),
        _sig(bear_t["r30"]  - r30,  scale=20),
        _sig(bear_t["sma"]  - sma_r, scale=20),
    ]))
    chop_score = float(np.mean([
        _sig(chop_t["dd60"] - dd60, scale=10),
        _sig(r30 - chop_t["r30"],   scale=20),
        _sig(vol_r - chop_t["vol"], scale=10),
    ]))
    steady_conds = [
        _sig(sma_r - steady_t["sma"], scale=20),
        _sig(r60   - steady_t["r60"], scale=10),
        _sig(steady_t["vol"] - vol_r, scale=10),
        _sig(dd60  - steady_t["dd60"], scale=20),
    ]
    if "r30" in steady_t:
        steady_conds.append(_sig(r30 - steady_t["r30"], scale=20))
    steady_score = float(np.mean(steady_conds))
    sideways_score = float(max(0.0, 1.0 - max(bear_score, chop_score, steady_score)))

    fits = {
        "bear_trend":      bear_score,
        "choppy_recovery": chop_score,
        "steady_bull":     steady_score,
        "sideways_range":  sideways_score,
    }
    label = max(fits, key=fits.get)
    return label, fits[label]


def generate_labels(thresholds: dict, start: str, end: str) -> pd.DataFrame:
    daily = load_btc_daily_sidecar(start=start, end=end)
    feat = compute_features(daily)
    rows = feat.apply(
        lambda r: heuristic_label_continuous(r, thresholds=thresholds),
        axis=1, result_type="expand",
    )
    feat["regime_label"] = rows[0]
    feat["regime_confidence"] = rows[1]
    feat["rule_label"] = feat["regime_label"]
    out = feat.reset_index()[["timestamp", "regime_label", "regime_confidence", "rule_label"]].copy()
    out.columns = ["date", "regime_label", "regime_confidence", "rule_label"]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if out["date"].dt.tz is None:
        out["date"] = out["date"].dt.tz_localize("UTC")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-05-17T23:59:59Z")
    p.add_argument("--out", required=True, help="Output parquet path.")
    p.add_argument("--canonical-strict", action="store_true",
                   help="Use the canonical --strict --continuous thresholds "
                        "(reproduces server team's published labels).")
    args = p.parse_args()

    if not args.canonical_strict:
        print("ERROR: only --canonical-strict mode implemented for now. "
              "Sweep variants come via direct generate_labels() import.",
              file=sys.stderr)
        return 2

    out = generate_labels(CANONICAL_STRICT, args.start, args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[ok] wrote {len(out)} rows -> {out_path}")
    print(f"\nLabel distribution:\n{out['regime_label'].value_counts().to_string()}")
    print(f"\nConfidence: mean={out['regime_confidence'].mean():.3f}  "
          f"min={out['regime_confidence'].min():.3f}  "
          f"max={out['regime_confidence'].max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
