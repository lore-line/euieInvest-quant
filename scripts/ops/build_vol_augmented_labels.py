"""Build regime_labels_v2 augmented with rolling-252d vol-tercile column.

Output: data/server_research_labels/cliff_aware_variants/regime_labels_v2_vol_augmented.parquet

For issue #25 cliff-aware deployment scaling. Tercile bins are WF-OOS-correct
(rolling-252d, no look-ahead) per server-team design correction #1 (2026-05-21).
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SIDECAR_INTRADAY_URL = (
    "http://100.68.86.56:8443/api/v1/intraday?symbol=BTC-USD&interval_min=60"
)
SOURCE_LABELS = REPO / "data" / "quant_publish" / "regime_labels_v2.parquet"
OUT_DIR = REPO / "data" / "server_research_labels" / "cliff_aware_variants"
OUT_PATH = OUT_DIR / "regime_labels_v2_vol_augmented.parquet"

# Vol classifier params
VOL_WINDOW_DAYS = 30      # realized-vol estimation window
TERCILE_WINDOW_DAYS = 252  # rolling-tercile binning window (~1 trading year)
TERCILE_MIN_HISTORY = 60   # need at least this much history before assigning


def load_btc_daily_sidecar() -> pd.DataFrame:
    """Fetch BTC-USD 1h bars from sidecar, resample to daily close."""
    print(f"[sidecar] GET {SIDECAR_INTRADAY_URL}", file=sys.stderr)
    data = urllib.request.urlopen(SIDECAR_INTRADAY_URL, timeout=60).read()
    df = pd.read_parquet(io.BytesIO(data))
    # Sidecar /intraday ignores symbol query param; client-side filter required.
    df = df[df["symbol"] == "BTC-USD"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    daily = df.set_index("timestamp")["close"].resample("1D").last().dropna().to_frame("close")
    print(f"[sidecar] resampled to {len(daily)} BTC daily bars", file=sys.stderr)
    return daily


def compute_vol_terciles(daily: pd.DataFrame) -> pd.DataFrame:
    """Add log-returns, rolling-30d realized vol, and rolling-252d tercile bin.

    Tercile assignment is WF-OOS-correct: for day t, use the prior 252-day
    distribution of vol_30d (excluding day t itself) to compute q33/q67, then
    classify vol_30d[t] into {low, mid, high}. No look-ahead.
    """
    df = daily.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    df["vol_30d"] = df["log_ret"].rolling(VOL_WINDOW_DAYS).std()

    # Rolling tercile assignment. Use a numeric encoding (0/1/2) for the
    # rolling.apply() return-type constraint, then map to string at the end.
    def _tercile_numeric(window: np.ndarray) -> float:
        # window includes the current value as the last element
        if len(window) < TERCILE_MIN_HISTORY:
            return np.nan
        current = window[-1]
        history = window[:-1]
        history = history[~np.isnan(history)]
        if len(history) < TERCILE_MIN_HISTORY or np.isnan(current):
            return np.nan
        q33, q67 = np.percentile(history, [33.33, 66.67])
        if current <= q33:
            return 0.0
        elif current >= q67:
            return 2.0
        else:
            return 1.0

    df["vol_tercile_num"] = (
        df["vol_30d"]
        .rolling(TERCILE_WINDOW_DAYS, min_periods=TERCILE_MIN_HISTORY)
        .apply(_tercile_numeric, raw=True)
    )
    tercile_map = {0.0: "low", 1.0: "mid", 2.0: "high"}
    df["vol_tercile"] = df["vol_tercile_num"].map(tercile_map)
    # Days before TERCILE_MIN_HISTORY days of vol_30d are NaN -> set to "mid"
    # so the harness has a safe default rather than failing on NaN. Document
    # this in README.
    df["vol_tercile"] = df["vol_tercile"].fillna("mid")
    return df


def main() -> int:
    # Load source regime labels.
    if not SOURCE_LABELS.exists():
        print(f"ERROR: missing {SOURCE_LABELS}", file=sys.stderr)
        return 1
    labels = pd.read_parquet(SOURCE_LABELS)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    if labels["date"].dt.tz is None:
        labels["date"] = labels["date"].dt.tz_localize("UTC")
    print(f"[labels] loaded {len(labels)} rows, "
          f"range {labels['date'].min().date()} -> {labels['date'].max().date()}",
          file=sys.stderr)

    # Compute vol + tercile on BTC daily.
    btc = load_btc_daily_sidecar()
    btc_vol = compute_vol_terciles(btc)
    btc_vol = btc_vol.reset_index().rename(columns={"timestamp": "date"})
    btc_vol["date"] = btc_vol["date"].dt.normalize()
    if btc_vol["date"].dt.tz is None:
        btc_vol["date"] = btc_vol["date"].dt.tz_localize("UTC")

    # Merge labels with vol info on date.
    vol_slim = btc_vol[["date", "vol_30d", "vol_tercile"]].copy()
    out = labels.merge(vol_slim, on="date", how="left")

    # If any label date didn't have a vol observation (shouldn't happen given
    # date coverage, but defensive), fall back to mid.
    n_missing = out["vol_tercile"].isna().sum()
    if n_missing:
        print(f"[warn] {n_missing} label dates had no vol observation; "
              "defaulting to 'mid'", file=sys.stderr)
        out["vol_tercile"] = out["vol_tercile"].fillna("mid")

    # Drop tz from date for compatibility with downstream loaders that
    # assume naive dates (e.g., harness load_regime_lookup() does
    # `.dt.tz_localize("UTC")` and fails on tz-aware input). Matches the
    # source regime_labels_v2.parquet format (object/naive dates).
    out["date"] = out["date"].dt.tz_localize(None)

    # Write output.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[ok] wrote {len(out)} rows -> {OUT_PATH.relative_to(REPO)}")

    print("\nVol-tercile distribution (overall):")
    print(out["vol_tercile"].value_counts().to_string())

    print("\nVol-tercile × regime_label cross-tab:")
    print(pd.crosstab(out["regime_label"], out["vol_tercile"]).to_string())

    print(f"\nvol_30d stats: mean={out['vol_30d'].mean():.4f}, "
          f"min={out['vol_30d'].min():.4f}, max={out['vol_30d'].max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
