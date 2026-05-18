"""Day 2 smoke test: feature pipeline + rule-based labels on the joined panel.

Verifies:
  - All 14 features compute without error on 5y panel
  - Coverage rate of rule-based labels is in target 50-70% band
  - Per-regime counts are reasonable (no single regime dominates >80%)
  - No catastrophic null propagation

Run: PYTHONPATH=src python -m quant.tracks.regime_classifier_smoke
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from quant.tracks.regime_classifier_features import (
    SPEC_DEFAULT, compute_regime_features,
)
from quant.tracks.regime_classifier_labels import (
    THRESHOLDS_DEFAULT, assign_regime_labels, label_statistics,
)


SNAPSHOTS = Path("data/snapshots")


def load_panels() -> dict[str, pl.DataFrame]:
    """Load all input panels for the feature pipeline."""
    out = {}

    # macro yfinance panel (long format)
    macro = pl.read_parquet(SNAPSHOTS / "regime_macro_panel.parquet")
    for sym in macro["symbol"].unique().to_list():
        out[sym] = macro.filter(pl.col("symbol") == sym).select(["date", "close"])

    # crypto sidecar panel
    crypto = pl.read_parquet(SNAPSHOTS / "regime_crypto_panel.parquet")
    out["BTC-USD"] = crypto.filter(pl.col("symbol") == "BTC-USD").select([
        "date", "open", "high", "low", "close", "volume"
    ])

    # alt basket: long-format, all non-BTC crypto
    out["_alt_basket"] = crypto.filter(pl.col("symbol") != "BTC-USD").select([
        "date", "symbol", "close"
    ])

    # SPY OHLCV from local snapshot
    spy = pl.read_parquet(SNAPSHOTS / "ohlcv.parquet").filter(pl.col("symbol") == "SPY")
    out["SPY"] = spy.select(["date", "open", "high", "low", "close", "volume"])

    return out


def main() -> None:
    print("=== Loading panels ===")
    panels = load_panels()
    for name, df in panels.items():
        print(f"  {name}: {df.height} rows")

    print("\n=== Computing 14-feature regime panel ===")
    features = compute_regime_features(
        btc_ohlcv=panels["BTC-USD"],
        spy_ohlcv=panels["SPY"],
        vix_close=panels.get("^VIX"),
        hyg_close=panels.get("HYG"),
        lqd_close=panels.get("LQD"),
        dxy_close=panels.get("UUP"),       # UUP as DXY proxy
        gld_close=panels.get("GLD"),
        alt_basket_close=panels.get("_alt_basket"),
        spec=SPEC_DEFAULT,
    )
    print(f"  feature panel: {features.height} rows, {len(features.columns)} cols")
    print(f"  date range: {features['date'].min()} -> {features['date'].max()}")
    print(f"  columns: {features.columns}")

    # Null-rate check per feature
    print("\n=== Per-feature null rates ===")
    for col in SPEC_DEFAULT.feature_columns:
        if col in features.columns:
            null_rate = features[col].null_count() / features.height
            print(f"  {col:40s}: {null_rate*100:5.1f}% null")

    # Drop rows missing any feature for label computation
    feature_cols = [c for c in SPEC_DEFAULT.feature_columns if c in features.columns]
    complete = features.drop_nulls(subset=feature_cols)
    print(f"\n  rows with all features non-null: {complete.height} / {features.height}")

    print("\n=== Applying rule-based labels ===")
    labeled = assign_regime_labels(complete, THRESHOLDS_DEFAULT)
    stats = label_statistics(labeled)
    print(f"  total days: {stats['n_total_days']}")
    print(f"  labeled days: {stats['n_labeled_days']} ({stats['coverage_rate']*100:.1f}%)")
    print(f"  date range: {stats['date_range']}")
    print("\n  per-regime counts:")
    for regime, count in stats["per_regime_counts"].items():
        pct = count / stats["n_total_days"] * 100
        print(f"    {regime:35s}: {count:5d} ({pct:5.1f}%)")

    # Acceptance check (ASCII-only for Windows cp1252 console compat)
    print("\n=== Day 2 acceptance check ===")
    coverage = stats["coverage_rate"]
    if 0.50 <= coverage <= 0.70:
        print(f"  [PASS] Coverage {coverage*100:.1f}% in target 50-70% band")
    elif coverage < 0.50:
        print(f"  [FAIL] Coverage {coverage*100:.1f}% BELOW 50% - rules too strict, need to loosen thresholds")
    else:
        print(f"  [WARN] Coverage {coverage*100:.1f}% ABOVE 70% - rules too loose, may want to tighten")

    # Check no single regime dominates
    if stats["n_labeled_days"] > 0:
        max_regime, max_count = max(
            ((k, v) for k, v in stats["per_regime_counts"].items() if k != "unlabeled"),
            key=lambda kv: kv[1],
        )
        max_pct = max_count / stats["n_labeled_days"] * 100
        if max_pct < 80:
            print(f"  [PASS] No single regime dominates ({max_regime}: {max_pct:.1f}% of labeled)")
        else:
            print(f"  [WARN] {max_regime} dominates at {max_pct:.1f}% - rules need rebalancing")

    # Save intermediate artifact for Day 3 training input
    out_path = SNAPSHOTS / "regime_features_and_labels.parquet"
    labeled.write_parquet(out_path)
    print(f"\n  wrote {out_path} ({labeled.height} rows)")


if __name__ == "__main__":
    main()
