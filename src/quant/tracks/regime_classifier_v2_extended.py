"""P1 v0.6 / v2 — pre-2022 history extension pipeline.

Per server-team go-ahead (issue #20 comment ~04:32Z 2026-05-19).
Extends history backward to enable WF-OOS validation of the bear-
amplification mechanism through 2018 / 2020 COVID / 2021-2022 bears.

Data sources for v2:
  - BTC-USD, ETH-USD 1d bars 2017-08-17 → 2026-05-19 from sidecar
    `/api/v1/intraday?symbol=BTC-USD&interval_min=1440` (8.75y coverage)
  - SPY OHLCV 2015-01 → 2026-05 from yfinance
    (`data/snapshots/spy_extended.parquet`)
  - Macro (^VIX, HYG, LQD, GLD, UUP, XLU, XLE) 2015-01 → 2026-05 from yfinance
    (`data/snapshots/regime_macro_panel_extended.parquet`)

Scope reduction vs v1:
  - Alt basket reduced from 7 alts → 1 (ETH only) since pre-2021 alts
    don't exist on sidecar. `crypto_alt_to_btc_corr` becomes BTC-ETH
    correlation (still a regime signal: decoupling vs co-moving).

Pipeline:
  1. Fetch extended BTC + ETH from sidecar (`/api/v1/intraday`)
  2. Load SPY + macro from local extended snapshots
  3. Compute feature panel via existing `regime_classifier_features.py`
  4. Apply rule-based labels via existing `regime_classifier_labels.py`
  5. Walkforward-OOS train + predict via per-day loop (like v0.5)
  6. Publish `regime_labels_v2.parquet` to publish surface

Run: PYTHONPATH=src python -m quant.tracks.regime_classifier_v2_extended
"""
from __future__ import annotations

import io
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl
import requests
import xgboost as xgb


from quant.tracks.regime_classifier_features import (
    SPEC_DEFAULT, compute_regime_features,
)
from quant.tracks.regime_classifier_labels import (
    THRESHOLDS_DEFAULT, assign_regime_labels,
)
from quant.tracks.regime_classifier_train import (
    V0_4_CLASSES, XGB_PARAMS, class_weights,
)


SIDECAR_BASE_URL = os.environ.get("EUIEINVEST_API_BASE_URL", "http://100.68.86.56:8443")

SNAPSHOTS = Path("data/snapshots")
PUBLISH = Path("data/quant_publish")
MACRO_EXTENDED = SNAPSHOTS / "regime_macro_panel_extended.parquet"
SPY_EXTENDED = SNAPSHOTS / "spy_extended.parquet"
CRYPTO_EXTENDED_OUT = SNAPSHOTS / "regime_crypto_panel_extended.parquet"
FEATURES_LABELS_OUT = SNAPSHOTS / "regime_features_and_labels_v2.parquet"
WALKFORWARD_OUT = PUBLISH / "regime_labels_v2.parquet"

WARMUP_START = date(2017, 8, 17)       # earliest BTC on sidecar
PREDICTION_START = date(2018, 8, 17)   # 12mo warmup before first WF prediction
BUFFER_DAYS = 30


def fetch_intraday(symbol: str) -> pl.DataFrame:
    """Pull 1d bars via /api/v1/intraday (extended history endpoint).

    Endpoint columns: [symbol, timestamp, interval_min, open, high, low,
    close, volume]. Derives `date` from `timestamp` for downstream join
    compatibility with the v0.4 schema.
    """
    url = f"{SIDECAR_BASE_URL.rstrip('/')}/api/v1/intraday?symbol={symbol}&interval_min=1440"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pl.read_parquet(io.BytesIO(r.content))
    # Timestamp comes as ISO-8601 string ("2017-08-17T00:00:00.000Z"); parse explicitly
    df = df.filter(pl.col("symbol") == symbol).with_columns(
        date=pl.col("timestamp").str.to_datetime(time_zone="UTC").dt.date(),
    ).sort("date")
    return df


def fetch_and_save_crypto_extended() -> pl.DataFrame:
    print("  fetching BTC-USD extended (1d)...")
    btc = fetch_intraday("BTC-USD")
    print(f"    {btc.height} rows, {btc['date'].min()} → {btc['date'].max()}")
    print("  fetching ETH-USD extended (1d)...")
    eth = fetch_intraday("ETH-USD")
    print(f"    {eth.height} rows, {eth['date'].min()} → {eth['date'].max()}")
    combined = pl.concat([btc, eth], how="vertical").sort(["symbol", "date"])
    # Some intraday rows may have datetime dtype — normalize to Date
    if combined["date"].dtype != pl.Date:
        combined = combined.with_columns(pl.col("date").cast(pl.Date))
    combined.write_parquet(CRYPTO_EXTENDED_OUT)
    print(f"  wrote {CRYPTO_EXTENDED_OUT}")
    return combined


def build_v2_feature_panel() -> pl.DataFrame:
    print("\n=== Building v2 feature panel ===")
    crypto = pl.read_parquet(CRYPTO_EXTENDED_OUT)
    btc = crypto.filter(pl.col("symbol") == "BTC-USD").select(
        ["date", "open", "high", "low", "close", "volume"]
    ).sort("date")

    spy = pl.read_parquet(SPY_EXTENDED).filter(
        pl.col("symbol") == "SPY"
    ).select(["date", "open", "high", "low", "close", "volume"]).sort("date")
    print(f"  BTC: {btc.height} rows, SPY: {spy.height} rows")

    macro = pl.read_parquet(MACRO_EXTENDED)
    panels = {}
    for sym in macro["symbol"].unique().to_list():
        panels[sym] = macro.filter(pl.col("symbol") == sym).select(["date", "close"]).sort("date")
        print(f"  {sym}: {panels[sym].height} rows")

    # Alt basket = just ETH for v2 (only alt with pre-2021 data)
    alt_basket = crypto.filter(pl.col("symbol") != "BTC-USD").select(
        ["date", "symbol", "close"]
    )

    features = compute_regime_features(
        btc_ohlcv=btc, spy_ohlcv=spy,
        vix_close=panels.get("^VIX"),
        hyg_close=panels.get("HYG"),
        lqd_close=panels.get("LQD"),
        dxy_close=panels.get("UUP"),
        gld_close=panels.get("GLD"),
        alt_basket_close=alt_basket,
        spec=SPEC_DEFAULT,
    )
    print(f"  v2 feature panel: {features.height} rows")
    return features


def apply_v2_labels(features: pl.DataFrame) -> pl.DataFrame:
    print("\n=== Applying rule-based labels to v2 panel ===")
    feature_cols = [c for c in SPEC_DEFAULT.feature_columns if c in features.columns]
    complete = features.drop_nulls(subset=feature_cols)
    print(f"  rows with all features non-null: {complete.height} / {features.height}")

    labeled = assign_regime_labels(complete, THRESHOLDS_DEFAULT)
    cnts = Counter(labeled["regime_label_rule"].to_list())
    print(f"  per-regime counts:")
    for regime in sorted(cnts, key=lambda r: -cnts[r]):
        print(f"    {regime:30s} {cnts[regime]:5d}")
    labeled.write_parquet(FEATURES_LABELS_OUT)
    print(f"  wrote {FEATURES_LABELS_OUT}")
    return labeled


def walkforward_oos(labeled: pl.DataFrame) -> pl.DataFrame:
    print("\n=== Walkforward-OOS generation ===")
    feature_cols = [c for c in SPEC_DEFAULT.feature_columns if c in labeled.columns]
    full_complete = labeled.drop_nulls(subset=feature_cols).sort("date")
    pred_days = full_complete.filter(pl.col("date") >= PREDICTION_START)
    print(f"  {pred_days.height} prediction-eligible days "
          f"({pred_days['date'].min()} → {pred_days['date'].max()})")

    label_to_idx = {c: i for i, c in enumerate(V0_4_CLASSES)}
    predictions = []
    last_train_size = -1
    clf = None
    n_retrains = 0
    n_skipped = 0

    for i, row in enumerate(pred_days.iter_rows(named=True)):
        D = row["date"]
        train_end = D - timedelta(days=BUFFER_DAYS)
        train = full_complete.filter(
            (pl.col("date") <= train_end)
            & (pl.col("date") >= WARMUP_START)
            & (pl.col("regime_label_rule").is_in(V0_4_CLASSES))
        )
        if train.height != last_train_size or train.height < 30:
            if train.height < 30:
                n_skipped += 1
                continue
            X = train.select(feature_cols).to_numpy()
            y_str = train["regime_label_rule"].to_list()
            y_canon = np.array([label_to_idx[c] for c in y_str])
            # XGBoost enforces contiguous [0..N-1] class indices. Remap canonical
            # class indices (which are sparse early on) to contiguous, keeping a
            # mapping back via `xgb_to_canon` to translate predictions later.
            present_canon = sorted(set(y_canon.tolist()))
            canon_to_xgb = {c: i for i, c in enumerate(present_canon)}
            xgb_to_canon = present_canon  # list-as-mapping
            y = np.array([canon_to_xgb[c] for c in y_canon])
            sw = class_weights(y_str, V0_4_CLASSES)
            params = dict(XGB_PARAMS)
            params["num_class"] = len(present_canon)
            clf = xgb.XGBClassifier(**params)
            clf.fit(X, y, sample_weight=sw, verbose=False)
            last_train_size = train.height
            n_retrains += 1

        x = np.array([[row[c] for c in feature_cols]])
        proba_raw = clf.predict_proba(x)[0]
        # Map XGBoost's contiguous predictions back to canonical V0_4_CLASSES indices
        proba = np.zeros(len(V0_4_CLASSES))
        for xgb_idx, _ in enumerate(clf.classes_):
            canon_idx = xgb_to_canon[int(_)] if not isinstance(_, (int, np.integer)) else xgb_to_canon[int(_)]
            proba[canon_idx] = float(proba_raw[xgb_idx])
        pred_idx = int(np.argmax(proba))
        predictions.append({
            "date": D,
            "regime_label": V0_4_CLASSES[pred_idx],
            "regime_confidence": float(proba[pred_idx]),
            **{f"p_{c}": float(proba[idx]) for idx, c in enumerate(V0_4_CLASSES)},
            "rule_label": row.get("regime_label_rule"),
            "train_set_size": train.height,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{pred_days.height} ({n_retrains} retrains, {n_skipped} skipped)")

    print(f"  done: {len(predictions)} predictions, {n_retrains} retrains, "
          f"{n_skipped} skipped (early days)")
    return pl.DataFrame(predictions)


def main() -> None:
    print("=== P1 v0.6 / v2 — extended history pipeline ===\n")

    print("[1/4] fetching extended BTC + ETH from sidecar...")
    fetch_and_save_crypto_extended()

    print("\n[2/4] building feature panel + applying rule labels...")
    features = build_v2_feature_panel()
    labeled = apply_v2_labels(features)

    print("\n[3/4] walkforward-OOS regeneration...")
    wf = walkforward_oos(labeled)
    print(f"  predictions: {wf.height} rows")

    # Distribution
    print("\n  regime distribution:")
    for r in wf.group_by("regime_label").len().sort("len", descending=True).iter_rows(named=True):
        print(f"    {r['regime_label']:25s} {r['len']:5d}")

    # Per-year breakdown (especially the bear coverage)
    wf_with_year = wf.with_columns(year=pl.col("date").dt.year())
    print("\n  per-year regime distribution (looking for bear coverage):")
    for year_row in wf_with_year.group_by(["year", "regime_label"]).len().sort(["year", "regime_label"]).iter_rows(named=True):
        if year_row["year"] in (2018, 2020, 2021, 2022, 2023, 2024, 2025, 2026):
            print(f"    {year_row['year']} {year_row['regime_label']:25s} {year_row['len']:5d}")

    print("\n[4/4] publishing...")
    PUBLISH.mkdir(parents=True, exist_ok=True)
    wf.write_parquet(WALKFORWARD_OUT)
    print(f"  wrote {WALKFORWARD_OUT} ({wf.height} rows)")


if __name__ == "__main__":
    main()
