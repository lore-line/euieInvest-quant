"""P1 v0.5 — walkforward-OOS label regeneration.

Per server-team causality concern (issue #20 comment 4484191299): the
published `regime_labels_v1.parquet` predictions are in-sample-foresight
(final model trained on full 2022-2026 history, then scored back across
all 1057 days — model "knew" test-period labels during training).

This module regenerates labels via TRUE walkforward-OOS process:
  - For each day D from 2023-04-01 (12mo warmup) to last labeled day
  - Train XGBoost on rule-labels for days 2022-03-24 → (D - 30)
    - 30-day buffer prevents information bleed via 30d rolling features
  - Predict D's regime + confidence
  - Loop ~1100 times

Optimization: re-train ONLY when a new labeled day enters the training
set (since most days don't have rule labels, training set grows slowly).
Cuts wall-clock by ~10×.

Output: `data/quant_publish/regime_labels_v1_walkforward.parquet`
Schema matches v0.4 publish exactly so the multi-strategy harness can
swap by changing the input filename.

Run: PYTHONPATH=src python -m quant.tracks.regime_classifier_walkforward
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl
import xgboost as xgb


from quant.tracks.regime_classifier_train import (
    V0_4_CLASSES, XGB_PARAMS, class_weights,
)


SNAPSHOTS = Path("data/snapshots")
PUBLISH = Path("data/quant_publish")
INPUT_PARQUET = SNAPSHOTS / "regime_features_and_labels.parquet"
WALKFORWARD_OUT = PUBLISH / "regime_labels_v1_walkforward.parquet"

WARMUP_START = date(2022, 3, 24)
PREDICTION_START = date(2023, 4, 1)    # 12mo warmup before first prediction
BUFFER_DAYS = 30                       # avoid info bleed via 30d rolling features


def main() -> None:
    print("=== P1 v0.5 walkforward-OOS label regeneration ===\n")

    print("[1/4] loading features + rule-labels...")
    full = pl.read_parquet(INPUT_PARQUET).sort("date")
    feature_cols = [c for c in full.columns if c not in (
        "date", "regime_label_rule", "regime_label_rule_was_unlabeled"
    )]
    print(f"      {full.height} day-rows, {len(feature_cols)} features")
    print(f"      date span: {full['date'].min()} → {full['date'].max()}")

    # Filter to days with full features (post-warmup)
    full_complete = full.drop_nulls(subset=feature_cols)
    print(f"      {full_complete.height} days with all features")

    label_to_idx = {c: i for i, c in enumerate(V0_4_CLASSES)}

    # Prediction-eligible days: post-PREDICTION_START
    pred_days = full_complete.filter(pl.col("date") >= PREDICTION_START).sort("date")
    print(f"      {pred_days.height} prediction-eligible days "
          f"({pred_days['date'].min()} → {pred_days['date'].max()})")

    print("\n[2/4] walkforward loop with re-training only on training-set growth...")

    predictions = []
    last_train_set_size = -1
    last_train_end_date = None
    clf = None
    n_retrains = 0
    n_skipped = 0

    for i, row in enumerate(pred_days.iter_rows(named=True)):
        D = row["date"]
        # Training set: rule-labeled v0.4 classes up to D - BUFFER_DAYS
        train_end = D - timedelta(days=BUFFER_DAYS)
        train_mask = (
            (pl.col("date") <= train_end)
            & (pl.col("date") >= WARMUP_START)
            & (pl.col("regime_label_rule").is_in(V0_4_CLASSES))
        )
        train = full_complete.filter(train_mask)

        # Re-train only if training set grew (i.e., a new labeled day entered)
        if train.height != last_train_set_size or train.height < 30:
            if train.height < 30:
                n_skipped += 1
                continue  # not enough data yet
            X_train = train.select(feature_cols).to_numpy()
            y_train_str = train["regime_label_rule"].to_list()
            y_train = np.array([label_to_idx[c] for c in y_train_str])
            sw = class_weights(y_train_str, V0_4_CLASSES)
            # Override num_class to actually-seen-classes-in-training count
            # (XGBoost errors if num_class > distinct labels in y)
            params = dict(XGB_PARAMS)
            params["num_class"] = len(set(y_train))
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train, sample_weight=sw, verbose=False)
            last_train_set_size = train.height
            last_train_end_date = train_end
            n_retrains += 1

        # Predict D — handle dynamic class count (early training sets
        # may not have all 4 V0_4_CLASSES present)
        X = np.array([[row[c] for c in feature_cols]])
        proba_raw = clf.predict_proba(X)[0]
        # Map XGBoost's internal class indices (0..N-1 where N=n_classes_seen)
        # back to the canonical V0_4_CLASSES indices via clf.classes_
        proba = np.zeros(len(V0_4_CLASSES))
        for xgb_idx, canonical_idx in enumerate(clf.classes_):
            proba[int(canonical_idx)] = float(proba_raw[xgb_idx])
        pred_idx = int(np.argmax(proba))
        predictions.append({
            "date": D,
            "regime_label": V0_4_CLASSES[pred_idx],
            "regime_confidence": float(proba[pred_idx]),
            **{f"p_{c}": float(proba[idx]) for idx, c in enumerate(V0_4_CLASSES)},
            "rule_label": row.get("regime_label_rule"),
            "train_set_size": train.height,
            "train_end_date": last_train_end_date,
        })

        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{pred_days.height} days predicted "
                  f"({n_retrains} retrains, {n_skipped} skipped)")

    print(f"      done: {len(predictions)} predictions, "
          f"{n_retrains} retrains, {n_skipped} skipped (early days)")

    print("\n[3/4] building output dataframe...")
    out = pl.DataFrame(predictions)
    print(f"      {out.height} rows × {len(out.columns)} cols")

    # Distribution
    print("\n      regime distribution:")
    dist = out.group_by("regime_label").len().sort("len", descending=True)
    for r in dist.iter_rows(named=True):
        print(f"        {r['regime_label']:25s} {r['len']:5d}")

    # Compare to v0.4 in-sample on overlapping days
    print("\n      comparison to v0.4 in-sample predictions:")
    v04 = pl.read_parquet(PUBLISH / "regime_labels_v1.parquet").select([
        "date", pl.col("regime_label").alias("v04_label"),
        pl.col("regime_confidence").alias("v04_conf"),
    ])
    cmp = out.select(["date", "regime_label", "regime_confidence"]).join(v04, on="date", how="inner")
    agree = cmp.filter(pl.col("regime_label") == pl.col("v04_label")).height
    print(f"        overlapping days: {cmp.height}")
    print(f"        wf-OOS agrees with in-sample: {agree} ({agree/cmp.height*100:.1f}%)")
    disagree = cmp.filter(pl.col("regime_label") != pl.col("v04_label"))
    if disagree.height > 0:
        print(f"        disagreement breakdown (wf-OOS prediction vs in-sample):")
        for r in disagree.group_by(["regime_label", "v04_label"]).len().sort("len", descending=True).iter_rows(named=True):
            print(f"          {r['regime_label']:20s} → {r['v04_label']:20s}: {r['len']}")

    print("\n[4/4] publishing...")
    PUBLISH.mkdir(parents=True, exist_ok=True)
    out.write_parquet(WALKFORWARD_OUT)
    print(f"      wrote {WALKFORWARD_OUT} ({out.height} rows)")


if __name__ == "__main__":
    main()
