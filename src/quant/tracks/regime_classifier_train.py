"""Regime classifier Day 4-5 — XGBoost training + walkforward.

Per issue #20 spec + server-team v0.4 relaxed-gate path (PR #1 issue#20
comment 4478870555 area, 2026-05-19 00:19Z): ship 4-class XGBoost over
the dominant regimes (bear_trend, choppy_recovery, sideways_range,
steady_bull) to unblock the Stream 2 regime-gating experiment, with
relaxed gate macro-F1 ≥ 0.55 (vs spec's 0.65) and per-class precision
≥ 0.40 (vs spec's 0.50). Skip crash_shock (3 days) and
crypto_decoupled_bull (20 days) and high_correlation_risk_off (24 days)
as sparse classes — extend in v0.5 once we have more years.

Walkforward methodology (issue #20 spec):
  - 5y train / 6mo validate / slide forward 3mo
  - Per-fold OOS macro-F1 + per-class precision/recall
  - Class-weighted loss to handle imbalance

Output:
  - data/snapshots/regime_walkforward_results.json (per-fold metrics)
  - data/snapshots/regime_model_v0_4.json (final model trained on full data)
  - data/quant_publish/regime_labels_v1.parquet (published predictions
    for all 1057 days)

Run: PYTHONPATH=src python -m quant.tracks.regime_classifier_train
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support


SNAPSHOTS = Path("data/snapshots")
PUBLISH = Path("data/quant_publish")
INPUT_PARQUET = SNAPSHOTS / "regime_features_and_labels.parquet"

# Per server-team 00:19Z message: ship v0.4 with 4 dominant classes
V0_4_CLASSES = ["bear_trend", "choppy_recovery", "sideways_range", "steady_bull"]

# Walkforward params per issue #20 spec
TRAIN_WINDOW_DAYS = 365 * 5      # 5y
VAL_WINDOW_DAYS = 30 * 6         # ~6mo
SLIDE_DAYS = 30 * 3              # 3mo

# Acceptance gates per server-team relaxed v0.4 path
GATE_MACRO_F1 = 0.55
GATE_PER_CLASS_PRECISION = 0.40
GATE_FOLD_STABILITY = 0.15       # max delta in macro-F1 across folds

# XGBoost hyperparams — conservative for small data
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=len(V0_4_CLASSES),
    max_depth=4,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
)


def load_labeled() -> tuple[pl.DataFrame, list[str]]:
    """Load the labeled feature panel + return feature column list."""
    df = pl.read_parquet(INPUT_PARQUET)
    # Restrict to v0.4 classes (drop unlabeled + sparse classes)
    df_v04 = df.filter(pl.col("regime_label_rule").is_in(V0_4_CLASSES))
    feature_cols = [c for c in df.columns if c not in (
        "date", "regime_label_rule", "regime_label_rule_was_unlabeled"
    )]
    return df_v04, feature_cols


def make_walkforward_folds(
    df: pl.DataFrame,
    train_days: int = TRAIN_WINDOW_DAYS,
    val_days: int = VAL_WINDOW_DAYS,
    slide_days: int = SLIDE_DAYS,
) -> list[tuple[date, date, date, date]]:
    """Build (train_start, train_end, val_start, val_end) tuples.

    On a 4-year labeled span the canonical 5y/6mo/3mo schedule produces
    only 1-2 folds. For our 2022-2026 window we use expanding-train +
    rolling-validation: train start fixed at first label date, train end
    advances, val window slides forward.
    """
    label_dates = sorted(df["date"].to_list())
    if not label_dates:
        return []
    first = label_dates[0]
    last = label_dates[-1]

    folds = []
    # First fold: train on first 12mo, validate on next 6mo
    initial_train_end = first + timedelta(days=365)
    val_start = initial_train_end + timedelta(days=1)
    while val_start + timedelta(days=val_days) <= last:
        val_end = val_start + timedelta(days=val_days)
        folds.append((first, val_start - timedelta(days=1), val_start, val_end))
        val_start = val_start + timedelta(days=slide_days)
    return folds


def class_weights(y: np.ndarray, classes: list[str]) -> np.ndarray:
    """Inverse-frequency weight per sample."""
    counts = Counter(y)
    n = len(y)
    weights = np.array([n / (len(classes) * counts.get(c, 1)) for c in y])
    return weights


def train_fold(
    df: pl.DataFrame, feature_cols: list[str], classes: list[str],
    train_start: date, train_end: date, val_start: date, val_end: date,
) -> dict:
    """Train + evaluate one fold. Returns metrics dict."""
    label_to_idx = {c: i for i, c in enumerate(classes)}

    train = df.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
    val = df.filter((pl.col("date") >= val_start) & (pl.col("date") <= val_end))

    X_train = train.select(feature_cols).to_numpy()
    y_train_str = train["regime_label_rule"].to_list()
    y_train = np.array([label_to_idx[c] for c in y_train_str])
    sw = class_weights(y_train_str, classes)

    X_val = val.select(feature_cols).to_numpy()
    y_val_str = val["regime_label_rule"].to_list()
    y_val = np.array([label_to_idx[c] for c in y_val_str])

    if len(y_val) == 0 or len(set(y_train)) < 2:
        return {"skip": True, "n_train": len(y_train), "n_val": len(y_val)}

    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sw, verbose=False)

    y_pred = clf.predict(X_val)
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    p, r, f, supp = precision_recall_fscore_support(
        y_val, y_pred, labels=range(len(classes)), zero_division=0,
    )
    per_class = {classes[i]: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(supp[i])}
                 for i in range(len(classes))}

    return {
        "skip": False,
        "train_window": (str(train_start), str(train_end)),
        "val_window": (str(val_start), str(val_end)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def train_final_and_score_all(
    labeled: pl.DataFrame, full_panel: pl.DataFrame,
    feature_cols: list[str], classes: list[str],
) -> tuple[xgb.XGBClassifier, pl.DataFrame]:
    """Train on all labeled data; predict on the FULL panel (incl. unlabeled)."""
    label_to_idx = {c: i for i, c in enumerate(classes)}
    X_train = labeled.select(feature_cols).to_numpy()
    y_train_str = labeled["regime_label_rule"].to_list()
    y_train = np.array([label_to_idx[c] for c in y_train_str])
    sw = class_weights(y_train_str, classes)

    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sw, verbose=False)

    # Score the full panel (incl. unlabeled days)
    X_all = full_panel.select(feature_cols).to_numpy()
    proba = clf.predict_proba(X_all)
    pred_idx = proba.argmax(axis=1)

    # Build output dataframe
    out_cols = {
        "date": full_panel["date"],
        "regime_label": pl.Series([classes[i] for i in pred_idx]),
        "regime_confidence": pl.Series(proba.max(axis=1).astype(float)),
    }
    for i, c in enumerate(classes):
        out_cols[f"p_{c}"] = pl.Series(proba[:, i].astype(float))
    # Pass-through: was this day in the rule-based labeled set?
    out_cols["rule_label"] = full_panel["regime_label_rule"]

    return clf, pl.DataFrame(out_cols)


def main() -> None:
    print("=== loading labeled data (v0.4: 4 dominant classes) ===")
    labeled_v04, feature_cols = load_labeled()
    full_panel = pl.read_parquet(INPUT_PARQUET)
    print(f"  labeled v0.4: {labeled_v04.height} rows ({len(V0_4_CLASSES)} classes)")
    print(f"  features: {len(feature_cols)}")
    print(f"  full panel (for scoring): {full_panel.height} rows")
    cls_counts = Counter(labeled_v04["regime_label_rule"].to_list())
    print(f"  per-class counts: {dict(cls_counts)}")

    print("\n=== walkforward folds ===")
    folds = make_walkforward_folds(labeled_v04)
    print(f"  {len(folds)} folds")
    for i, (ts, te, vs, ve) in enumerate(folds):
        print(f"    fold {i}: train {ts}..{te} | val {vs}..{ve}")

    print("\n=== training per fold ===")
    fold_results = []
    for i, (ts, te, vs, ve) in enumerate(folds):
        r = train_fold(labeled_v04, feature_cols, V0_4_CLASSES, ts, te, vs, ve)
        r["fold"] = i
        fold_results.append(r)
        if r.get("skip"):
            print(f"  fold {i}: SKIP (n_train={r['n_train']}, n_val={r['n_val']})")
        else:
            print(f"  fold {i}: macro-F1={r['macro_f1']:.3f} "
                  f"weighted-F1={r['weighted_f1']:.3f} "
                  f"n_train={r['n_train']} n_val={r['n_val']}")

    print("\n=== aggregate acceptance check (v0.4 relaxed gate) ===")
    valid = [r for r in fold_results if not r.get("skip")]
    if not valid:
        print("  [FAIL] no valid folds")
        return

    macro_f1s = [r["macro_f1"] for r in valid]
    mean_macro_f1 = float(np.mean(macro_f1s))
    f1_range = float(max(macro_f1s) - min(macro_f1s))
    print(f"  mean macro-F1 across folds: {mean_macro_f1:.3f}")
    print(f"  fold range:                 {f1_range:.3f}")
    print(f"  gate macro-F1 >= {GATE_MACRO_F1}:           "
          f"{'[PASS]' if mean_macro_f1 >= GATE_MACRO_F1 else '[FAIL]'}")
    print(f"  gate fold-stability <= {GATE_FOLD_STABILITY}:   "
          f"{'[PASS]' if f1_range <= GATE_FOLD_STABILITY else '[FAIL]'}")

    # Per-class precision check (aggregate across all valid folds)
    pcp = {c: [] for c in V0_4_CLASSES}
    for r in valid:
        for c, m in r["per_class"].items():
            if m["support"] > 0:
                pcp[c].append(m["precision"])
    mean_per_class_precision = {c: (float(np.mean(v)) if v else 0.0)
                                 for c, v in pcp.items()}
    print(f"  mean per-class precision: {mean_per_class_precision}")
    failing_classes = [c for c, p in mean_per_class_precision.items()
                        if p < GATE_PER_CLASS_PRECISION]
    print(f"  gate per-class >= {GATE_PER_CLASS_PRECISION}:        "
          f"{'[PASS]' if not failing_classes else f'[FAIL: {failing_classes}]'}")

    overall_pass = (mean_macro_f1 >= GATE_MACRO_F1
                    and f1_range <= GATE_FOLD_STABILITY
                    and not failing_classes)
    print(f"\n  v0.4 OVERALL: {'[PASS]' if overall_pass else '[FAIL]'}")

    # Persist walkforward results
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    wf_out = SNAPSHOTS / "regime_walkforward_results.json"
    with open(wf_out, "w") as f:
        json.dump({
            "spec_version": "v0.4",
            "classes": V0_4_CLASSES,
            "gate": {"macro_f1": GATE_MACRO_F1,
                     "per_class_precision": GATE_PER_CLASS_PRECISION,
                     "fold_stability": GATE_FOLD_STABILITY},
            "mean_macro_f1": mean_macro_f1,
            "fold_range": f1_range,
            "mean_per_class_precision": mean_per_class_precision,
            "overall_pass": overall_pass,
            "per_fold": fold_results,
        }, f, indent=2)
    print(f"\n  wrote {wf_out}")

    print("\n=== training final model on full labeled data + scoring all 1057 days ===")
    clf_final, scored = train_final_and_score_all(
        labeled_v04, full_panel, feature_cols, V0_4_CLASSES,
    )

    # SHAP-equivalent: XGBoost feature importance (gain)
    importance = clf_final.get_booster().get_score(importance_type="gain")
    # Map xgb f0/f1/.. back to actual feature names
    imp_named = {}
    for k, v in importance.items():
        idx = int(k[1:])
        imp_named[feature_cols[idx]] = float(v)
    top_feats = sorted(imp_named.items(), key=lambda x: -x[1])[:5]
    print(f"  top-5 features by gain:")
    for name, gain in top_feats:
        print(f"    {name:40s}  gain={gain:.2f}")

    # Save model
    model_path = SNAPSHOTS / "regime_model_v0_4.json"
    clf_final.save_model(str(model_path))
    print(f"  wrote {model_path}")

    # Publish parquet
    PUBLISH.mkdir(parents=True, exist_ok=True)
    publish_path = PUBLISH / "regime_labels_v1.parquet"
    scored.write_parquet(publish_path)
    print(f"\n  wrote {publish_path} ({scored.height} rows)")

    # Sample of publish output
    print(f"\n  publish parquet head:")
    print(scored.head(5))


if __name__ == "__main__":
    main()
