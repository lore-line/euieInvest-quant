"""Sustained-winner XGB training + rule extraction — Workstream C step 2.

For one gain-threshold spec, trains an XGB binary classifier on the
sustained-winner label, extracts rules from the tree paths, and writes
the artifacts to `runs/{date}-sustained_winner_v1_{spec_name}/`.

Designed to be called per-g in the Pareto-frontier sweep:

  for g in sweep_specs():
      python -m quant.tracks.sustained_winner_train --spec g20
      python -m quant.tracks.sustained_winner_train --spec g19
      ...

Per-spec wall clock budget: ~30-60s training + ~30s rule extraction on
the existing 2.4M-row data scale (CPU XGB hist tree). Total sweep
(20 specs) ≈ 20-30 min wall clock.

Outputs per spec:
- xgb_model.json — the trained classifier
- rules.parquet — extracted rules (rule_id, conditions_json, lift, etc.)
  in the SAME schema as step3a_xgb_rule_extraction so downstream
  walkforward_validate consumes it unchanged
- manifest.json — config + training metrics

Walk-forward validation runs SEPARATELY via the existing
`walkforward_validate.py` driver (no change needed; just point it at
the new rules.parquet).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from quant.tracks.sustained_winner_label import (
    SPECS,
    SustainedWinnerSpec,
    compute_sustained_winner_label,
    label_statistics,
    sweep_specs,
)
from quant.tracks.xgb_rule_extraction import (
    _NON_FEATURE_COLS,
    Rule,
    _evaluate_rules,
    extract_paths,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "sustained_winner_discovery_v1"
DEFAULT_TRAIN_CUTOFF = date(2024, 12, 31)
DEFAULT_N_ROUNDS = 400
DEFAULT_MIN_TRAIN_COHORT = 5_000  # per server-team spec
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,  # rule-extraction prefers fewer, more reliable splits
    "verbosity": 0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features", type=Path,
        default=Path("data/features/features.parquet"),
    )
    p.add_argument(
        "--spec", type=str, default="g20",
        help="Sweep spec name (e.g. 'g20', 'g15', 'g10') or 'standard'/'strict' "
             "for the canonical-named variants.",
    )
    p.add_argument(
        "--train-cutoff", type=date.fromisoformat, default=DEFAULT_TRAIN_CUTOFF,
    )
    p.add_argument(
        "--n-rounds", type=int, default=DEFAULT_N_ROUNDS,
    )
    p.add_argument(
        "--min-train-cohort", type=int, default=DEFAULT_MIN_TRAIN_COHORT,
        help="Refuse to train if positive examples < this on the train split.",
    )
    p.add_argument(
        "--out-root", type=Path, default=None,
        help="Output dir root. Default: runs/{today}-sustained_winner_v1_sweep/{spec.label_column()}/",
    )
    p.add_argument(
        "--rule-min-lift", type=float, default=1.2,
        help="Drop extracted rules with lift below this threshold.",
    )
    p.add_argument(
        "--rule-min-coverage-pct", type=float, default=0.1,
        help="Drop extracted rules with coverage below this %% of train.",
    )
    p.add_argument(
        "--rule-min-precision", type=float, default=0.20,
        help="Drop extracted rules with precision below this.",
    )
    return p.parse_args(argv)


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_spec(name: str) -> SustainedWinnerSpec:
    if name in SPECS:
        return SPECS[name]
    # Look in the sweep
    for s in sweep_specs():
        if s.name == name:
            return s
    raise ValueError(
        f"unknown spec '{name}' — must be one of {list(SPECS.keys())} "
        f"or a sweep name like 'g01'..'g20'"
    )


def _prepare_training_matrix(
    labeled: pl.DataFrame, spec: SustainedWinnerSpec, train_cutoff: date
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Filter to labelable rows, split chronologically, return (train, val, feature_cols).

    Mirrors xgb_rule_extraction's _replay_feature_selection but uses the
    sustained-winner label column instead of is_winner."""
    label_col = spec.label_column()
    # Drop bookkeeping columns from the labeling pipeline that aren't features
    drop_for_features = _NON_FEATURE_COLS | {
        "forward_max_pct", "forward_endpoint_pct", label_col,
    }
    labelable = labeled.filter(pl.col(label_col).is_not_null())
    feature_cols = [c for c in labelable.columns if c not in drop_for_features]
    bool_cols = [c for c in feature_cols if labelable[c].dtype == pl.Boolean]
    if bool_cols:
        labelable = labelable.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])
    str_cols = [c for c in feature_cols if labelable[c].dtype == pl.Utf8]
    if str_cols:
        labelable = labelable.to_dummies(columns=str_cols)
    feature_cols = [c for c in labelable.columns if c not in drop_for_features]
    train_df = labelable.filter(pl.col("date") <= train_cutoff)
    val_df = labelable.filter(pl.col("date") > train_cutoff)
    return train_df, val_df, feature_cols


def _train_xgb(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    n_rounds: int,
) -> tuple[xgb.Booster, dict]:
    """Train XGB binary classifier, return (booster, training_metrics)."""
    X_train = train_df.select(feature_cols).to_numpy(allow_copy=True)
    y_train = train_df[label_col].cast(pl.Int8).to_numpy()
    X_val = val_df.select(feature_cols).to_numpy(allow_copy=True)
    y_val = val_df[label_col].cast(pl.Int8).to_numpy()
    # XGB rejects inf; map to nan and let XGB handle missing
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_val = np.nan_to_num(X_val, nan=np.nan, posinf=np.nan, neginf=np.nan)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, missing=np.nan)
    evals_result: dict = {}
    booster = xgb.train(
        DEFAULT_XGB_PARAMS,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )
    from sklearn.metrics import roc_auc_score
    train_auc = float(roc_auc_score(y_train, booster.predict(dtrain)))
    val_auc = float(roc_auc_score(y_val, booster.predict(dval)))
    return booster, {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "n_rounds": n_rounds,
        "n_train_rows": int(train_df.height),
        "n_train_positives": int(train_df.filter(pl.col(label_col) == True).height),
        "n_val_rows": int(val_df.height),
        "n_val_positives": int(val_df.filter(pl.col(label_col) == True).height),
        "n_features": len(feature_cols),
    }


def _extract_and_filter_rules(
    booster: xgb.Booster,
    feature_cols: list[str],
    eval_df: pl.DataFrame,
    label_col: str,
    min_lift: float,
    min_coverage_pct: float,
    min_precision: float,
) -> pl.DataFrame:
    """Walk the booster's trees → unique rules → evaluate on eval_df → filter.

    Mirrors xgb_rule_extraction's main() pipeline:
      1. extract_paths() returns list[tuple[Condition, ...]]
      2. Dedup paths via Rule.from_path() into a {Rule: id} dict
      3. _evaluate_rules() computes coverage/precision/lift per Rule
      4. Filter by lift/coverage/precision; sort by lift × coverage
    """
    # XGB's get_dump emits feature names that match what we set on the booster
    booster.feature_names = feature_cols
    all_paths = extract_paths(booster)
    print(f"    {len(all_paths):,} root-to-leaf paths")
    # Dedup paths → unique rules
    rule_set: dict[Rule, int] = {}
    for path in all_paths:
        r = Rule.from_path(path)
        rule_set.setdefault(r, len(rule_set))
    rules = list(rule_set.keys())
    print(f"    {len(rules):,} unique rules after dedup")
    # Evaluate on eval_df using the cached-mask code path
    label_arr = eval_df[label_col].cast(pl.Boolean).to_numpy()
    t_eval = time.perf_counter()
    rule_records = _evaluate_rules(rules, eval_df, label_arr)
    print(f"    evaluated {len(rule_records):,} rules with non-zero coverage in {time.perf_counter() - t_eval:.1f}s")
    # Filter + sort by lift × coverage_pct (same as step3a's "the brief's ranking")
    kept = [
        r for r in rule_records
        if r["lift"] >= min_lift
        and r["coverage_pct"] >= min_coverage_pct
        and r["precision"] >= min_precision
    ]
    kept.sort(key=lambda r: r["lift"] * r["coverage_pct"], reverse=True)
    # Reassign rule_id post-sort so parquet rule_id is rank order
    for new_id, r in enumerate(kept):
        r["rule_id"] = new_id
    if not kept:
        return pl.DataFrame(schema={
            "rule_id": pl.Int64, "conditions_json": pl.String, "n_conditions": pl.Int64,
            "coverage_n": pl.Int64, "coverage_pct": pl.Float64, "precision": pl.Float64,
            "lift": pl.Float64, "example_symbol_dates_json": pl.String,
        })
    return pl.DataFrame(kept)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    spec = _resolve_spec(args.spec)
    print(f"sustained_winner_train — {spec.label_column()}")
    print(f"  touch_threshold={spec.touch_threshold_pct}%  "
          f"endpoint_threshold={spec.endpoint_threshold_pct}%  "
          f"horizon={spec.horizon_days}td")

    features_path = _resolve_path(args.features)
    print(f"  features: {features_path.relative_to(_REPO_ROOT)}")
    features = pl.read_parquet(features_path)
    print(f"  loaded {features.height:,} rows × {features.width} cols")

    # Compute label
    t_label = time.perf_counter()
    labeled = compute_sustained_winner_label(features, spec)
    stats = label_statistics(labeled, spec)
    print(f"  labelable: {stats['n_labelable_rows']:,}, "
          f"winners: {stats['n_winners']:,} ({stats['winner_rate']*100:.2f}%)  "
          f"({time.perf_counter() - t_label:.1f}s)")

    # Resolve output dir
    if args.out_root is not None:
        out_root = _resolve_path(args.out_root)
    else:
        out_root = _REPO_ROOT / f"runs/{date.today().isoformat()}-sustained_winner_v1_sweep"
    out_dir = out_root / spec.label_column()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir: {out_dir.relative_to(_REPO_ROOT)}")

    # Prepare training matrix
    train_df, val_df, feature_cols = _prepare_training_matrix(
        labeled, spec, args.train_cutoff
    )
    n_train_pos = int(train_df.filter(pl.col(spec.label_column()) == True).height)
    print(f"  train: {train_df.height:,} rows ({n_train_pos:,} positive) | val: {val_df.height:,}")
    if n_train_pos < args.min_train_cohort:
        msg = f"train cohort {n_train_pos:,} < MIN_TRAIN_COHORT={args.min_train_cohort:,} — SKIP {spec.label_column()}"
        print(f"  {msg}")
        # Still write a manifest documenting the skip
        manifest = {
            "spec": dataclasses.asdict(spec),
            "pipeline_step": PIPELINE_STEP,
            "skipped": True,
            "skip_reason": msg,
            "n_train_positives": n_train_pos,
            "wall_clock_s": round(time.perf_counter() - t0, 3),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return 0

    # Train XGB
    print(f"  training XGB ({args.n_rounds} rounds) ...")
    t_train = time.perf_counter()
    booster, training_metrics = _train_xgb(
        train_df, val_df, feature_cols, spec.label_column(), args.n_rounds
    )
    print(f"  train_auc={training_metrics['train_auc']:.4f}  "
          f"val_auc={training_metrics['val_auc']:.4f}  "
          f"({time.perf_counter() - t_train:.1f}s)")
    booster.save_model(str(out_dir / "xgb_model.json"))

    # Extract + filter rules. Evaluate on val_df (post-train-cutoff) per
    # Phase B v3 convention — picks rules with out-of-train lift, and the
    # subsequent walk-forward validation re-evaluates on 11 separate windows.
    # val_df is smaller (~640K) than train (~1.7M) so the cached-mask memory
    # footprint is manageable.
    print(f"  extracting + filtering rules on val ({val_df.height:,} rows) ...")
    t_rules = time.perf_counter()
    rules_df = _extract_and_filter_rules(
        booster, feature_cols, val_df, spec.label_column(),
        args.rule_min_lift, args.rule_min_coverage_pct, args.rule_min_precision,
    )
    print(f"  rules: {rules_df.height:,} surviving filter  "
          f"(min_lift={args.rule_min_lift}, min_cov_pct={args.rule_min_coverage_pct}, "
          f"min_precision={args.rule_min_precision})  ({time.perf_counter() - t_rules:.1f}s)")
    rules_df.write_parquet(out_dir / "rules.parquet")

    # Manifest
    manifest = {
        "spec": dataclasses.asdict(spec),
        "pipeline_step": PIPELINE_STEP,
        "skipped": False,
        "label_statistics": stats,
        "training": training_metrics,
        "rule_filter_thresholds": {
            "min_lift": args.rule_min_lift,
            "min_coverage_pct": args.rule_min_coverage_pct,
            "min_precision": args.rule_min_precision,
        },
        "n_rules_surviving_filter": int(rules_df.height),
        "rules_distribution": {
            "lift_quartiles": [
                float(rules_df["lift"].quantile(q)) if rules_df.height else 0.0
                for q in [0.25, 0.50, 0.75]
            ] if rules_df.height else [],
            "precision_quartiles": [
                float(rules_df["precision"].quantile(q)) if rules_df.height else 0.0
                for q in [0.25, 0.50, 0.75]
            ] if rules_df.height else [],
            "coverage_n_quartiles": [
                int(rules_df["coverage_n"].quantile(q)) if rules_df.height else 0
                for q in [0.25, 0.50, 0.75]
            ] if rules_df.height else [],
        },
        "features_path": str(features_path.relative_to(_REPO_ROOT)),
        "features_sha256": f"sha256:{_file_sha256(features_path)}",
        "train_cutoff": args.train_cutoff.isoformat(),
        "git_commit_of_quant_repo": _git_head_sha(),
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n=== TRAINING RESULT for {spec.label_column()} ===")
    print(f"  spec:               touch={spec.touch_threshold_pct}% endpoint={spec.endpoint_threshold_pct}% horizon={spec.horizon_days}td")
    print(f"  positives in train: {n_train_pos:,}")
    print(f"  XGB val AUC:        {training_metrics['val_auc']:.4f}")
    print(f"  rules surviving:    {rules_df.height}")
    if rules_df.height:
        print(f"  lift distribution:  {rules_df['lift'].quantile(0.25):.2f} / "
              f"{rules_df['lift'].quantile(0.50):.2f} / "
              f"{rules_df['lift'].quantile(0.75):.2f}")
        print(f"  coverage_n median:  {int(rules_df['coverage_n'].quantile(0.50)):,}")
    print(f"  wall clock:         {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
