"""Track 4 — Multi-label rule extraction.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 4.

Re-train XGB on 5 alternative labels, extract rules from each (same
mechanism as Track 1), cross-tabulate features appearing across
labels. Features that appear in top-20 rules across 3+ labels are
*structurally stable* signals. Features in only one label are
label-specific noise.

Labels:
  L1  existing +20%/30d on close_adj                                       (same as Step 2)
  L2  +30%/90d                                                              (stronger move, longer window)
  L3  +15%/10d                                                              (shorter window, lower threshold — quick burst)
  L4  +20%/30d AND max_drawdown[t..t+30] >= -0.10                          (smooth winners only)
  L5  drawdown predictor: min(close_adj[t+1..t+30]) / close_adj[t] <= 0.85  (losers — sign-flipped target)

Outputs:
  rules_L1.parquet ... rules_L5.parquet — same schema as Track 1's rules.parquet
  stable-features.parquet               — (feature_name, n_labels_top20, label_ids_present)

CPU only; ~3-5 min total (5 train+walk cycles, each fast since we
use a small 100-tree model).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import xgboost as xgb

from quant.backtest.temporal import split_by_date
from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id
from quant.tracks.xgb_rule_extraction import (
    Condition,
    Rule,
    _build_condition_masks,
    _evaluate_rules,
    _replay_feature_selection,
    extract_paths,
)

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ----- label definitions -----


def _label_l1_l2_l3(
    df: pl.DataFrame, lookahead: int, threshold: float
) -> pl.DataFrame:
    """Append `label` column: max(close_adj[t+1..t+lookahead]) / close_adj[t] >= 1+threshold."""
    return df.sort(["symbol", "date"]).with_columns(
        _fmax=pl.col("close_adj")
        .rolling_max(window_size=lookahead, min_samples=lookahead)
        .shift(-lookahead)
        .over("symbol")
    ).with_columns(
        label=(pl.col("_fmax") / pl.col("close_adj") >= 1.0 + threshold)
    ).drop("_fmax")


def _label_l4(df: pl.DataFrame, lookahead: int = 30, up_thresh: float = 0.20, max_dd: float = -0.10) -> pl.DataFrame:
    """+20%/30d AND drawdown in the same window >= -10%."""
    return df.sort(["symbol", "date"]).with_columns(
        _fmax=pl.col("close_adj")
        .rolling_max(window_size=lookahead, min_samples=lookahead)
        .shift(-lookahead)
        .over("symbol"),
        _fmin=pl.col("close_adj")
        .rolling_min(window_size=lookahead, min_samples=lookahead)
        .shift(-lookahead)
        .over("symbol"),
    ).with_columns(
        label=(
            (pl.col("_fmax") / pl.col("close_adj") >= 1.0 + up_thresh)
            & (pl.col("_fmin") / pl.col("close_adj") >= 1.0 + max_dd)
        )
    ).drop("_fmax", "_fmin")


def _label_l5(df: pl.DataFrame, lookahead: int = 30, down_thresh: float = -0.15) -> pl.DataFrame:
    """Loser predictor: min(close_adj[t+1..t+30]) / close_adj[t] <= 0.85 (≥15% drawdown)."""
    return df.sort(["symbol", "date"]).with_columns(
        _fmin=pl.col("close_adj")
        .rolling_min(window_size=lookahead, min_samples=lookahead)
        .shift(-lookahead)
        .over("symbol")
    ).with_columns(
        label=(pl.col("_fmin") / pl.col("close_adj") <= 1.0 + down_thresh)
    ).drop("_fmin")


LABELS: dict[str, dict[str, Any]] = {
    "L1": {"fn": lambda df: _label_l1_l2_l3(df, 30, 0.20), "desc": "+20%/30d (replication of Step 2)"},
    "L2": {"fn": lambda df: _label_l1_l2_l3(df, 90, 0.30), "desc": "+30%/90d"},
    "L3": {"fn": lambda df: _label_l1_l2_l3(df, 10, 0.15), "desc": "+15%/10d (quick-burst)"},
    "L4": {"fn": lambda df: _label_l4(df), "desc": "+20%/30d AND drawdown >= -10% (smooth winners)"},
    "L5": {"fn": lambda df: _label_l5(df), "desc": "drawdown <= -15% in next 30d (losers)"},
}


# ----- per-label training + rule extraction -----


def _train_xgb_on_label(
    train_df: pl.DataFrame, feature_cols: list[str]
) -> xgb.Booster:
    """Small 100-tree booster — enough to surface candidate rules; fast.

    Distinct from the Step 2 model (which is 400 trees with full hyper-
    parameters) — we don't need predictor quality here, just diverse
    decision paths. Same scale_pos_weight pattern.
    """
    y = train_df["label"].cast(pl.Boolean).to_numpy().astype(np.int8)
    X = train_df.select(feature_cols).to_numpy().astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32, copy=False)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_child_weight": 5,
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=100)
    return booster


def _extract_filtered_rules(
    booster: xgb.Booster,
    holdout_df: pl.DataFrame,
    *,
    min_lift: float,
    min_coverage_pct: float,
    min_precision: float,
) -> list[dict[str, Any]]:
    """Walk paths → dedup → evaluate → filter → sort by lift × coverage."""
    paths = extract_paths(booster)
    rule_set: dict[Rule, int] = {}
    for p in paths:
        r = Rule.from_path(p)
        rule_set.setdefault(r, len(rule_set))
    rules = list(rule_set.keys())
    is_winner = holdout_df["label"].cast(pl.Boolean).to_numpy()
    records = _evaluate_rules(rules, holdout_df, is_winner)
    kept = [
        r for r in records
        if r["lift"] >= min_lift
        and r["coverage_pct"] >= min_coverage_pct
        and r["precision"] >= min_precision
    ]
    kept.sort(key=lambda r: r["lift"] * r["coverage_pct"], reverse=True)
    for new_id, r in enumerate(kept):
        r["rule_id"] = new_id
    return kept


# ----- main -----


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 4 — multi-label rule extraction")
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--train-end", type=date.fromisoformat, default=date(2023, 12, 31))
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--min-lift", type=float, default=1.5)
    p.add_argument("--min-coverage-pct", type=float, default=0.5)
    p.add_argument("--min-precision", type=float, default=0.35)
    p.add_argument("--top-k-for-stability", type=int, default=20)
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step3c_multi_label_rules"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(dir=run_dir, run_id=make_run_id(run_date_str, pipeline_step), pipeline_step=pipeline_step, epoch_total=len(LABELS))
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        print(f"track 4 (multi-label rules) — run dir: {run_dir.relative_to(_REPO_ROOT)}")
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        if not features_path.exists():
            raise FileNotFoundError(features_path)

        # Load full features ONCE; replay feature selection ONCE.
        labeled_raw = pl.read_parquet(features_path)
        # We re-derive labels below, so drop the existing is_winner here.
        labeled_raw = labeled_raw.drop("is_winner")
        # Replay needs is_winner to filter — but we're computing new labels.
        # Add a temporary not-null is_winner via L1's computation, then we'll
        # swap labels per-iteration.
        l1_full = _label_l1_l2_l3(labeled_raw, 30, 0.20).rename({"label": "is_winner"})
        labeled, feature_cols = _replay_feature_selection(
            l1_full.filter(pl.col("is_winner").is_not_null())
        )
        labeled = labeled.drop("is_winner")  # we'll add `label` per iteration
        print(f"  base frame: {labeled.height:,} rows × {len(feature_cols)} features")

        rules_per_label: dict[str, list[dict[str, Any]]] = {}
        base_rates: dict[str, float] = {}
        for i, (label_id, spec) in enumerate(LABELS.items(), start=1):
            if stop_flag["stop"]:
                raise KeyboardInterrupt
            print(f"  [{i}/{len(LABELS)}] {label_id} — {spec['desc']}")
            t_label = time.perf_counter()
            scored = spec["fn"](labeled).filter(pl.col("label").is_not_null())
            n_rows = scored.height
            n_pos = int(scored["label"].sum())
            base_rate = n_pos / n_rows if n_rows else 0.0
            print(f"    labeled rows: {n_rows:,}  positives: {n_pos:,} ({base_rate*100:.2f}%)")

            train, val, holdout = split_by_date(scored, args.train_end, args.val_end)
            print(f"    splits: train={train.height:,} val={val.height:,} holdout={holdout.height:,}")
            booster = _train_xgb_on_label(train, feature_cols)

            print(f"    walking + evaluating ...")
            kept = _extract_filtered_rules(
                booster, holdout,
                min_lift=args.min_lift,
                min_coverage_pct=args.min_coverage_pct,
                min_precision=args.min_precision,
            )
            for r in kept:
                r["label_id"] = label_id
            rules_per_label[label_id] = kept
            base_rates[label_id] = base_rate
            print(f"    kept {len(kept)} rules ({time.perf_counter() - t_label:.1f}s)")
            status.record_checkpoint(epoch=i)
            status.update(state="training", epoch_current=i)

        # Per-label rules.parquet files.
        schema = {
            "rule_id": pl.Int64,
            "conditions_json": pl.Utf8,
            "n_conditions": pl.Int64,
            "coverage_n": pl.Int64,
            "coverage_pct": pl.Float64,
            "precision": pl.Float64,
            "lift": pl.Float64,
            "example_symbol_dates_json": pl.Utf8,
            "label_id": pl.Utf8,
        }
        for label_id, kept in rules_per_label.items():
            df = pl.DataFrame(kept, schema=schema)
            path = run_dir / f"rules_{label_id}.parquet"
            df.write_parquet(path)
            print(f"  wrote {path.relative_to(_REPO_ROOT)}  ({df.height} rules)")

        # Cross-tabulation: features in top-K rules per label → label coverage.
        from collections import defaultdict
        feature_labels: dict[str, set[str]] = defaultdict(set)
        top_k = args.top_k_for_stability
        for label_id, kept in rules_per_label.items():
            for r in kept[:top_k]:
                conds = json.loads(r["conditions_json"])
                for c in conds:
                    feature_labels[c["feature"]].add(label_id)
        stable_rows = sorted(
            (
                {
                    "feature_name": feat,
                    "n_labels_top20": len(labels),
                    "label_ids_present": ",".join(sorted(labels)),
                }
                for feat, labels in feature_labels.items()
            ),
            key=lambda r: -r["n_labels_top20"],
        )
        stable_df = pl.DataFrame(
            stable_rows,
            schema={
                "feature_name": pl.Utf8,
                "n_labels_top20": pl.Int64,
                "label_ids_present": pl.Utf8,
            },
        )
        stable_path = run_dir / "stable-features.parquet"
        stable_df.write_parquet(stable_path)
        print(f"  wrote {stable_path.relative_to(_REPO_ROOT)}  ({stable_df.height} features)")
        print(f"    features in all 5 labels:  {stable_df.filter(pl.col('n_labels_top20') == 5).height}")
        print(f"    features in 3+ labels:     {stable_df.filter(pl.col('n_labels_top20') >= 3).height}")

        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "train_end": args.train_end.isoformat(),
            "val_end": args.val_end.isoformat(),
            "labels": {
                label_id: {
                    "description": spec["desc"],
                    "base_rate_full": round(base_rates.get(label_id, 0.0), 6),
                    "n_rules_kept": len(rules_per_label.get(label_id, [])),
                }
                for label_id, spec in LABELS.items()
            },
            "filter_min_lift": args.min_lift,
            "filter_min_coverage_pct": args.min_coverage_pct,
            "filter_min_precision": args.min_precision,
            "top_k_for_stability": top_k,
            "n_stable_features_5_of_5": int(stable_df.filter(pl.col("n_labels_top20") == 5).height),
            "n_stable_features_3_plus": int(stable_df.filter(pl.col("n_labels_top20") >= 3).height),
            "feature_count": len(feature_cols),
            "runtime_device": "cpu",
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(f"=== TRACK 4 RESULT: {sum(len(r) for r in rules_per_label.values())} rules across {len(LABELS)} labels ({wall_clock_s:.1f}s) ===")
        status.update(state="done", epoch_current=len(LABELS))
        return 0
    except KeyboardInterrupt:
        status.update(state="paused", epoch_current=0)
        return 130
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
