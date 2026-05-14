"""Track B-walk — walk-forward validation of Phase A rule catalog.

Re-scores every Phase A rule (Tracks 1 / 4 / 5) on rolling 6-month
holdouts over 2024-01-01 → 2026-03-30. The goal: separate rules that
survive across multiple out-of-sample windows from rules that
overfit the Phase A train slice.

See `docs/theses-from-discovery-v2.md` and PR #1 issuecomment-4441437671
(server team's Phase B brief) for context.

Rule sources & rule_key convention
----------------------------------

Three Phase A tracks produce rules. We canonicalize them under a single
rule_key namespace so downstream ingest treats them uniformly:

- **Track 1** (step3a_xgb_rule_extraction) — 1,100 rules.
  rule_key = ``str(rule_id)`` (bare; e.g. ``"42"``).
- **Track 4** (step3c_multi_label_rules) — 5 alt-label models.
  rule_key = ``f"{label_id}_{rule_id}"`` (e.g. ``"L2_137"``).
- **Track 5** (step3d_per_regime_rules) — 4 regime models.
  rule_key = ``f"{regime}_{rule_id}"`` (e.g. ``"bear_42"``).

Server team's brief uses bare for Track 1, label-prefixed for Track 4,
regime-prefixed for Track 5 — same convention.

Walk-forward windows
--------------------

5 non-overlapping windows over 2024-01-01 → 2026-03-30:

- 2024-01-01 → 2024-06-30
- 2024-07-01 → 2024-12-31
- 2025-01-01 → 2025-06-30
- 2025-07-01 → 2025-12-31
- 2026-01-01 → 2026-03-30  (3-month tail; deliberately shorter to capture
  the most recent regime)

Outputs
-------

- ``rule-validation.parquet`` — one row per (rule_key, walk_forward_window)
  Columns: rule_key, walk_forward_window, window_start, window_end,
  val_lift, val_precision, val_coverage, val_n_winners, val_n_samples,
  train_lift, train_precision, source_track, source_label_or_regime.
- ``rule-validation-aggregate.parquet`` — one row per rule_key
  Columns: rule_key, source_track, train_lift, mean_val_lift, min_val_lift,
  max_val_lift, std_val_lift, n_windows_lift_ge_1_5, lift_decay,
  is_walk_forward_survivor (lift_decay < 0.5 AND min_val_lift >= 1.5
  per the brief's loose-survivor criterion).

Cost
----

CPU-bound. ~4,500 rules × 5 windows × ~5ms per evaluation ≈ 2 min wall
clock on a workstation. No GPU.
"""
from __future__ import annotations

import argparse
import json
import operator
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import polars as pl

from quant.train import RunStatus
from quant.tracks import make_run_id

__all__ = ["main", "evaluate_rule_on_slice"]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Walk-forward windows per server-team brief.
WINDOWS: list[tuple[date, date]] = [
    (date(2024, 1, 1), date(2024, 6, 30)),
    (date(2024, 7, 1), date(2024, 12, 31)),
    (date(2025, 1, 1), date(2025, 6, 30)),
    (date(2025, 7, 1), date(2025, 12, 31)),
    (date(2026, 1, 1), date(2026, 3, 30)),
]

# Operator map for rule conditions. Phase A only uses these.
_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Rule:
    """Canonical rule record shared across Tracks 1/4/5."""

    rule_key: str  # e.g. "42", "L1_42", "bear_42"
    source_track: str  # "step3a", "step3c", "step3d"
    source_label_or_regime: str  # "" for Track 1, "L1"..."L5" for Track 4, "bull"/"bear"/... for Track 5
    conditions: list[dict[str, Any]]  # [{feature, op, threshold}, ...]
    train_lift: float
    train_precision: float


def _condition_expr(cond: dict[str, Any]) -> pl.Expr:
    """Convert one condition dict to a polars expression on the labeled df."""
    feat = cond["feature"]
    op = cond["op"]
    thr = cond["threshold"]
    col = pl.col(feat)
    if op == "<":
        return col < thr
    if op == "<=":
        return col <= thr
    if op == ">":
        return col > thr
    if op == ">=":
        return col >= thr
    if op == "==":
        return col == thr
    if op == "!=":
        return col != thr
    raise ValueError(f"unknown op: {op}")


def evaluate_rule_on_slice(slice_df: pl.DataFrame, conditions: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Evaluate a single rule against a labeled DataFrame slice.

    Returns (n_matches, n_winner_matches, n_samples).

    n_samples excludes rows with null ``is_winner`` (the last 30 days per
    symbol are unlabeled in Phase A).
    """
    valid = slice_df.filter(pl.col("is_winner").is_not_null())
    n_samples = valid.height
    if n_samples == 0:
        return 0, 0, 0
    # Build a single combined mask. Missing features in any condition
    # → rule doesn't match (treated as 0). This shouldn't happen in
    # practice for Phase A rules since they all come from the same
    # feature set, but defensive.
    expr = None
    for cond in conditions:
        feat = cond["feature"]
        if feat not in valid.columns:
            return 0, 0, n_samples
        e = _condition_expr(cond)
        expr = e if expr is None else (expr & e)
    if expr is None:  # no conditions — degenerate rule
        return 0, 0, n_samples
    matches = valid.filter(expr)
    n_matches = matches.height
    if n_matches == 0:
        return 0, 0, n_samples
    n_winner_matches = matches.filter(pl.col("is_winner") == True).height
    return n_matches, n_winner_matches, n_samples


def load_phase_a_rules(
    track1_dir: Path,
    track4_dir: Path,
    track5_dir: Path,
) -> list[Rule]:
    """Load all rules from Tracks 1/4/5 into a single Rule list."""
    rules: list[Rule] = []

    # Track 1.
    t1_path = track1_dir / "rules.parquet"
    if t1_path.exists():
        t1 = pl.read_parquet(t1_path)
        for row in t1.iter_rows(named=True):
            rules.append(
                Rule(
                    rule_key=str(row["rule_id"]),
                    source_track="step3a",
                    source_label_or_regime="",
                    conditions=json.loads(row["conditions_json"]),
                    train_lift=float(row["lift"]),
                    train_precision=float(row["precision"]),
                )
            )

    # Track 4 — per label.
    for label_id in ("L1", "L2", "L3", "L4", "L5"):
        t4_path = track4_dir / f"rules_{label_id}.parquet"
        if not t4_path.exists():
            continue
        t4 = pl.read_parquet(t4_path)
        for row in t4.iter_rows(named=True):
            rules.append(
                Rule(
                    rule_key=f"{label_id}_{row['rule_id']}",
                    source_track="step3c",
                    source_label_or_regime=label_id,
                    conditions=json.loads(row["conditions_json"]),
                    train_lift=float(row["lift"]),
                    train_precision=float(row["precision"]),
                )
            )

    # Track 5 — per regime.
    for regime in ("bull", "bear", "chop", "recovery"):
        t5_path = track5_dir / f"rules-{regime}.parquet"
        if not t5_path.exists():
            continue
        t5 = pl.read_parquet(t5_path)
        for row in t5.iter_rows(named=True):
            rules.append(
                Rule(
                    rule_key=f"{regime}_{row['rule_id']}",
                    source_track="step3d",
                    source_label_or_regime=regime,
                    conditions=json.loads(row["conditions_json"]),
                    train_lift=float(row["lift"]),
                    train_precision=float(row["precision"]),
                )
            )

    return rules


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B Track B-walk — walk-forward rule validation")
    p.add_argument(
        "--features", type=Path, default=Path("data/features/features.parquet"),
        help="Path to the labeled feature matrix (post-Phase-A engineering).",
    )
    p.add_argument(
        "--track1-dir", type=Path, default=Path("runs/2026-05-13-step3a_xgb_rule_extraction"),
    )
    p.add_argument(
        "--track4-dir", type=Path, default=Path("runs/2026-05-13-step3c_multi_label_rules"),
    )
    p.add_argument(
        "--track5-dir", type=Path, default=Path("runs/2026-05-13-step3d_per_regime_rules"),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--survivor-lift-min", type=float, default=1.5,
        help="Per the brief: walk-forward survivor = min_val_lift >= 1.5 across all windows.",
    )
    p.add_argument(
        "--survivor-decay-max", type=float, default=0.5,
        help="Optional secondary criterion: lift_decay < this. Set high (e.g. 999) to disable.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step4_walkforward_validation"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(
        dir=run_dir,
        run_id=make_run_id(run_date_str, pipeline_step),
        pipeline_step=pipeline_step,
        epoch_total=len(WINDOWS),
    )
    status.update(state="training", epoch_current=0)

    try:
        print(f"track B-walk — walk-forward rule validation")
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        track1_dir = args.track1_dir if args.track1_dir.is_absolute() else (_REPO_ROOT / args.track1_dir)
        track4_dir = args.track4_dir if args.track4_dir.is_absolute() else (_REPO_ROOT / args.track4_dir)
        track5_dir = args.track5_dir if args.track5_dir.is_absolute() else (_REPO_ROOT / args.track5_dir)

        print(f"  features:    {features_path.relative_to(_REPO_ROOT)}")
        print(f"  track1 dir:  {track1_dir.relative_to(_REPO_ROOT)}")
        print(f"  track4 dir:  {track4_dir.relative_to(_REPO_ROOT)}")
        print(f"  track5 dir:  {track5_dir.relative_to(_REPO_ROOT)}")

        rules = load_phase_a_rules(track1_dir, track4_dir, track5_dir)
        print(f"  loaded {len(rules):,} rules total:")
        by_track = {}
        for r in rules:
            by_track[r.source_track] = by_track.get(r.source_track, 0) + 1
        for tk, n in sorted(by_track.items()):
            print(f"    {tk}: {n:,}")

        labeled = pl.read_parquet(features_path)
        print(f"  labeled features: {labeled.height:,} rows, {len(labeled.columns)} cols")

        # Pre-slice once per window — major speedup vs re-filtering on every rule.
        window_slices: list[pl.DataFrame] = []
        for window_id, (start, end) in enumerate(WINDOWS):
            sl = labeled.filter(
                (pl.col("date") >= start) & (pl.col("date") <= end)
            )
            window_slices.append(sl)
            print(f"  window {window_id}: {start.isoformat()} → {end.isoformat()}  {sl.height:,} rows")

        # Per-window base rate (used for lift normalization).
        window_base_rates: list[float] = []
        for sl in window_slices:
            valid = sl.filter(pl.col("is_winner").is_not_null())
            if valid.height == 0:
                window_base_rates.append(0.0)
                continue
            n_win = valid.filter(pl.col("is_winner") == True).height
            window_base_rates.append(n_win / valid.height)
        print(f"  window base rates: {[round(r, 4) for r in window_base_rates]}")

        # Evaluate.
        print(f"  evaluating {len(rules)} rules × {len(WINDOWS)} windows ...")
        rows: list[dict[str, Any]] = []
        t_eval = time.perf_counter()
        for i, rule in enumerate(rules):
            for window_id, (start, end) in enumerate(WINDOWS):
                sl = window_slices[window_id]
                n_matches, n_winner_matches, n_samples = evaluate_rule_on_slice(sl, rule.conditions)
                base = window_base_rates[window_id]
                if n_matches > 0 and base > 0:
                    val_precision = n_winner_matches / n_matches
                    val_lift = val_precision / base
                    val_coverage = n_matches / n_samples if n_samples > 0 else 0.0
                else:
                    val_precision = 0.0
                    val_lift = 0.0
                    val_coverage = 0.0
                rows.append({
                    "rule_key": rule.rule_key,
                    "source_track": rule.source_track,
                    "source_label_or_regime": rule.source_label_or_regime,
                    "walk_forward_window": window_id,
                    "window_start": start.isoformat(),
                    "window_end": end.isoformat(),
                    "val_lift": round(val_lift, 6),
                    "val_precision": round(val_precision, 6),
                    "val_coverage": round(val_coverage, 6),
                    "val_n_winners": int(n_winner_matches),
                    "val_n_samples": int(n_samples),
                    "train_lift": round(rule.train_lift, 6),
                    "train_precision": round(rule.train_precision, 6),
                })
            if (i + 1) % 500 == 0:
                elapsed = time.perf_counter() - t_eval
                eta = elapsed / (i + 1) * (len(rules) - i - 1)
                print(f"    {i+1:,}/{len(rules):,} rules ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")
                status.update(state="training", epoch_current=window_id + 1, extras={
                    "rules_processed": i + 1, "rules_total": len(rules)
                })

        df = pl.DataFrame(rows)
        out_path = run_dir / "rule-validation.parquet"
        df.write_parquet(out_path)
        print(f"  wrote {out_path.relative_to(_REPO_ROOT)}  ({df.height:,} rows)")

        # Aggregate per-rule.
        agg = df.group_by("rule_key").agg(
            pl.col("source_track").first(),
            pl.col("source_label_or_regime").first(),
            pl.col("train_lift").first(),
            pl.col("val_lift").mean().alias("mean_val_lift"),
            pl.col("val_lift").min().alias("min_val_lift"),
            pl.col("val_lift").max().alias("max_val_lift"),
            pl.col("val_lift").std().alias("std_val_lift"),
            (pl.col("val_lift") >= args.survivor_lift_min).sum().alias("n_windows_lift_ge_1_5"),
            pl.col("val_precision").mean().alias("mean_val_precision"),
            pl.col("val_coverage").mean().alias("mean_val_coverage"),
        ).with_columns(
            (pl.col("train_lift") - pl.col("mean_val_lift")).alias("lift_decay"),
        ).with_columns(
            (
                (pl.col("min_val_lift") >= args.survivor_lift_min)
                & (pl.col("lift_decay") < args.survivor_decay_max)
            ).alias("is_walk_forward_survivor")
        ).sort("mean_val_lift", descending=True)

        agg_path = run_dir / "rule-validation-aggregate.parquet"
        agg.write_parquet(agg_path)
        print(f"  wrote {agg_path.relative_to(_REPO_ROOT)}  ({agg.height:,} rules)")

        n_survivors = int(agg["is_walk_forward_survivor"].sum())
        print(f"  WALK-FORWARD SURVIVORS: {n_survivors} / {agg.height}")

        # Manifest.
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "n_rules_total": len(rules),
            "n_rules_by_source": by_track,
            "n_windows": len(WINDOWS),
            "windows": [
                {"window_id": i, "start": s.isoformat(), "end": e.isoformat(),
                 "n_rows": window_slices[i].height,
                 "base_rate": round(window_base_rates[i], 6)}
                for i, (s, e) in enumerate(WINDOWS)
            ],
            "survivor_lift_min": args.survivor_lift_min,
            "survivor_decay_max": args.survivor_decay_max,
            "n_walk_forward_survivors": n_survivors,
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote manifest.json")

        print()
        print(f"=== TRACK B-WALK RESULT: {n_survivors}/{agg.height} rules survive ({wall_clock_s/60:.1f}min) ===")
        status.record_checkpoint(epoch=len(WINDOWS))
        status.update(state="done", epoch_current=len(WINDOWS))
        return 0
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
