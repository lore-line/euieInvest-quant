"""Walk-forward validation for sustained_winner_discovery_v1 — Workstream C step 4.

Re-scores every rule from each `runs/{date}-sustained_winner_v1_g{NN}/rules.parquet`
against 5 chronological 6-month windows over 2024-01-01 → 2026-03-30.

The discovery sweep (step 3) trained XGB on data ≤ 2024-12-31 (train cutoff)
and reported val_auc on > 2024-12-31. That's ONE out-of-sample split. This
module gives finer-grained chronological stability — a rule that survives
all 5 windows is much more credible than one that survives only the
combined 2025+ split.

Output per spec dir `runs/{date}-sustained_winner_v1_g{NN}/`:
  - `walk_forward.parquet` — one row per (rule_id, window_idx) with
    n_match, n_winner_match, n_sample, lift, precision, coverage_pct
  - `walk_forward_aggregate.parquet` — one row per rule_id with
    mean/min/max/std lift across windows, lift_decay, is_walk_forward_survivor

Plus a sweep-level summary: `runs/{date}-sustained_winner_walkforward_summary.json`

Survivor criterion (from server-team Phase B brief, reused here):
  is_walk_forward_survivor = (min_val_lift >= 1.2) AND (lift_decay < 0.5)
  where lift_decay = 1 - (mean_val_lift / train_lift)

Performance: per spec ≈ (5 windows × ~4K unique conditions × ~3ms mask
eval on ~500K-row slice) ≈ 60s. Total sweep ≈ 20 min wall clock.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant.tracks.sustained_winner_label import (
    SPECS,
    SustainedWinnerSpec,
    compute_sustained_winner_label,
    sweep_specs,
)
from quant.tracks.xgb_rule_extraction import (
    Condition,
    Rule,
    _evaluate_rules,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "sustained_winner_walkforward_v1"

# 5 chronological windows over Phase B's standard validation period.
# Matches `walkforward_validate.WINDOWS` for direct comparability against
# Phase A (Track 1/4/5) rule catalogs.
WINDOWS: list[tuple[date, date]] = [
    (date(2024, 1, 1), date(2024, 6, 30)),
    (date(2024, 7, 1), date(2024, 12, 31)),
    (date(2025, 1, 1), date(2025, 6, 30)),
    (date(2025, 7, 1), date(2025, 12, 31)),
    (date(2026, 1, 1), date(2026, 3, 30)),
]

# Survivor thresholds — same criterion as Phase B (walkforward_validate)
SURVIVOR_MIN_LIFT = 1.2
SURVIVOR_MAX_LIFT_DECAY = 0.5


# -------------------- rule + feature prep --------------------

def _rules_from_parquet(path: Path) -> tuple[list[Rule], dict[int, float]]:
    """Load rules.parquet and reconstruct Rule objects.

    Returns (rules, train_lift_by_rule_id) — train_lift is the lift the
    rule had on the discovery train split, retained for lift_decay calc.
    """
    df = pl.read_parquet(path)
    rules: list[Rule] = []
    train_lifts: dict[int, float] = {}
    for row in df.iter_rows(named=True):
        rid = int(row["rule_id"])
        conds_dicts = json.loads(row["conditions_json"])
        conds = tuple(
            Condition(
                feature=c["feature"],
                op=c["op"],
                threshold=float(c["threshold"]),
            )
            for c in conds_dicts
        )
        rules.append(Rule(conditions=conds))
        train_lifts[rid] = float(row["lift"])
    return rules, train_lifts


def _label_features_once(features: pl.DataFrame, horizon_days: int) -> pl.DataFrame:
    """Compute forward_max_pct + forward_endpoint_pct ONCE, and apply the
    SAME feature transformations the training driver applied (bool → int8,
    string → one-hot dummies). This is critical: rules.parquet has
    conditions referencing the post-transform column names (e.g.
    `market_regime_chop` rather than `market_regime`).

    These columns are spec-independent given a fixed horizon (whole sweep
    uses 20 trading days). Per-spec label derivation is then just a cheap
    threshold-comparison op.
    """
    standard = SPECS["standard"]
    if standard.horizon_days != horizon_days:
        raise ValueError(
            f"horizon mismatch: standard spec uses {standard.horizon_days}td "
            f"but caller requested {horizon_days}td"
        )
    labeled = compute_sustained_winner_label(features, standard).drop(
        "is_sustained_winner_standard"
    )
    # Replay sustained_winner_train._prepare_training_matrix transformations
    # so feature column names match what rules reference.
    # (We can't import _NON_FEATURE_COLS-based logic verbatim because we
    # haven't dropped the label yet; do the dtype-driven transforms only.)
    bool_cols = [c for c in labeled.columns if labeled[c].dtype == pl.Boolean
                 and c != "is_winner"]  # is_winner gets added later per-spec
    if bool_cols:
        labeled = labeled.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])
    str_cols = [c for c in labeled.columns if labeled[c].dtype == pl.Utf8
                and c not in ("symbol",)]  # symbol stays as a key column
    if str_cols:
        labeled = labeled.to_dummies(columns=str_cols)
    return labeled


def _derive_label_for_spec(
    labeled_with_returns: pl.DataFrame, spec: SustainedWinnerSpec
) -> pl.DataFrame:
    """Apply spec thresholds to the precomputed forward returns."""
    return labeled_with_returns.with_columns(
        is_winner=pl.when(
            (pl.col("close_adj") >= spec.min_entry_price_usd)
            & pl.col("forward_max_pct").is_not_null()
            & pl.col("forward_endpoint_pct").is_not_null()
        )
        .then(
            (pl.col("forward_max_pct") >= spec.touch_threshold_pct)
            & (pl.col("forward_endpoint_pct") >= spec.endpoint_threshold_pct)
        )
        .otherwise(None)
    )


# -------------------- per-spec walk-forward --------------------

def _walkforward_one_spec(
    labeled_with_returns: pl.DataFrame,
    spec: SustainedWinnerSpec,
    rules: list[Rule],
    train_lifts: dict[int, float],
    out_dir: Path,
) -> dict:
    """Score every rule across the 5 windows, write walk_forward.parquet
    and walk_forward_aggregate.parquet to out_dir.

    Returns a sweep-summary row dict.
    """
    labeled = _derive_label_for_spec(labeled_with_returns, spec)
    labelable = labeled.filter(pl.col("is_winner").is_not_null())

    rows_out: list[dict] = []
    for w_idx, (w_start, w_end) in enumerate(WINDOWS):
        window_slice = labelable.filter(
            (pl.col("date") >= w_start) & (pl.col("date") <= w_end)
        )
        if window_slice.height == 0:
            continue
        is_winner = window_slice["is_winner"].cast(pl.Int8).to_numpy().astype(bool)
        per_rule = _evaluate_rules(rules, window_slice, is_winner, examples_max=0)
        # _evaluate_rules returns rule_id as positional index in `rules` —
        # which aligns 1:1 with rules.parquet rule_id (we iterated in same
        # order at load time).
        for r in per_rule:
            cov_n = int(r["coverage_n"])
            prec = float(r["precision"])
            rows_out.append({
                "rule_id": int(r["rule_id"]),
                "window_idx": w_idx,
                "window_start": w_start,
                "window_end": w_end,
                "n_match": cov_n,
                "n_winner_match": int(round(cov_n * prec)),
                "n_sample": int(window_slice.height),
                "lift": float(r["lift"]),
                "precision": prec,
                "coverage_pct": float(r["coverage_pct"]),
            })

    wf_df = pl.DataFrame(rows_out)
    wf_path = out_dir / "walk_forward.parquet"
    wf_df.write_parquet(wf_path)

    # Aggregate: per rule_id across windows
    agg_rows: list[dict] = []
    if wf_df.height > 0:
        by_rule = wf_df.group_by("rule_id").agg([
            pl.col("lift").mean().alias("mean_val_lift"),
            pl.col("lift").min().alias("min_val_lift"),
            pl.col("lift").max().alias("max_val_lift"),
            pl.col("lift").std().alias("std_val_lift"),
            pl.col("precision").mean().alias("mean_val_precision"),
            pl.col("n_match").sum().alias("total_n_match"),
            pl.col("n_winner_match").sum().alias("total_n_winner_match"),
            pl.len().alias("n_windows_scored"),
        ])
        for row in by_rule.iter_rows(named=True):
            rid = int(row["rule_id"])
            train_lift = train_lifts.get(rid, float("nan"))
            mean_val = float(row["mean_val_lift"])
            lift_decay = (
                1.0 - (mean_val / train_lift)
                if train_lift and not np.isnan(train_lift) and train_lift > 0
                else float("nan")
            )
            is_survivor = (
                float(row["min_val_lift"]) >= SURVIVOR_MIN_LIFT
                and (np.isnan(lift_decay) or lift_decay < SURVIVOR_MAX_LIFT_DECAY)
            )
            agg_rows.append({
                "rule_id": rid,
                "train_lift": train_lift,
                "mean_val_lift": mean_val,
                "min_val_lift": float(row["min_val_lift"]),
                "max_val_lift": float(row["max_val_lift"]),
                "std_val_lift": float(row["std_val_lift"] or 0.0),
                "mean_val_precision": float(row["mean_val_precision"]),
                "total_n_match": int(row["total_n_match"]),
                "total_n_winner_match": int(row["total_n_winner_match"]),
                "n_windows_scored": int(row["n_windows_scored"]),
                "lift_decay": lift_decay,
                "is_walk_forward_survivor": is_survivor,
            })
    agg_df = pl.DataFrame(agg_rows)
    agg_path = out_dir / "walk_forward_aggregate.parquet"
    agg_df.write_parquet(agg_path)

    survivors = int(agg_df.filter(pl.col("is_walk_forward_survivor")).height) if agg_df.height > 0 else 0
    return {
        "spec": spec.name,
        "n_rules_input": len(rules),
        "n_rules_aggregated": int(agg_df.height),
        "n_walk_forward_survivors": survivors,
        "survivor_rate": (survivors / len(rules)) if rules else 0.0,
        "median_train_lift": float(agg_df["train_lift"].median()) if agg_df.height else float("nan"),
        "median_mean_val_lift": float(agg_df["mean_val_lift"].median()) if agg_df.height else float("nan"),
        "median_lift_decay": float(agg_df["lift_decay"].median()) if agg_df.height else float("nan"),
        "wf_parquet": str(wf_path),
        "agg_parquet": str(agg_path),
    }


# -------------------- CLI --------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features", type=Path,
        default=Path("data/features/features.parquet"),
    )
    p.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
    )
    p.add_argument(
        "--date-prefix", type=str, default="2026-05-17",
        help="Filter runs/{prefix}-sustained_winner_v1_g*/ dirs.",
    )
    p.add_argument(
        "--specs", type=str, default=None,
        help="Comma-separated subset of spec names (e.g. 'g20,g15,g10'). Default: all 20.",
    )
    p.add_argument(
        "--horizon-days", type=int, default=20,
        help="Forward horizon in trading days (matches sweep spec). Default 20.",
    )
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features_path = _resolve(args.features)
    runs_dir = _resolve(args.runs_dir)

    spec_dirs = sorted(
        runs_dir.glob(f"{args.date_prefix}-sustained_winner_v1_g*"),
        key=lambda d: int(d.name.split("_g")[-1]),
        reverse=True,
    )
    if args.specs:
        wanted = {s.strip() for s in args.specs.split(",")}
        spec_dirs = [d for d in spec_dirs if d.name.split("_v1_")[-1] in wanted]
    if not spec_dirs:
        print(f"no spec dirs matching {args.date_prefix}-sustained_winner_v1_g* in {runs_dir}")
        return 1

    print(f"walkforward v1 — {len(spec_dirs)} specs queued")
    print(f"features:  {features_path}")
    print(f"horizon:   {args.horizon_days}td")
    print(f"windows:   {len(WINDOWS)} chronological splits")
    print()

    t0 = time.perf_counter()
    features = pl.read_parquet(features_path)
    print(f"loaded {features.height:,} rows × {len(features.columns)} cols in {time.perf_counter()-t0:.1f}s")

    t_label = time.perf_counter()
    labeled_with_returns = _label_features_once(features, args.horizon_days)
    print(f"computed forward returns (max + endpoint) in {time.perf_counter()-t_label:.1f}s")
    print()

    sweep_lookup = {s.name: s for s in sweep_specs()} | {s.name: s for s in SPECS.values()}

    summary_rows: list[dict] = []
    for spec_dir in spec_dirs:
        spec_name = spec_dir.name.split("_v1_")[-1]
        spec = sweep_lookup.get(spec_name)
        if spec is None:
            print(f"  skip {spec_dir.name}: unknown spec '{spec_name}'")
            continue
        rules_parquet = spec_dir / "rules.parquet"
        if not rules_parquet.exists():
            print(f"  skip {spec_dir.name}: no rules.parquet")
            continue

        t_spec = time.perf_counter()
        rules, train_lifts = _rules_from_parquet(rules_parquet)
        result = _walkforward_one_spec(
            labeled_with_returns, spec, rules, train_lifts, spec_dir,
        )
        elapsed = time.perf_counter() - t_spec
        print(
            f"  {spec.name}: {result['n_rules_input']:,} rules → "
            f"{result['n_walk_forward_survivors']:,} survivors "
            f"({100.0 * result['survivor_rate']:.1f}%)  "
            f"med_train_lift={result['median_train_lift']:.2f}  "
            f"med_val_lift={result['median_mean_val_lift']:.2f}  "
            f"med_decay={result['median_lift_decay']:.2f}  ({elapsed:.1f}s)"
        )
        summary_rows.append(result)

    summary_path = runs_dir / f"{args.date_prefix}-sustained_winner_walkforward_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "pipeline_step": PIPELINE_STEP,
            "windows": [(s.isoformat(), e.isoformat()) for s, e in WINDOWS],
            "survivor_criterion": {
                "min_lift": SURVIVOR_MIN_LIFT,
                "max_lift_decay": SURVIVOR_MAX_LIFT_DECAY,
            },
            "specs": summary_rows,
            "total_wall_clock_s": round(time.perf_counter() - t0, 1),
        }, f, indent=2)
    print()
    print(f"summary: {summary_path}")
    print(f"total wall clock: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
