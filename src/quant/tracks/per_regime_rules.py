"""Track 5 — Per-regime rule stability.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 5.

Take Track 1's 1,100 filtered rules and re-evaluate each on four
regime slices (bull / bear / chop / recovery). A rule that maintains
lift ≥ 1.5 in 3+ regimes is *regime-durable* — strong thesis material.
A rule firing only in bull is bull-conditioned (still useful but
tagged for synthesis).

Regimes (brief verbatim):
  bull       2023-10-01 → 2024-12-31
  bear       2022-01-01 → 2022-09-30   ← real bear-tape, missing from Step 2
  chop       2021-05-01 → 2021-12-31 + 2024-09-01 → 2025-02-28
  recovery   2022-10-01 → 2023-09-30

Per the brief, the bear-tape slice is the most important regime-
robustness test in Phase A — most of the discovery work happened on
bull-tape (2023-2025).

Outputs:
  rules-bull.parquet, rules-bear.parquet, rules-chop.parquet,
    rules-recovery.parquet  — Track 1's rules, regime-evaluated
  regime-stability.parquet  — per-rule lift across all 4 regimes +
                              durability flags

CPU only; ~30-60s — same eval logic as Track 1, just 4 different
slices of the same features parquet.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id
from quant.tracks.xgb_rule_extraction import (
    Condition,
    Rule,
    _evaluate_rules,
    _replay_feature_selection,
)

__all__ = ["main", "REGIMES"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


# Date ranges per the brief. Each regime is a list of (lo, hi) inclusive ranges
# so chop can span two disjoint windows.
# Bull end-date corrected from 2024-12-31 → 2024-08-31 per server-team
# regime boundary fix (PR #1 issuecomment-4436499617) — eliminates the
# 4-month overlap with chop's second window (2024-09-01 → 2025-02-28).
REGIMES: dict[str, list[tuple[date, date]]] = {
    "bull":     [(date(2023, 10, 1), date(2024, 8, 31))],
    "bear":     [(date(2022, 1, 1),  date(2022, 9, 30))],
    "chop":     [(date(2021, 5, 1),  date(2021, 12, 31)),
                 (date(2024, 9, 1),  date(2025, 2, 28))],
    "recovery": [(date(2022, 10, 1), date(2023, 9, 30))],
}


def _filter_to_regime(df: pl.DataFrame, ranges: list[tuple[date, date]]) -> pl.DataFrame:
    """Polars OR over date ranges."""
    expr = None
    for lo, hi in ranges:
        cond = (pl.col("date") >= lo) & (pl.col("date") <= hi)
        expr = cond if expr is None else (expr | cond)
    return df.filter(expr) if expr is not None else df


def _load_track1_rules(rules_parquet: Path) -> tuple[list[Rule], list[int]]:
    """Deserialize Track 1's rules back to Rule objects.

    Returns ``(rules, source_rule_ids)`` — source_rule_ids preserves
    Track 1's rank ordering so cross-regime tracking stays aligned with
    the original numbering.
    """
    df = pl.read_parquet(rules_parquet)
    rules: list[Rule] = []
    ids: list[int] = []
    for row in df.iter_rows(named=True):
        conds_json = json.loads(row["conditions_json"])
        conds = tuple(
            Condition(feature=c["feature"], op=c["op"], threshold=float(c["threshold"]))
            for c in conds_json
        )
        rules.append(Rule(conditions=conds))
        ids.append(int(row["rule_id"]))
    return rules, ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 5 — per-regime rule stability")
    p.add_argument(
        "--rules", type=Path,
        default=Path("runs/2026-05-13-step3a_xgb_rule_extraction/rules.parquet"),
        help="Track 1's filtered rules.parquet",
    )
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--min-lift", type=float, default=1.5)
    p.add_argument("--min-coverage-pct", type=float, default=0.5)
    p.add_argument("--min-precision", type=float, default=0.35)
    p.add_argument(
        "--durable-min-regimes", type=int, default=3,
        help="A rule is 'regime-durable' if its lift >= --min-lift in >= N regimes (default 3 of 4).",
    )
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
    pipeline_step = "step3d_per_regime_rules"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(dir=run_dir, run_id=make_run_id(run_date_str, pipeline_step), pipeline_step=pipeline_step, epoch_total=len(REGIMES))
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        print(f"track 5 (per-regime rules) — run dir: {run_dir.relative_to(_REPO_ROOT)}")
        rules_path = args.rules if args.rules.is_absolute() else (_REPO_ROOT / args.rules)
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        if not rules_path.exists():
            raise FileNotFoundError(f"Track 1 rules not found at {rules_path}; run Track 1 first.")
        if not features_path.exists():
            raise FileNotFoundError(features_path)

        rules, source_ids = _load_track1_rules(rules_path)
        print(f"  loaded {len(rules)} rules from Track 1 ({rules_path.relative_to(_REPO_ROOT)})")

        labeled = pl.read_parquet(features_path).filter(pl.col("is_winner").is_not_null())
        labeled, feature_cols = _replay_feature_selection(labeled)
        print(f"  base frame: {labeled.height:,} rows × {len(feature_cols)} features")

        per_regime_records: dict[str, list[dict[str, Any]]] = {}
        for i, (regime_name, ranges) in enumerate(REGIMES.items(), start=1):
            if stop_flag["stop"]:
                raise KeyboardInterrupt
            slice_df = _filter_to_regime(labeled, ranges)
            n_rows = slice_df.height
            if n_rows == 0:
                print(f"  [{i}/{len(REGIMES)}] {regime_name}: 0 rows (dates out of data range)")
                per_regime_records[regime_name] = []
                continue
            is_winner = slice_df["is_winner"].cast(pl.Boolean).to_numpy()
            base_rate = float(is_winner.sum()) / n_rows
            print(
                f"  [{i}/{len(REGIMES)}] {regime_name}: {n_rows:,} rows  "
                f"base_rate={base_rate*100:.2f}%"
            )
            t_reg = time.perf_counter()
            records = _evaluate_rules(rules, slice_df, is_winner)
            kept = [
                r for r in records
                if r["lift"] >= args.min_lift
                and r["coverage_pct"] >= args.min_coverage_pct
                and r["precision"] >= args.min_precision
            ]
            kept.sort(key=lambda r: r["lift"] * r["coverage_pct"], reverse=True)
            # Stamp the source-rule-id so cross-regime tracking works.
            for r in kept:
                src_id = source_ids[r["rule_id"]]
                r["source_rule_id"] = src_id
            for new_id, r in enumerate(kept):
                r["rule_id"] = new_id
            print(f"    kept {len(kept)} rules ({time.perf_counter() - t_reg:.1f}s)")
            per_regime_records[regime_name] = kept
            status.record_checkpoint(epoch=i)
            status.update(state="training", epoch_current=i)

        # Per-regime rules files.
        schema = {
            "rule_id": pl.Int64,
            "source_rule_id": pl.Int64,
            "conditions_json": pl.Utf8,
            "n_conditions": pl.Int64,
            "coverage_n": pl.Int64,
            "coverage_pct": pl.Float64,
            "precision": pl.Float64,
            "lift": pl.Float64,
            "example_symbol_dates_json": pl.Utf8,
        }
        for regime_name, kept in per_regime_records.items():
            df = pl.DataFrame(kept, schema=schema)
            path = run_dir / f"rules-{regime_name}.parquet"
            df.write_parquet(path)
            print(f"  wrote {path.relative_to(_REPO_ROOT)}  ({df.height} rules)")

        # Cross-regime stability.parquet: per source_rule_id, lift in each regime.
        rule_id_to_lift: dict[int, dict[str, float]] = {}
        for regime_name, kept in per_regime_records.items():
            for r in kept:
                rule_id_to_lift.setdefault(r["source_rule_id"], {})[regime_name] = r["lift"]
        stability_rows = []
        regime_order = list(REGIMES.keys())
        for src_id in sorted(rule_id_to_lift.keys()):
            lifts = rule_id_to_lift[src_id]
            durable_count = sum(1 for v in lifts.values() if v >= args.min_lift)
            stability_rows.append({
                "source_rule_id": src_id,
                **{f"lift_{r}": lifts.get(r) for r in regime_order},
                "n_regimes_durable": durable_count,
                "is_durable": durable_count >= args.durable_min_regimes,
            })
        stability_rows.sort(key=lambda r: -r["n_regimes_durable"])
        stability_df = pl.DataFrame(stability_rows)
        stability_path = run_dir / "regime-stability.parquet"
        stability_df.write_parquet(stability_path)
        print(f"  wrote {stability_path.relative_to(_REPO_ROOT)}  ({stability_df.height} rules)")
        n_durable = int(stability_df.filter(pl.col("is_durable")).height)
        print(f"    regime-durable (≥{args.durable_min_regimes} of {len(REGIMES)} regimes): {n_durable}")

        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "source_track1_rules": str(rules_path.relative_to(_REPO_ROOT)),
            "n_source_rules": len(rules),
            "regimes": {
                name: {
                    "ranges": [[lo.isoformat(), hi.isoformat()] for (lo, hi) in ranges],
                    "n_rules_kept": len(per_regime_records.get(name, [])),
                }
                for name, ranges in REGIMES.items()
            },
            "n_regime_durable": n_durable,
            "durable_min_regimes": args.durable_min_regimes,
            "filter_min_lift": args.min_lift,
            "filter_min_coverage_pct": args.min_coverage_pct,
            "filter_min_precision": args.min_precision,
            "feature_count": len(feature_cols),
            "runtime_device": "cpu",
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(f"=== TRACK 5 RESULT: {n_durable} regime-durable rules ({wall_clock_s:.1f}s) ===")
        status.update(state="done", epoch_current=len(REGIMES))
        return 0
    except KeyboardInterrupt:
        status.update(state="paused", epoch_current=0)
        return 130
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
