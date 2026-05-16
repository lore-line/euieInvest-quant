"""Quant signal emission for the trading-platform integration.

Per `docs/quant-signal-contract-v1.md` on the trading-platform side
(lore-line/euieInvest@ac11d69), emit ENTRY signals as parquet that
lands in the platform's `quant_signal_events` table and flows into
Claude's daily analysis prompt as advisory context.

v1 is entry-only — the 2,003 walk-forward survivor rules from Phase B
re-emitted as ENTRY signal rows. EXIT signal design is a separate
modeling task (Stage 2; see PR #1 issuecomment-4467065115).

Per-(symbol, rule_key) 30-day dedup per the spec's "Same signal fires
every day for a week → Claude rating churns" failure-mode note.

Output: `euieInvest-reports/runs/{YYYY-MM-DD-NNN}/quant_signal_events.parquet`
+ `manifest.json`. The platform's `ingest-quant-signals.py` picks
these up on its 11:00 UTC cron.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from quant.tracks.walkforward_validate import Rule, _condition_expr, load_phase_a_rules

__all__ = ["main", "parse_args"]

_REPO_ROOT = Path(__file__).resolve().parents[3]

CONTRACT_VERSION = "v1"
PIPELINE_STEP = "quant_signal_emission"
DEFAULT_DEDUP_WINDOW_DAYS = 30  # per server-team failure-mode note
DEFAULT_EXPECTED_HORIZON_DAYS = 30  # per contract spec
DEFAULT_EXPECTED_RETURN_PCT = 20.0  # per contract spec
LIFT_STRENGTH_DIVISOR = 3.0  # signal_strength = min(1.0, lift / 3.0)
ENTRY_DEFAULT_TRAIN_END = "2026-04-01"  # phase B v3 walk-forward train cutoff


_CONTRACT_SCHEMA: dict[str, pl.DataType] = {
    "signal_id": pl.String,
    "symbol": pl.String,
    "signal_date": pl.String,
    "signal_type": pl.String,
    "signal_strength": pl.Float64,
    "pattern": pl.String,
    "expected_horizon_days": pl.Int64,
    "expected_return_pct": pl.Float64,
    "conditions_json": pl.String,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features", type=Path,
        default=Path("data/features/features.parquet"),
        help="Labeled feature matrix (where rules evaluate).",
    )
    p.add_argument(
        "--walkforward-aggregate", type=Path,
        default=Path("runs/2026-05-14-step4_walkforward_validation/rule-validation-aggregate.parquet"),
        help="Source of is_walk_forward_survivor + mean_val_lift per rule.",
    )
    p.add_argument(
        "--track1-dir", type=Path,
        default=Path("runs/2026-05-13-step3a_xgb_rule_extraction"),
    )
    p.add_argument(
        "--track4-dir", type=Path,
        default=Path("runs/2026-05-13-step3c_multi_label_rules"),
    )
    p.add_argument(
        "--track5-dir", type=Path,
        default=Path("runs/2026-05-13-step3d_per_regime_rules"),
    )
    p.add_argument(
        "--signal-date", type=date.fromisoformat, default=None,
        help="Date to emit signals for. Defaults to max date in features.parquet.",
    )
    p.add_argument(
        "--backfill-days", type=int, default=0,
        help="Emit signals for the past N market days IN ADDITION to signal_date. "
             "Use 7 for first-publish to fill the platform's 7-day prepare-context window.",
    )
    p.add_argument(
        "--dedup-window-days", type=int, default=DEFAULT_DEDUP_WINDOW_DAYS,
        help="Per-(symbol, rule_key) dedup window. A rule that fires N consecutive days "
             "only emits a signal on the FIRST day within this window.",
    )
    p.add_argument(
        "--min-val-lift", type=float, default=1.5,
        help="Drop survivor rules with mean_val_lift below this threshold.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Defaults to runs/YYYY-MM-DD-NNN/ where NNN is "
             "auto-incremented from existing dirs on the same date.",
    )
    p.add_argument(
        "--run-sequence", type=int, default=None,
        help="Override sequence number for run_id. Default: auto-detect next.",
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


def _next_sequence(runs_root: Path, run_date_str: str) -> int:
    """Auto-increment sequence for runs/{date}-NNN/ pattern (no step suffix
    per the contract spec example `2026-05-16-001`)."""
    if not runs_root.exists():
        return 1
    used: list[int] = []
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        # Match exactly YYYY-MM-DD-NNN (4 dash-separated parts, no step name).
        parts = d.name.split("-")
        if len(parts) != 4:
            continue
        if "-".join(parts[:3]) != run_date_str:
            continue
        try:
            used.append(int(parts[3]))
        except ValueError:
            continue
    return max(used, default=0) + 1


def _load_survivor_rules(
    aggregate_path: Path,
    track1_dir: Path,
    track4_dir: Path,
    track5_dir: Path,
    min_val_lift: float,
) -> tuple[list[Rule], dict[str, float]]:
    """Load walk-forward survivor rules + their mean_val_lift map."""
    agg = pl.read_parquet(aggregate_path)
    survivors = (
        agg.filter(pl.col("is_walk_forward_survivor"))
        .filter(pl.col("mean_val_lift") >= min_val_lift)
    )
    survivor_keys = set(survivors["rule_key"].to_list())
    survivor_lift = dict(
        zip(survivors["rule_key"].to_list(), survivors["mean_val_lift"].to_list())
    )
    all_rules = load_phase_a_rules(track1_dir, track4_dir, track5_dir)
    rules = [r for r in all_rules if r.rule_key in survivor_keys]
    return rules, survivor_lift


def _evaluate_rule_firings(rule: Rule, features: pl.DataFrame) -> pl.DataFrame:
    """Returns (symbol, date) rows where all of `rule.conditions` hold.
    Empty frame if any condition references a missing feature column."""
    expr = None
    for cond in rule.conditions:
        if cond["feature"] not in features.columns:
            return pl.DataFrame(schema={"symbol": pl.String, "date": pl.Date})
        e = _condition_expr(cond)
        expr = e if expr is None else (expr & e)
    if expr is None:
        return pl.DataFrame(schema={"symbol": pl.String, "date": pl.Date})
    return features.filter(expr).select(["symbol", "date"])


def _apply_dedup(
    raw_signals: pl.DataFrame, dedup_window_days: int
) -> pl.DataFrame:
    """For each (symbol, rule_key) group, emit a row only if the most-recent
    EMITTED signal in that group is more than `dedup_window_days` ago.

    The state is the LAST EMITTED row, not the previous row in the sorted
    sequence — a rule firing every day for 60 days emits on day 1 and day
    32 (NOT day 2 because day 2's prev-emit is day 1, gap=1, suppress;
    NOT day 31 because gap to day-1-emit is 30, not > 30; day 32 has
    gap=31, emit, and now the suppression window starts at day 32).

    The input must contain columns (symbol, date, rule_key).
    """
    if raw_signals.height == 0:
        return raw_signals
    # Iterative because the suppression state depends on the dedup's own
    # output — a polars window function over the raw input doesn't have
    # access to that. ~O(N) in row count; fast enough at typical 10-100K
    # candidate-signal scale.
    sorted_df = raw_signals.sort(["symbol", "rule_key", "date"])
    last_emit_by_group: dict[tuple[str, str], date] = {}
    kept: list[dict] = []
    for row in sorted_df.iter_rows(named=True):
        group = (row["symbol"], row["rule_key"])
        last = last_emit_by_group.get(group)
        if last is None or (row["date"] - last).days > dedup_window_days:
            kept.append(row)
            last_emit_by_group[group] = row["date"]
    if not kept:
        return pl.DataFrame(schema=raw_signals.schema)
    return pl.DataFrame(kept, schema=raw_signals.schema)


def _build_signal_rows(
    deduped: pl.DataFrame,
    survivor_lift: dict[str, float],
    rule_definitions: dict[str, Rule],
    run_id: str,
    emission_dates: set[date],
) -> pl.DataFrame:
    """Convert deduped (symbol, date, rule_key) rows into contract-schema rows.

    Filters to emission_dates (the actual publish window — dedup needed a
    wider lookback but only signals on emission_dates ship)."""
    if deduped.height == 0:
        return pl.DataFrame(schema=_CONTRACT_SCHEMA)
    emission = deduped.filter(pl.col("date").is_in(list(emission_dates)))
    if emission.height == 0:
        return pl.DataFrame(schema=_CONTRACT_SCHEMA)
    rows = []
    for symbol, dt, rule_key in emission.iter_rows():
        rule = rule_definitions.get(rule_key)
        if rule is None:
            continue
        lift = survivor_lift.get(rule_key, 0.0)
        strength = min(1.0, lift / LIFT_STRENGTH_DIVISOR)
        # signal_id format extends the spec's recommendation with `pattern` to
        # preserve uniqueness when multiple rules fire on the same (symbol, date).
        # Spec example `{run_id}_{symbol}_{signal_date}_{signal_type}` collides
        # under the "one row per (rule × symbol × date)" cardinality rule.
        rows.append({
            "signal_id": f"{run_id}_{symbol}_{dt.isoformat()}_ENTRY_{rule_key}",
            "symbol": symbol,
            "signal_date": dt.isoformat(),
            "signal_type": "ENTRY",
            "signal_strength": strength,
            "pattern": rule_key,
            "expected_horizon_days": DEFAULT_EXPECTED_HORIZON_DAYS,
            "expected_return_pct": DEFAULT_EXPECTED_RETURN_PCT,
            "conditions_json": json.dumps(rule.conditions),
        })
    return pl.DataFrame(rows, schema=_CONTRACT_SCHEMA)


def _validate_signal_row(row: dict) -> tuple[bool, str | None]:
    """Mirror of the platform-side `ingest-quant-signals.py` validation."""
    if not (0.0 <= row["signal_strength"] <= 1.0):
        return False, f"signal_strength={row['signal_strength']} out of [0, 1]"
    if row["signal_type"] not in ("ENTRY", "EXIT"):
        return False, f"signal_type={row['signal_type']} not in (ENTRY, EXIT)"
    if not row.get("signal_id"):
        return False, "signal_id is empty"
    try:
        date.fromisoformat(row["signal_date"])
    except (ValueError, TypeError):
        return False, f"signal_date={row['signal_date']!r} not ISO YYYY-MM-DD"
    if not row.get("symbol"):
        return False, "symbol is empty"
    return True, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    features_path = _resolve_path(args.features)
    aggregate_path = _resolve_path(args.walkforward_aggregate)
    track1_dir = _resolve_path(args.track1_dir)
    track4_dir = _resolve_path(args.track4_dir)
    track5_dir = _resolve_path(args.track5_dir)

    print(f"quant signal emission v{CONTRACT_VERSION} (entry-only)")
    print(f"  features: {features_path.relative_to(_REPO_ROOT)}")
    features = pl.read_parquet(features_path)
    print(f"  loaded {features.height:,} feature rows × {features.width} cols")

    # Resolve signal_date (default: max date in features)
    max_feature_date = features["date"].max()
    if isinstance(max_feature_date, str):
        max_feature_date = date.fromisoformat(max_feature_date)
    signal_date: date = args.signal_date or max_feature_date
    print(f"  signal_date: {signal_date} (max features.date: {max_feature_date})")

    # Resolve emission window (signal_date + backfill_days)
    if args.backfill_days > 0:
        recent_dates = (
            features.filter(pl.col("date") <= signal_date)["date"]
            .unique()
            .sort(descending=True)
            .to_list()
        )
        emission_dates = set(recent_dates[: args.backfill_days + 1])
        print(f"  backfill: {args.backfill_days}d → emitting for {len(emission_dates)} dates")
    else:
        emission_dates = {signal_date}

    # Resolve run_id + out_dir (strict YYYY-MM-DD-NNN format per contract spec)
    run_date_str = signal_date.isoformat()
    runs_root = _REPO_ROOT / "runs"
    sequence = args.run_sequence if args.run_sequence is not None else _next_sequence(runs_root, run_date_str)
    run_id = f"{run_date_str}-{sequence:03d}"
    out_dir = _resolve_path(args.out_dir) if args.out_dir is not None else (runs_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run_id: {run_id}")
    print(f"  out_dir: {out_dir.relative_to(_REPO_ROOT)}")

    # Load survivor rules
    print(f"loading walk-forward survivor rules ...")
    rules, survivor_lift = _load_survivor_rules(
        aggregate_path, track1_dir, track4_dir, track5_dir, args.min_val_lift
    )
    print(f"  {len(rules):,} survivor rules (min_val_lift ≥ {args.min_val_lift})")
    rule_definitions = {r.rule_key: r for r in rules}

    # Eval window includes dedup lookback so we can correctly dedup against
    # firings BEFORE the emission window
    eval_start = min(emission_dates) - timedelta(days=args.dedup_window_days + 1)
    eval_features = features.filter(
        (pl.col("date") >= eval_start) & (pl.col("date") <= signal_date)
    )
    print(f"  eval window: {eval_start} .. {signal_date} ({eval_features.height:,} rows)")

    # Evaluate every rule, accumulate firings
    print(f"evaluating rules ...")
    t_eval = time.perf_counter()
    per_rule_frames: list[pl.DataFrame] = []
    skipped = 0
    for i, rule in enumerate(rules):
        firings = _evaluate_rule_firings(rule, eval_features)
        if firings.height == 0:
            skipped += 1
            continue
        per_rule_frames.append(firings.with_columns(pl.lit(rule.rule_key).alias("rule_key")))
        if (i + 1) % 500 == 0:
            n_rows = sum(f.height for f in per_rule_frames)
            print(f"  {i+1:,}/{len(rules):,} rules ({n_rows:,} raw firings, {time.perf_counter() - t_eval:.1f}s)")
    if per_rule_frames:
        raw_signals = pl.concat(per_rule_frames)
    else:
        raw_signals = pl.DataFrame(schema={"symbol": pl.String, "date": pl.Date, "rule_key": pl.String})
    print(f"  raw firings: {raw_signals.height:,} (skipped {skipped} rules with no firings)")

    # Per-(symbol, rule_key) 30-day dedup
    print(f"applying {args.dedup_window_days}-day dedup per (symbol, rule_key) ...")
    deduped = _apply_dedup(raw_signals, args.dedup_window_days)
    print(f"  after dedup: {deduped.height:,} (-{raw_signals.height - deduped.height:,})")

    # Build contract-schema rows for emission window
    print(f"building contract rows ...")
    signals_df = _build_signal_rows(deduped, survivor_lift, rule_definitions, run_id, emission_dates)
    print(f"  contract rows: {signals_df.height:,}")

    # Per-row validation (mirrors platform-side ingest)
    print(f"validating rows ...")
    n_bad = 0
    for row in signals_df.iter_rows(named=True):
        valid, err = _validate_signal_row(row)
        if not valid:
            n_bad += 1
            if n_bad <= 5:
                print(f"  INVALID: {err}")
    if n_bad > 0:
        raise RuntimeError(f"{n_bad} invalid rows — fix before publish")
    print(f"  all {signals_df.height:,} rows valid")

    # Uniqueness check on signal_id (contract requires global uniqueness)
    n_unique_ids = signals_df["signal_id"].n_unique() if signals_df.height else 0
    if n_unique_ids != signals_df.height:
        raise RuntimeError(
            f"signal_id uniqueness violated: {signals_df.height:,} rows but only "
            f"{n_unique_ids:,} unique IDs. The generation formula must be reviewed."
        )

    # Write parquet
    parquet_path = out_dir / "quant_signal_events.parquet"
    signals_df.write_parquet(parquet_path)
    print(f"  wrote {parquet_path.relative_to(_REPO_ROOT)}")

    # Write manifest per contract spec (model_sha + git_commit_of_quant_repo
    # are the FK targets in ml_runs upsert)
    manifest = {
        "run_id": run_id,
        "pipeline_step": PIPELINE_STEP,
        "contract_version": CONTRACT_VERSION,
        "model_sha": f"sha256:{_file_sha256(aggregate_path)}",
        "git_commit_of_quant_repo": _git_head_sha(),
        "feature_count": features.width,
        "train_end": ENTRY_DEFAULT_TRAIN_END,
        "holdout_end": signal_date.isoformat(),
        "signal_date": signal_date.isoformat(),
        "emission_dates": sorted(d.isoformat() for d in emission_dates),
        "backfill_days": args.backfill_days,
        "dedup_window_days": args.dedup_window_days,
        "min_val_lift": args.min_val_lift,
        "n_survivor_rules_loaded": len(rules),
        "n_raw_firings": int(raw_signals.height),
        "n_signals_after_dedup": int(deduped.height),
        "n_signals_emitted": int(signals_df.height),
        "n_signals_invalid": n_bad,
        "n_unique_symbols": int(signals_df["symbol"].n_unique()) if signals_df.height else 0,
        "n_unique_patterns": int(signals_df["pattern"].n_unique()) if signals_df.height else 0,
        "walkforward_aggregate_path": str(aggregate_path.relative_to(_REPO_ROOT)),
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {manifest_path.relative_to(_REPO_ROOT)}")

    # Final summary print
    print(f"\n=== QUANT SIGNAL EMISSION RESULT ===")
    print(f"  signals emitted:    {signals_df.height:,}")
    print(f"  emission dates:     {len(emission_dates)} day(s)")
    print(f"  unique symbols:     {manifest['n_unique_symbols']:,}")
    print(f"  unique patterns:    {manifest['n_unique_patterns']:,}")
    if signals_df.height:
        q25 = float(signals_df["signal_strength"].quantile(0.25))
        q50 = float(signals_df["signal_strength"].quantile(0.5))
        q75 = float(signals_df["signal_strength"].quantile(0.75))
        n_above_threshold = signals_df.filter(pl.col("signal_strength") >= 0.75).height
        print(f"  strength quartiles: {q25:.3f} / {q50:.3f} / {q75:.3f}")
        print(f"  ≥0.75 (advisory):   {n_above_threshold:,} ({n_above_threshold/signals_df.height*100:.1f}%)")
    print(f"  wall clock:         {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
