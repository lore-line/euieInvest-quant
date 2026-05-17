"""Sustained-winner ENTRY signal emission — Workstream C step 6.

Emits ENTRY signals to the trading-platform contract using the
Pareto-picked spec from `sustained_winner_joint_validate`.

Differs from `emit_quant_signals` in three ways:
  1. Source: `runs/{date}-sustained_winner_v1_g{NN}/joint_validation.parquet`
     and the sibling `rules.parquet`, filtered to rules with passes_all=True.
  2. Pattern name: `sw1_g{NN}_rule_{rule_id}` per server-team naming spec
     (PR #1 issuecomment-4469094953).
  3. Per-rule `expected_return_pct` (from the rule's `mean_endpoint_pct`)
     and `signal_strength` (from `ev_per_trade_pct / strength_divisor`),
     rather than the v1 module's spec-constant defaults.

Default spec: g06 — the Pareto pick where the highest-g spec clears all
4 aggregate gates (EV>0, median_gain >= g, win_rate >= 0.50, mean_hold
>= 20td).

Output: `euieInvest-reports/runs/{YYYY-MM-DD-NNN}/quant_signal_events.parquet`
+ `manifest.json`. Same contract v1 schema as `emit_quant_signals`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant.tracks.emit_quant_signals import (
    _CONTRACT_SCHEMA,
    _apply_dedup,
    _next_sequence,
    _validate_signal_row,
)
from quant.tracks.sustained_winner_label import SPECS, sweep_specs
from quant.tracks.sustained_winner_walkforward import _label_features_once
from quant.tracks.xgb_rule_extraction import (
    Condition,
    Rule,
    _build_condition_masks,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

CONTRACT_VERSION = "v1"
PIPELINE_STEP = "sustained_winner_signal_emission_v1"
DEFAULT_DEDUP_WINDOW_DAYS = 30
DEFAULT_SPEC = "g06"  # Pareto pick from sweep
DEFAULT_STRENGTH_DIVISOR = 10.0  # signal_strength = min(1.0, ev_per_trade_pct / 10.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument(
        "--spec", type=str, default=DEFAULT_SPEC,
        help=f"Spec name (e.g. 'g06'). Default: {DEFAULT_SPEC} (Pareto pick).",
    )
    p.add_argument(
        "--spec-dir", type=Path, default=None,
        help="Override spec run dir. Default: runs/{date_prefix}-sustained_winner_v1_{spec}/",
    )
    p.add_argument(
        "--date-prefix", type=str, default="2026-05-17",
        help="Date prefix for the sweep run dirs. Default 2026-05-17.",
    )
    p.add_argument(
        "--signal-date", type=date.fromisoformat, default=None,
        help="Date to emit signals for. Defaults to max date in features.parquet.",
    )
    p.add_argument(
        "--backfill-days", type=int, default=0,
        help="Emit signals for the past N market days IN ADDITION to signal_date.",
    )
    p.add_argument(
        "--dedup-window-days", type=int, default=DEFAULT_DEDUP_WINDOW_DAYS,
    )
    p.add_argument(
        "--strength-divisor", type=float, default=DEFAULT_STRENGTH_DIVISOR,
        help=f"signal_strength = min(1.0, ev_per_trade_pct / X). Default {DEFAULT_STRENGTH_DIVISOR}.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir. Default: runs/YYYY-MM-DD-NNN/ (auto sequence).",
    )
    p.add_argument(
        "--run-sequence", type=int, default=None,
    )
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
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


def _load_emit_rules(spec_dir: Path) -> tuple[list[Rule], list[int], dict[int, dict]]:
    """Load rules + per-rule joint-validation metrics. Filter to rules
    with passes_all == True.

    Returns (rules_filtered, original_rule_ids, per_rule_metrics).
    The returned `rules_filtered` are in the same order as `original_rule_ids`;
    `per_rule_metrics[original_rule_id]` is the joint-validation row.
    """
    rules_path = spec_dir / "rules.parquet"
    jv_path = spec_dir / "joint_validation.parquet"
    if not rules_path.exists() or not jv_path.exists():
        raise FileNotFoundError(
            f"missing rules.parquet or joint_validation.parquet in {spec_dir}"
        )

    rules_df = pl.read_parquet(rules_path)
    jv_df = pl.read_parquet(jv_path)
    # Recompute passes_all here so we don't depend on it being persisted
    # (it currently isn't — only the parquet has the raw metrics).
    # Pulling spec touch_threshold_pct from the spec_dir name avoids
    # threading the spec object through.
    spec_name = spec_dir.name.split("_v1_")[-1]
    spec = next(
        (s for s in sweep_specs() if s.name == spec_name),
        SPECS.get(spec_name),
    )
    if spec is None:
        raise ValueError(f"unknown spec '{spec_name}'")

    jv_df = jv_df.with_columns(
        passes_all=(
            (pl.col("ev_per_trade_pct") > 0.0)
            & (pl.col("median_endpoint_pct") >= spec.touch_threshold_pct)
            & (pl.col("win_rate") >= 0.50)
            & (pl.col("mean_hold_trading_days") >= 20)
        )
    )
    keep_ids = jv_df.filter(pl.col("passes_all")).get_column("rule_id").to_list()
    keep_set = set(keep_ids)

    rules_by_id: dict[int, Rule] = {}
    for row in rules_df.iter_rows(named=True):
        rid = int(row["rule_id"])
        if rid not in keep_set:
            continue
        conds_dicts = json.loads(row["conditions_json"])
        conds = tuple(
            Condition(
                feature=c["feature"], op=c["op"], threshold=float(c["threshold"])
            )
            for c in conds_dicts
        )
        rules_by_id[rid] = Rule(conditions=conds)

    metrics: dict[int, dict] = {}
    for row in jv_df.filter(pl.col("passes_all")).iter_rows(named=True):
        metrics[int(row["rule_id"])] = dict(row)

    ordered_ids = sorted(rules_by_id.keys())
    ordered_rules = [rules_by_id[i] for i in ordered_ids]
    return ordered_rules, ordered_ids, metrics


def _evaluate_rules_in_window(
    rules: list[Rule],
    rule_ids: list[int],
    features: pl.DataFrame,
) -> pl.DataFrame:
    """Return (symbol, date, rule_key) for every fire in `features`.

    Builds condition masks once and applies per-rule. Returns empty frame
    if no rules fire.
    """
    if not rules or features.height == 0:
        return pl.DataFrame(schema={
            "symbol": pl.String, "date": pl.Date, "rule_key": pl.String,
        })
    all_conds: set[Condition] = set()
    for r in rules:
        all_conds.update(r.conditions)
    masks = _build_condition_masks(features, list(all_conds))

    symbols = features["symbol"].to_numpy()
    dates = features["date"].to_numpy()

    rows: list[dict] = []
    for rule, rid in zip(rules, rule_ids):
        m = np.ones(features.height, dtype=bool)
        for cond in rule.conditions:
            m &= masks[cond]
        idx = m.nonzero()[0]
        rule_key = f"rule_{rid}"  # spec prefix added in _build_signal_rows
        for i in idx:
            rows.append({
                "symbol": str(symbols[i]),
                "date": dates[i].astype("datetime64[D]").astype("O"),
                "rule_key": rule_key,
            })
    if not rows:
        return pl.DataFrame(schema={
            "symbol": pl.String, "date": pl.Date, "rule_key": pl.String,
        })
    return pl.DataFrame(rows)


def _build_signal_rows(
    deduped: pl.DataFrame,
    rule_metrics: dict[int, dict],
    spec_name: str,
    horizon_days: int,
    strength_divisor: float,
    run_id: str,
    emission_dates: set[date],
) -> pl.DataFrame:
    """Convert deduped (symbol, date, rule_key) into contract-schema rows."""
    if deduped.height == 0:
        return pl.DataFrame(schema=_CONTRACT_SCHEMA)
    emission = deduped.filter(pl.col("date").is_in(list(emission_dates)))
    if emission.height == 0:
        return pl.DataFrame(schema=_CONTRACT_SCHEMA)

    rows = []
    for symbol, dt, rule_key in emission.iter_rows():
        rid = int(rule_key.split("_")[1])  # rule_key = "rule_{rid}"
        m = rule_metrics.get(rid)
        if m is None:
            continue
        # Pattern naming per server-team spec (issuecomment-4469094953):
        # sw1_g{NN}_rule_{rule_id}
        pattern = f"sw1_{spec_name}_rule_{rid}"
        ev = float(m["ev_per_trade_pct"])
        strength = min(1.0, max(0.0, ev / strength_divisor))
        # Conditions are threaded in via metrics["_conditions_dicts"]
        # (enriched from rules.parquet at load time in main()).
        conds_dicts = m.get("_conditions_dicts", [])
        rows.append({
            "signal_id": f"{run_id}_{symbol}_{dt.isoformat()}_ENTRY_{pattern}",
            "symbol": symbol,
            "signal_date": dt.isoformat(),
            "signal_type": "ENTRY",
            "signal_strength": strength,
            "pattern": pattern,
            "expected_horizon_days": int(horizon_days),
            "expected_return_pct": round(float(m["mean_endpoint_pct"]), 2),
            "conditions_json": json.dumps(conds_dicts),
        })
    return pl.DataFrame(rows, schema=_CONTRACT_SCHEMA)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    features_path = _resolve(args.features)
    if args.spec_dir is not None:
        spec_dir = _resolve(args.spec_dir)
    else:
        spec_dir = (
            _REPO_ROOT / "runs" / f"{args.date_prefix}-sustained_winner_v1_{args.spec}"
        )
        # Honor QUANT_RUNS_DIR via the Docker mount that maps it to /workspace/runs
        if not spec_dir.exists():
            alt = Path("/workspace/runs") / f"{args.date_prefix}-sustained_winner_v1_{args.spec}"
            if alt.exists():
                spec_dir = alt

    print(f"sustained-winner emit v{CONTRACT_VERSION}")
    print(f"  spec:      {args.spec}")
    print(f"  spec_dir:  {spec_dir}")
    print(f"  features:  {features_path}")

    spec_name = args.spec  # canonical, e.g. "g06"
    spec = next(
        (s for s in sweep_specs() if s.name == spec_name),
        SPECS.get(spec_name),
    )
    if spec is None:
        raise ValueError(f"unknown spec '{spec_name}'")
    horizon = spec.horizon_days  # 20

    print(f"  horizon:   {horizon}td  expected_return: per-rule (from joint validation)")

    # Load + filter rules
    rules, rule_ids, metrics = _load_emit_rules(spec_dir)
    print(f"  rules passing all gates: {len(rules):,}")

    # Enrich metrics with conditions_json from rules.parquet so emit can output them
    rules_df = pl.read_parquet(spec_dir / "rules.parquet")
    rules_id_to_conds = {
        int(r["rule_id"]): json.loads(r["conditions_json"])
        for r in rules_df.iter_rows(named=True)
    }
    for rid in metrics:
        metrics[rid]["_conditions_dicts"] = rules_id_to_conds.get(rid, [])

    # Load features + apply same dummies pipeline as training
    features_raw = pl.read_parquet(features_path)
    print(f"  loaded {features_raw.height:,} feature rows × {features_raw.width} cols")
    features = _label_features_once(features_raw, horizon)
    print(f"  with forward returns + dummies: {features.width} cols")

    # Resolve signal_date
    max_feature_date = features["date"].max()
    if isinstance(max_feature_date, str):
        max_feature_date = date.fromisoformat(max_feature_date)
    signal_date: date = args.signal_date or max_feature_date
    print(f"  signal_date: {signal_date}")

    # Emission window
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

    # Resolve run_id + out_dir
    run_date_str = signal_date.isoformat()
    runs_root = _REPO_ROOT / "runs"
    if not runs_root.exists():  # docker mount may put runs at /workspace/runs
        alt_runs = Path("/workspace/runs")
        if alt_runs.exists():
            runs_root = alt_runs
    seq = args.run_sequence if args.run_sequence is not None else _next_sequence(runs_root, run_date_str)
    run_id = f"{run_date_str}-{seq:03d}"
    out_dir = _resolve(args.out_dir) if args.out_dir is not None else (runs_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run_id: {run_id}")
    print(f"  out_dir: {out_dir}")

    # Eval window: include dedup lookback before emission window
    eval_start = min(emission_dates) - timedelta(days=args.dedup_window_days + 1)
    eval_features = features.filter(
        (pl.col("date") >= eval_start) & (pl.col("date") <= signal_date)
    )
    print(f"  eval window: {eval_start} .. {signal_date} ({eval_features.height:,} rows)")

    # Eval all rules; collect (symbol, date, rule_key) firings
    t_eval = time.perf_counter()
    raw_signals = _evaluate_rules_in_window(rules, rule_ids, eval_features)
    print(f"  raw firings: {raw_signals.height:,} ({time.perf_counter() - t_eval:.1f}s)")

    # Per-(symbol, rule_key) dedup
    deduped = _apply_dedup(raw_signals, args.dedup_window_days)
    print(f"  after {args.dedup_window_days}-day dedup: {deduped.height:,} (-{raw_signals.height - deduped.height:,})")

    # Build contract rows
    signals_df = _build_signal_rows(
        deduped, metrics, spec_name, horizon, args.strength_divisor, run_id, emission_dates,
    )
    print(f"  contract rows: {signals_df.height:,}")

    # Validate
    n_bad = 0
    for row in signals_df.iter_rows(named=True):
        valid, err = _validate_signal_row(row)
        if not valid:
            n_bad += 1
            if n_bad <= 5:
                print(f"  INVALID: {err}")
    if n_bad > 0:
        raise RuntimeError(f"{n_bad} invalid rows — fix before publish")

    # signal_id uniqueness
    n_unique = signals_df["signal_id"].n_unique() if signals_df.height else 0
    if signals_df.height and n_unique != signals_df.height:
        raise RuntimeError(
            f"signal_id uniqueness violated: {signals_df.height:,} rows but only {n_unique:,} unique IDs"
        )

    # Write parquet + manifest
    parquet_path = out_dir / "quant_signal_events.parquet"
    signals_df.write_parquet(parquet_path)
    print(f"  wrote {parquet_path}")

    manifest = {
        "run_id": run_id,
        "pipeline_step": PIPELINE_STEP,
        "contract_version": CONTRACT_VERSION,
        "spec_name": spec_name,
        "spec_touch_threshold_pct": spec.touch_threshold_pct,
        "spec_endpoint_threshold_pct": spec.endpoint_threshold_pct,
        "spec_horizon_days": horizon,
        "model_sha": f"sha256:{_file_sha256(spec_dir / 'rules.parquet')}",
        "git_commit_of_quant_repo": _git_head_sha(),
        "signal_date": signal_date.isoformat(),
        "emission_dates": sorted(d.isoformat() for d in emission_dates),
        "backfill_days": args.backfill_days,
        "dedup_window_days": args.dedup_window_days,
        "n_rules_passing_all_gates": len(rules),
        "n_raw_firings": int(raw_signals.height),
        "n_signals_after_dedup": int(deduped.height),
        "n_signals_emitted": int(signals_df.height),
        "n_signals_invalid": n_bad,
        "n_unique_symbols": int(signals_df["symbol"].n_unique()) if signals_df.height else 0,
        "n_unique_patterns": int(signals_df["pattern"].n_unique()) if signals_df.height else 0,
        "strength_divisor": args.strength_divisor,
        "spec_dir": str(spec_dir),
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {out_dir / 'manifest.json'}")

    print()
    print(f"=== SUSTAINED-WINNER EMISSION RESULT ({spec_name}) ===")
    print(f"  signals emitted:   {signals_df.height:,}")
    print(f"  emission dates:    {len(emission_dates)}")
    print(f"  unique symbols:    {manifest['n_unique_symbols']:,}")
    print(f"  unique patterns:   {manifest['n_unique_patterns']:,}")
    if signals_df.height:
        q25 = float(signals_df["signal_strength"].quantile(0.25))
        q50 = float(signals_df["signal_strength"].quantile(0.5))
        q75 = float(signals_df["signal_strength"].quantile(0.75))
        n_high = signals_df.filter(pl.col("signal_strength") >= 0.75).height
        print(f"  strength q25/q50/q75: {q25:.3f} / {q50:.3f} / {q75:.3f}")
        print(f"  >=0.75 (advisory): {n_high:,} ({100*n_high/signals_df.height:.1f}%)")
    print(f"  wall clock: {manifest['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
