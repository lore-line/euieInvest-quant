"""Emit breakout_seq_v1 ENTRY signals via the trading-platform contract.

For each (symbol, date) in the emission window:
  1. Build a 60-day OHLCV window ending at `date`
  2. Forward through trained BreakoutSeqCNN → sigmoid score
  3. If score >= τ (the Pareto-picked threshold from joint_validate) → emit
  4. Per-(symbol, pattern) 30-day dedup
  5. signal_strength = clamp(score, 0, 1) — model probability IS the strength

Pattern: `bsq60_g20` (single-rule signal source — server-team spec allows
`bsq60_g20_rule_{id}` but CNN doesn't produce discrete rules; using
single pattern for v1, score-bucket rules deferred to v2).

expected_return_pct: from joint_validation_summary.json (mean realized
gain at the chosen threshold). Capped per server-team direction (PR #1
issuecomment-4469414710).
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
import torch

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.tracks.breakout_seq_label import PATTERN_PREFIX, SPEC_DEFAULT
from quant.tracks.breakout_seq_model import BreakoutSeqCNN
from quant.tracks.breakout_seq_train import WindowedFeatureDataset
from quant.tracks.emit_quant_signals import (
    _CONTRACT_SCHEMA,
    _apply_dedup,
    _next_sequence,
    _validate_signal_row,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

CONTRACT_VERSION = "v1"
PIPELINE_STEP = "breakout_seq_signal_emission_v1"
DEFAULT_DEDUP_WINDOW_DAYS = 30
DEFAULT_EXPECTED_RETURN_CAP_PCT = 30.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to runs/{date}-breakout_seq_v1_g20/ (must contain "
             "model.pt + joint_validation_summary.json).",
    )
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--signal-date", type=date.fromisoformat, default=None)
    p.add_argument("--backfill-days", type=int, default=0)
    p.add_argument("--dedup-window-days", type=int, default=DEFAULT_DEDUP_WINDOW_DAYS)
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Override the threshold τ. Default: from joint_validation_summary.pareto_pick.threshold.",
    )
    p.add_argument(
        "--expected-return-cap-pct", type=float, default=DEFAULT_EXPECTED_RETURN_CAP_PCT,
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-sequence", type=int, default=None)
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    run_dir = _resolve(args.run_dir)
    model_path = run_dir / "model.pt"
    jv_summary_path = run_dir / "joint_validation_summary.json"
    if not model_path.exists() or not jv_summary_path.exists():
        print(f"ERROR: model.pt or joint_validation_summary.json missing in {run_dir}")
        return 1

    with open(jv_summary_path) as f:
        jv_summary = json.load(f)
    pareto = jv_summary.get("pareto_pick")
    if args.threshold is not None:
        threshold = args.threshold
        per_rule_expected = None
        # Look up the per-threshold metrics if available
        for r in jv_summary.get("per_threshold", []):
            if abs(r.get("threshold", -1) - threshold) < 1e-9:
                per_rule_expected = r.get("mean_endpoint_pct")
                break
    elif pareto:
        threshold = pareto["threshold"]
        per_rule_expected = None
        for r in jv_summary.get("per_threshold", []):
            if abs(r.get("threshold", -1) - threshold) < 1e-9:
                per_rule_expected = r.get("mean_endpoint_pct")
                break
    else:
        print("ERROR: no pareto_pick in joint_validation_summary.json; pass --threshold explicitly")
        return 1
    if per_rule_expected is None:
        per_rule_expected = 0.0
    capped_expected = min(per_rule_expected, args.expected_return_cap_pct)

    spec = SPEC_DEFAULT
    pattern = f"{PATTERN_PREFIX}_{spec.name}"  # bsq60_g20
    print(f"emit_breakout_seq_signals v{CONTRACT_VERSION}")
    print(f"  run_dir:           {run_dir}")
    print(f"  threshold τ:       {threshold}")
    print(f"  pattern:           {pattern}")
    print(f"  expected_return:   {capped_expected:.2f}% (raw={per_rule_expected:.2f}, cap={args.expected_return_cap_pct})")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreakoutSeqCNN().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model:             BreakoutSeqCNN, {n_params/1e3:.1f}K params on {device}")

    # Features + emission window
    features_path = _resolve(args.features)
    features = pl.read_parquet(features_path)
    print(f"  features:          {features.height:,} × {features.width}")

    # signal_date defaults to max date in features
    max_date = features["date"].max()
    if isinstance(max_date, str):
        max_date = date.fromisoformat(max_date)
    signal_date = args.signal_date or max_date
    print(f"  signal_date:       {signal_date}")

    # Emission window
    sorted_dates = features.filter(pl.col("date") <= signal_date)["date"].unique().sort(descending=True).to_list()
    if args.backfill_days > 0:
        emission_dates = set(sorted_dates[: args.backfill_days + 1])
    else:
        emission_dates = {signal_date}
    # Eval window must include enough HISTORY for the 60-trading-day WindowIndex
    # (each candidate window needs 59 prior days of OHLCV). 60 trading days
    # ≈ 90 calendar days; add dedup lookback as buffer.
    history_calendar_days = int(WINDOW * 1.5) + args.dedup_window_days + 5
    eval_start_date = min(emission_dates) - timedelta(days=history_calendar_days)
    eval_features = features.filter(
        (pl.col("date") >= eval_start_date) & (pl.col("date") <= signal_date)
    )
    print(f"  eval window:       {eval_start_date} .. {signal_date} ({eval_features.height:,} rows)")
    print(f"  emission dates:    {len(emission_dates)}")

    # Build a WindowIndex for the eval window. WindowIndex hardcodes is_winner;
    # dummy-add it.
    eval_features = eval_features.with_columns(pl.lit(0).cast(pl.Int8).alias("is_winner"))
    idx = build_window_index(eval_features)
    print(f"  candidates:        {idx.n_windows:,} valid 60-day windows")
    if idx.n_windows == 0:
        print("  no valid windows in eval — nothing to emit")
        return 0

    # Score all candidates
    print(f"  scoring with CNN (batch={args.batch_size}) ...")
    t = time.perf_counter()
    ds = WindowedFeatureDataset(idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            s = torch.sigmoid(model(xb)).cpu().numpy()
            all_scores.append(s)
    scores = np.concatenate(all_scores)
    print(f"    scored {len(scores):,} candidates in {time.perf_counter()-t:.1f}s")

    # Build (symbol, date, score) frame
    symbols_arr = np.array([idx.symbols[s] for s in idx.endpoints[:, 0]])
    dates_arr = idx.dates.astype("datetime64[D]").astype("O")
    cand_df = pl.DataFrame({
        "symbol": symbols_arr.tolist(),
        "date": dates_arr.tolist(),
        "score": scores.tolist(),
    }).with_columns(pl.col("date").cast(pl.Date))

    # Filter by threshold, then add rule_key for dedup
    fires = cand_df.filter(pl.col("score") >= threshold).with_columns(
        pl.lit(pattern).alias("rule_key")
    ).select(["symbol", "date", "rule_key", "score"])
    print(f"  fires (score >= {threshold}): {fires.height:,}")

    # Dedup
    deduped = _apply_dedup(
        fires.select(["symbol", "date", "rule_key"]),
        args.dedup_window_days,
    )
    # Re-join score onto deduped (only takes the FIRST emit per (sym, rule_key, date))
    deduped = deduped.join(
        fires.select(["symbol", "date", "rule_key", "score"]),
        on=["symbol", "date", "rule_key"], how="left",
    )
    print(f"  after {args.dedup_window_days}-day dedup: {deduped.height:,} (-{fires.height - deduped.height:,})")

    # Filter to emission window
    emission = deduped.filter(pl.col("date").is_in(list(emission_dates)))
    print(f"  in emission window: {emission.height:,}")

    # Build contract rows
    run_date_str = signal_date.isoformat()
    runs_root = _REPO_ROOT / "runs"
    if not runs_root.exists():
        alt = Path("/workspace/runs")
        if alt.exists():
            runs_root = alt
    seq = args.run_sequence if args.run_sequence is not None else _next_sequence(runs_root, run_date_str)
    run_id = f"{run_date_str}-{seq:03d}"
    out_dir = _resolve(args.out_dir) if args.out_dir else (runs_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run_id:            {run_id}")
    print(f"  out_dir:           {out_dir}")

    rows = []
    for row in emission.iter_rows(named=True):
        score = float(row["score"])
        strength = max(0.0, min(1.0, score))
        rows.append({
            "signal_id": f"{run_id}_{row['symbol']}_{row['date'].isoformat()}_ENTRY_{pattern}",
            "symbol": row["symbol"],
            "signal_date": row["date"].isoformat(),
            "signal_type": "ENTRY",
            "signal_strength": strength,
            "pattern": pattern,
            "expected_horizon_days": int(spec.horizon_days),
            "expected_return_pct": round(capped_expected, 2),
            "conditions_json": json.dumps([{
                "signal_type": "cnn_score",
                "model": "BreakoutSeqCNN",
                "threshold": threshold,
                "score": round(score, 4),
            }]),
        })
    signals_df = pl.DataFrame(rows, schema=_CONTRACT_SCHEMA)

    # Validate
    n_bad = 0
    for r in signals_df.iter_rows(named=True):
        valid, err = _validate_signal_row(r)
        if not valid:
            n_bad += 1
            if n_bad <= 5:
                print(f"  INVALID: {err}")
    if n_bad > 0:
        raise RuntimeError(f"{n_bad} invalid rows")
    if signals_df.height and signals_df["signal_id"].n_unique() != signals_df.height:
        raise RuntimeError("signal_id uniqueness violated")

    signals_df.write_parquet(out_dir / "quant_signal_events.parquet")
    manifest = {
        "run_id": run_id,
        "pipeline_step": PIPELINE_STEP,
        "contract_version": CONTRACT_VERSION,
        "spec_name": spec.name,
        "spec_touch_threshold_pct": spec.touch_threshold_pct,
        "spec_horizon_days": spec.horizon_days,
        "pattern": pattern,
        "decision_threshold": threshold,
        "expected_return_pct": capped_expected,
        "expected_return_cap_pct": args.expected_return_cap_pct,
        "model_sha": f"sha256:{_file_sha256(model_path)}",
        "git_commit_of_quant_repo": _git_head_sha(),
        "signal_date": signal_date.isoformat(),
        "emission_dates": sorted(d.isoformat() for d in emission_dates),
        "backfill_days": args.backfill_days,
        "dedup_window_days": args.dedup_window_days,
        "n_candidates_scored": int(idx.n_windows),
        "n_fires_raw": int(fires.height),
        "n_signals_after_dedup": int(deduped.height),
        "n_signals_emitted": int(signals_df.height),
        "n_signals_invalid": n_bad,
        "n_unique_symbols": int(signals_df["symbol"].n_unique()) if signals_df.height else 0,
        "notes": [
            f"breakout_seq_v1 — 1D CNN on 60d pre-entry sequences. "
            f"Decision threshold τ={threshold} from joint_validation Pareto pick. "
            f"Single-pattern emit (no per-rule decomposition for v1).",
        ],
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== BREAKOUT_SEQ EMISSION RESULT ===")
    print(f"  signals emitted:   {signals_df.height:,}")
    print(f"  emission dates:    {len(emission_dates)}")
    print(f"  unique symbols:    {manifest['n_unique_symbols']:,}")
    if signals_df.height:
        q25 = float(signals_df["signal_strength"].quantile(0.25))
        q50 = float(signals_df["signal_strength"].quantile(0.5))
        q75 = float(signals_df["signal_strength"].quantile(0.75))
        n_high = signals_df.filter(pl.col("signal_strength") >= 0.75).height
        print(f"  strength q25/q50/q75: {q25:.3f} / {q50:.3f} / {q75:.3f}")
        print(f"  >=0.75 advisory: {n_high:,} ({100*n_high/signals_df.height:.1f}%)")
    print(f"  wall clock:        {manifest['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
