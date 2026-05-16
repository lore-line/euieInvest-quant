"""Stage 2 of the quant signal contract — exit-signal variant sweep.

For each of the 4 top-prediction label variants (A/B/C/D), train an
XGB binary classifier and joint-validate against the Stage 1 ENTRY
signal pool. Surface a side-by-side comparison table so the server
team can pick the variant that clears all three joint-validation
gates with the cleanest hold-duration tail.

Server-team direction (PR #1 issuecomment-4467135126):
- A/B/C/D sweep — don't preselect D
- 60d EXIT eligibility post-ENTRY
- Joint-validation hard cap aligned to 60d (consistency with emission)
- Report **mean + median + P25 hold** per variant (TFSA safe-harbor
  matters at the left tail, not just the average)
- Report EXIT-coverage-of-ENTRY-pool % alongside gate metrics

Gates (per contract spec):
- Mean realized hold ≥ 30 trading days (TFSA business-income safe-harbor)
- Median realized gain ≥ +20%
- Win rate ≥ 50%

Output: runs/{date}-exit_variant_sweep/{variant_id}/ for each variant
+ a top-level comparison.parquet + comparison.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from quant.tracks.emit_quant_signals import (
    _apply_dedup as _entry_apply_dedup,
    _evaluate_rule_firings,
    _load_survivor_rules,
)
from quant.tracks.exit_label import (
    VARIANTS,
    ExitVariant,
    label_statistics,
    prepare_label_set,
)
from quant.tracks.xgb_rule_extraction import _NON_FEATURE_COLS, _replay_feature_selection

__all__ = ["main", "parse_args", "VariantResult"]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Server-team default per the spec + their joint-validation alignment note.
DEFAULT_EXIT_ELIGIBILITY_DAYS = 60
DEFAULT_JOINT_VALIDATION_HARD_CAP_DAYS = 60  # matches emission for consistency
DEFAULT_TRAIN_CUTOFF = date(2024, 12, 31)  # train ≤ this, holdout >
DEFAULT_HOLDOUT_END = date(2026, 3, 30)  # match Phase B v3 sleeve
DEFAULT_EXIT_SCORE_PERCENTILE = 0.10  # top N% of XGB scores per variant become EXIT predictions
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "verbosity": 0,
}
DEFAULT_N_ROUNDS = 200


@dataclass
class VariantResult:
    """Joint-validation metrics for one variant, used for the comparison table."""

    variant_id: str
    variant_gain_threshold_pct: float
    variant_forward_window_days: int
    variant_giveback_threshold_pct: float

    # Label / training diagnostics
    n_qualified_positions: int
    n_positive_labels: int
    label_positive_rate: float
    xgb_train_auc: float
    xgb_val_auc: float

    # Joint validation against ENTRY pool
    n_entries: int
    n_entries_with_exit: int
    n_entries_hit_hard_cap: int
    exit_coverage_pct: float

    # Hold duration (trading days)
    mean_hold_days: float
    median_hold_days: float
    p25_hold_days: float
    p75_hold_days: float
    min_hold_days: int
    max_hold_days: int

    # Realized gain (percent)
    mean_realized_gain_pct: float
    median_realized_gain_pct: float
    p25_realized_gain_pct: float
    p75_realized_gain_pct: float
    win_rate: float
    max_drawdown_pct: float

    # Gate clearance
    gate_mean_hold_passes: bool  # ≥30 trading days
    gate_median_gain_passes: bool  # ≥+20%
    gate_win_rate_passes: bool  # ≥50%
    all_gates_pass: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features", type=Path,
        default=Path("data/features/features.parquet"),
    )
    p.add_argument(
        "--walkforward-aggregate", type=Path,
        default=Path("runs/2026-05-14-step4_walkforward_validation/rule-validation-aggregate.parquet"),
        help="Used to regenerate historical entry signals across the holdout window.",
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
    p.add_argument(
        "--entry-min-val-lift", type=float, default=1.5,
        help="Minimum mean_val_lift on survivor rules for entry-signal generation.",
    )
    p.add_argument(
        "--entry-dedup-days", type=int, default=30,
        help="Per-(symbol, rule_key) dedup window for historical entry generation.",
    )
    p.add_argument(
        "--entry-min-strength", type=float, default=0.75,
        help="Joint-validate only against entries with signal_strength ≥ this. "
             "Default 0.75 matches the platform's advisory threshold — these are "
             "the entries Claude will actually weight in conviction bumps.",
    )
    p.add_argument(
        "--entry-cluster-membership", type=Path,
        default=Path("runs/2026-05-14-step4_walkforward_cluster_id/cluster-membership-walkforward.parquet"),
        help="Path to cluster-membership parquet. Used together with "
             "--entry-cluster-id to restrict entries to the v1-encoder "
             "walk-forward cluster-of-interest universe (Phase B v3's +28%% / "
             "Sharpe 1.88 deployment baseline).",
    )
    p.add_argument(
        "--entry-cluster-id", type=int, default=None,
        help="Filter entries to (symbol, date) pairs in this cluster. None = no "
             "cluster filter (Stage 2.0 baseline). 8 = v1 WF cluster-of-interest "
             "(Stage 2.1 finding — tighter entries to address the gain gate).",
    )
    p.add_argument(
        "--train-cutoff", type=date.fromisoformat, default=DEFAULT_TRAIN_CUTOFF,
        help="Chronological split: train ≤ cutoff, validate > cutoff.",
    )
    p.add_argument(
        "--holdout-end", type=date.fromisoformat, default=DEFAULT_HOLDOUT_END,
        help="Holdout window upper bound for joint validation.",
    )
    p.add_argument(
        "--exit-eligibility-days", type=int, default=DEFAULT_EXIT_ELIGIBILITY_DAYS,
        help="Symbol must have had an ENTRY within this many days to be exit-eligible.",
    )
    p.add_argument(
        "--joint-cap-days", type=int, default=DEFAULT_JOINT_VALIDATION_HARD_CAP_DAYS,
        help="Hard-cap on hold duration in joint-validation simulation.",
    )
    p.add_argument(
        "--exit-score-percentile", type=float, default=DEFAULT_EXIT_SCORE_PERCENTILE,
        help="Top N percentile of XGB scores become EXIT predictions. "
             "0.10 = top 10%% of scored holdout rows fire as exits. Auto-calibrates "
             "per variant so cross-variant comparison is fair (each variant emits the "
             "same fraction of its qualified-position pool as EXITs).",
    )
    p.add_argument(
        "--n-rounds", type=int, default=DEFAULT_N_ROUNDS,
        help="XGB boosting rounds.",
    )
    p.add_argument(
        "--variants", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Subset of variants to evaluate.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _generate_historical_entries(
    features: pl.DataFrame,
    aggregate_path: Path,
    track1_dir: Path,
    track4_dir: Path,
    track5_dir: Path,
    min_val_lift: float,
    dedup_days: int,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Regenerate the Stage 1 ENTRY-signal pipeline over the [start, end]
    range so joint validation has entries with full 60d forward window.

    Mirrors emit_quant_signals._evaluate_rule_firings + _apply_dedup but
    operates on a date range instead of single-date + backfill. Returns
    a DataFrame matching the Stage 1 schema subset needed for joint
    validation: (symbol, signal_date, signal_strength, pattern).
    """
    rules, survivor_lift = _load_survivor_rules(
        aggregate_path, track1_dir, track4_dir, track5_dir, min_val_lift
    )
    # Eval window: dedup_days before start (so dedup state at start is correct)
    # through end_date.
    eval_start = start_date - timedelta(days=dedup_days + 1)
    eval_features = features.filter(
        (pl.col("date") >= eval_start) & (pl.col("date") <= end_date)
    )
    per_rule_frames = []
    for rule in rules:
        firings = _evaluate_rule_firings(rule, eval_features)
        if firings.height == 0:
            continue
        per_rule_frames.append(firings.with_columns(pl.lit(rule.rule_key).alias("rule_key")))
    if not per_rule_frames:
        return pl.DataFrame(schema={
            "symbol": pl.String, "signal_date": pl.String,
            "signal_strength": pl.Float64, "pattern": pl.String,
        })
    raw = pl.concat(per_rule_frames)
    deduped = _entry_apply_dedup(raw, dedup_days)
    # Filter to entries actually in the [start, end] window (dedup needed the
    # wider lookback but only entries in the publish range count)
    deduped = deduped.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )
    # Compute signal_strength per the contract
    deduped = deduped.with_columns(
        signal_strength=pl.col("rule_key").map_elements(
            lambda k: min(1.0, survivor_lift.get(k, 0.0) / 3.0),
            return_dtype=pl.Float64,
        ),
    )
    return deduped.select([
        "symbol",
        pl.col("date").cast(pl.String).alias("signal_date"),
        "signal_strength",
        pl.col("rule_key").alias("pattern"),
    ])


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _train_xgb_for_variant(
    labeled: pl.DataFrame,
    train_cutoff: date,
    n_rounds: int,
) -> tuple[xgb.Booster, float, float, list[str]]:
    """Train binary XGB on a labeled DataFrame.

    Chronological train/val split at `train_cutoff`. Returns
    (booster, train_auc, val_auc, feature_columns)."""
    labeled = labeled.filter(pl.col("exit_label").is_not_null())
    # _replay_feature_selection assumes is_winner column existence — but we
    # don't need is_winner here. Just apply the bool→int8 + str→one-hot
    # transforms manually using the same logic.
    extra_cols = {"position_gain_pct", "forward_return_pct", "exit_label"}
    skip = _NON_FEATURE_COLS | extra_cols
    feature_cols = [c for c in labeled.columns if c not in skip]
    bool_cols = [c for c in feature_cols if labeled[c].dtype == pl.Boolean]
    if bool_cols:
        labeled = labeled.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])
    str_cols = [c for c in feature_cols if labeled[c].dtype == pl.Utf8]
    if str_cols:
        labeled = labeled.to_dummies(columns=str_cols)
    feature_cols = [c for c in labeled.columns if c not in skip]

    train = labeled.filter(pl.col("date") <= train_cutoff)
    val = labeled.filter(pl.col("date") > train_cutoff)
    if train.height == 0 or val.height == 0:
        raise RuntimeError(f"train/val split is degenerate: train={train.height}, val={val.height}")

    X_train = train.select(feature_cols).to_numpy(allow_copy=True)
    y_train = train["exit_label"].to_numpy()
    X_val = val.select(feature_cols).to_numpy(allow_copy=True)
    y_val = val["exit_label"].to_numpy()

    # Features contain occasional inf values (e.g. volume_ratio when prior
    # volume = 0). XGB rejects inf without explicit missing handling; map
    # inf → nan and tell XGB nan = missing.
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_val = np.nan_to_num(X_val, nan=np.nan, posinf=np.nan, neginf=np.nan)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, missing=np.nan)
    booster = xgb.train(
        DEFAULT_XGB_PARAMS,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )
    train_pred = booster.predict(dtrain)
    val_pred = booster.predict(dval)
    from sklearn.metrics import roc_auc_score
    train_auc = float(roc_auc_score(y_train, train_pred))
    val_auc = float(roc_auc_score(y_val, val_pred))
    return booster, train_auc, val_auc, feature_cols


def _score_holdout_features(
    booster: xgb.Booster,
    features: pl.DataFrame,
    feature_cols: list[str],
    score_percentile: float,
    start: date,
    end: date,
    variant: ExitVariant,
) -> pl.DataFrame:
    """Score every (symbol, date) row in the holdout window where the
    position is currently up ≥variant.gain_threshold_pct. Apply a
    percentile-based threshold so each variant emits the same FRACTION
    of qualified-position rows as EXITs (fair cross-variant comparison).

    Returns DataFrame with columns (symbol, date, exit_score, would_emit_exit)."""
    holdout = features.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    labeled = prepare_label_set(holdout, variant)
    if labeled.height == 0:
        return pl.DataFrame(schema={"symbol": pl.String, "date": pl.Date, "exit_score": pl.Float32, "would_emit_exit": pl.Boolean})
    # Apply same one-hot transforms used in training
    extra_cols = {"position_gain_pct", "forward_return_pct", "exit_label"}
    skip = _NON_FEATURE_COLS | extra_cols
    bool_cols = [c for c in labeled.columns if c not in skip and labeled[c].dtype == pl.Boolean]
    if bool_cols:
        labeled = labeled.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])
    str_cols = [c for c in labeled.columns if c not in skip and labeled[c].dtype == pl.Utf8]
    if str_cols:
        labeled = labeled.to_dummies(columns=str_cols)
    # Ensure all training feature columns exist (one-hot expansion may have
    # produced different columns on holdout if symbol sets differ)
    for col in feature_cols:
        if col not in labeled.columns:
            labeled = labeled.with_columns(pl.lit(0).alias(col))
    X = labeled.select(feature_cols).to_numpy(allow_copy=True)
    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
    dscore = xgb.DMatrix(X, feature_names=feature_cols, missing=np.nan)
    scores = booster.predict(dscore)
    scored = labeled.select(["symbol", "date"]).with_columns(
        exit_score=pl.lit(scores.astype("float32")).alias("exit_score"),
    )
    # Top-N percentile of scores become EXIT predictions
    if scored.height == 0:
        return scored.with_columns(would_emit_exit=pl.lit(False))
    threshold = float(np.quantile(scored["exit_score"].to_numpy(), 1.0 - score_percentile))
    return scored.with_columns(
        would_emit_exit=(pl.col("exit_score") >= threshold),
    )


def _join_entries_with_exits(
    entries: pl.DataFrame,
    exit_scores: pl.DataFrame,
    features: pl.DataFrame,
    eligibility_days: int,
    cap_days: int,
) -> pl.DataFrame:
    """Simulate paired ENTRY → EXIT trades.

    For each ENTRY signal: open position at entry_date+1 open price. Walk
    forward day-by-day on that symbol's price path; exit at the first of:
      (a) a would_emit_exit=True row within eligibility_days
      (b) cap_days trading days elapsed
      (c) the holdout end (last available price)

    Returns one row per ENTRY with realized hold + gain + exit_reason."""
    # Build per-symbol price tables for fast walkforward.
    # Keep dates as Python date objects (not numpy datetime64) so set
    # membership against exits_by_symbol works correctly — polars'
    # iter_rows() returns datetime.date for Date columns, and the price
    # path lookup must match the same type.
    symbol_price: dict[str, dict] = {}
    for sym_df in features.partition_by("symbol", as_dict=True).values():
        sym_df = sym_df.sort("date")
        symbol_price[sym_df["symbol"][0]] = {
            "dates": sym_df["date"].to_list(),  # list[date], not np.datetime64
            "open": sym_df["open"].to_numpy(),
            "close_adj": sym_df["close_adj"].to_numpy(),
        }
    exits_by_symbol: dict[str, set[date]] = defaultdict(set)
    for row in exit_scores.filter(pl.col("would_emit_exit")).iter_rows(named=True):
        exits_by_symbol[row["symbol"]].add(row["date"])
    pairs = []
    # Dedup entries to one per (symbol, signal_date) — pick strongest by
    # signal_strength (matches platform's prepare-context.ts behavior).
    entries_dedup = (
        entries.sort("signal_strength", descending=True)
        .unique(subset=["symbol", "signal_date"], keep="first")
        .sort("signal_date")
    )
    for entry in entries_dedup.iter_rows(named=True):
        symbol = entry["symbol"]
        entry_date_str = entry["signal_date"]
        entry_date = date.fromisoformat(entry_date_str)
        sym = symbol_price.get(symbol)
        if sym is None:
            continue
        # Find next-day open as entry price
        dates_arr = sym["dates"]
        fill_idx = None
        for i in range(len(dates_arr)):
            if dates_arr[i] > entry_date:
                fill_idx = i
                break
        if fill_idx is None:
            continue
        entry_price = float(sym["open"][fill_idx])
        if entry_price <= 0:
            continue
        # Walk forward
        exit_idx = None
        exit_reason = "end_of_period"
        exits_for_symbol = exits_by_symbol.get(symbol, set())
        for j in range(fill_idx, min(fill_idx + cap_days + 1, len(dates_arr))):
            current_date = dates_arr[j]
            days_held = j - fill_idx
            if days_held > cap_days:
                exit_idx = j
                exit_reason = "hard_cap"
                break
            if days_held > eligibility_days:
                # Past eligibility window; still close at end-of-period (eligibility
                # window for emission = 60d; hard cap = 60d so this is identical
                # unless user-overridden)
                exit_idx = j
                exit_reason = "eligibility_cap"
                break
            if days_held > 0 and current_date in exits_for_symbol:
                exit_idx = j
                exit_reason = "exit_signal"
                break
        if exit_idx is None:
            # Walked to end of cap_days or end of price series
            exit_idx = min(fill_idx + cap_days, len(dates_arr) - 1)
            exit_reason = "hard_cap" if (exit_idx - fill_idx >= cap_days) else "end_of_data"
        exit_price = float(sym["close_adj"][exit_idx])
        realized_gain_pct = (exit_price / entry_price - 1.0) * 100
        pairs.append({
            "symbol": symbol,
            "entry_date": entry_date_str,
            "exit_date": str(dates_arr[exit_idx]),
            "days_held": int(exit_idx - fill_idx),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_gain_pct": realized_gain_pct,
            "exit_reason": exit_reason,
            "entry_signal_strength": entry["signal_strength"],
        })
    return pl.DataFrame(pairs)


def _compute_variant_result(
    variant: ExitVariant,
    label_stats: dict,
    train_auc: float,
    val_auc: float,
    pairs: pl.DataFrame,
) -> VariantResult:
    if pairs.height == 0:
        return VariantResult(
            variant_id=variant.label_id(),
            variant_gain_threshold_pct=variant.gain_threshold_pct,
            variant_forward_window_days=variant.forward_window_days,
            variant_giveback_threshold_pct=variant.giveback_threshold_pct,
            n_qualified_positions=label_stats["n_qualified_positions"],
            n_positive_labels=label_stats["n_positive_labels"],
            label_positive_rate=label_stats["positive_rate"],
            xgb_train_auc=train_auc, xgb_val_auc=val_auc,
            n_entries=0, n_entries_with_exit=0, n_entries_hit_hard_cap=0,
            exit_coverage_pct=0.0,
            mean_hold_days=0.0, median_hold_days=0.0, p25_hold_days=0.0,
            p75_hold_days=0.0, min_hold_days=0, max_hold_days=0,
            mean_realized_gain_pct=0.0, median_realized_gain_pct=0.0,
            p25_realized_gain_pct=0.0, p75_realized_gain_pct=0.0,
            win_rate=0.0, max_drawdown_pct=0.0,
            gate_mean_hold_passes=False, gate_median_gain_passes=False,
            gate_win_rate_passes=False, all_gates_pass=False,
        )
    n_with_exit = int(pairs.filter(pl.col("exit_reason") == "exit_signal").height)
    n_hard_cap = int(pairs.filter(pl.col("exit_reason") == "hard_cap").height)
    coverage = n_with_exit / pairs.height if pairs.height else 0.0
    mean_hold = float(pairs["days_held"].mean())
    median_hold = float(pairs["days_held"].median())
    p25_hold = float(pairs["days_held"].quantile(0.25))
    p75_hold = float(pairs["days_held"].quantile(0.75))
    mean_gain = float(pairs["realized_gain_pct"].mean())
    median_gain = float(pairs["realized_gain_pct"].median())
    p25_gain = float(pairs["realized_gain_pct"].quantile(0.25))
    p75_gain = float(pairs["realized_gain_pct"].quantile(0.75))
    win_rate = float((pairs["realized_gain_pct"] > 0).mean())
    max_dd = float(pairs["realized_gain_pct"].min())  # min single-trade gain = "worst trade"
    gate_hold = mean_hold >= 30.0
    gate_gain = median_gain >= 20.0
    gate_win = win_rate >= 0.5
    return VariantResult(
        variant_id=variant.label_id(),
        variant_gain_threshold_pct=variant.gain_threshold_pct,
        variant_forward_window_days=variant.forward_window_days,
        variant_giveback_threshold_pct=variant.giveback_threshold_pct,
        n_qualified_positions=label_stats["n_qualified_positions"],
        n_positive_labels=label_stats["n_positive_labels"],
        label_positive_rate=label_stats["positive_rate"],
        xgb_train_auc=train_auc, xgb_val_auc=val_auc,
        n_entries=int(pairs.height),
        n_entries_with_exit=n_with_exit,
        n_entries_hit_hard_cap=n_hard_cap,
        exit_coverage_pct=coverage * 100,
        mean_hold_days=mean_hold, median_hold_days=median_hold,
        p25_hold_days=p25_hold, p75_hold_days=p75_hold,
        min_hold_days=int(pairs["days_held"].min()),
        max_hold_days=int(pairs["days_held"].max()),
        mean_realized_gain_pct=mean_gain, median_realized_gain_pct=median_gain,
        p25_realized_gain_pct=p25_gain, p75_realized_gain_pct=p75_gain,
        win_rate=win_rate, max_drawdown_pct=max_dd,
        gate_mean_hold_passes=gate_hold,
        gate_median_gain_passes=gate_gain,
        gate_win_rate_passes=gate_win,
        all_gates_pass=gate_hold and gate_gain and gate_win,
    )


def _render_comparison_markdown(results: list[VariantResult]) -> str:
    """Render the variant comparison as a markdown table for the README."""
    lines = ["# Exit-variant sweep comparison", ""]
    lines.append("## Joint validation against Stage 1 ENTRY pool")
    lines.append("")
    lines.append("| | A: g10/f10/b50 | B: g10/f20/b50 | C: g10/f10/b66 | D: g15/f20/b66 | gate |")
    lines.append("|---|---|---|---|---|---|")
    by_var: dict[str, VariantResult] = {r.variant_id.split("_")[1]: r for r in results}

    def cell(r: VariantResult | None, fmt: str, val_attr: str, gate: bool | None = None) -> str:
        if r is None:
            return "—"
        v = getattr(r, val_attr)
        s = format(v, fmt) if isinstance(v, (int, float)) else str(v)
        if gate is not None:
            s = f"**{s}** ✅" if gate else f"{s} ❌"
        return s

    rows_spec: list[tuple[str, str, str, str | None]] = [
        ("n_entries", "Trades simulated", ",d", None),
        ("n_entries_with_exit", "Exits via signal", ",d", None),
        ("n_entries_hit_hard_cap", "Hit hard cap (60d)", ",d", None),
        ("exit_coverage_pct", "EXIT coverage %", ".1f", None),
        ("mean_hold_days", "Mean hold (days)", ".1f", "gate_mean_hold_passes"),
        ("median_hold_days", "Median hold (days)", ".1f", None),
        ("p25_hold_days", "P25 hold (days)", ".1f", None),
        ("mean_realized_gain_pct", "Mean realized gain (%)", "+.2f", None),
        ("median_realized_gain_pct", "Median realized gain (%)", "+.2f", "gate_median_gain_passes"),
        ("p25_realized_gain_pct", "P25 realized gain (%)", "+.2f", None),
        ("win_rate", "Win rate", ".4f", "gate_win_rate_passes"),
        ("max_drawdown_pct", "Worst trade (%)", "+.2f", None),
        ("xgb_val_auc", "XGB val AUC", ".4f", None),
        ("label_positive_rate", "Label positive rate", ".4f", None),
        ("all_gates_pass", "All gates pass?", "", None),
    ]
    gates = {"gate_mean_hold_passes", "gate_median_gain_passes", "gate_win_rate_passes"}
    for attr, label, fmt, gate_attr in rows_spec:
        cells = []
        gate_label = ""
        for vid in ["A", "B", "C", "D"]:
            r = by_var.get(vid)
            if attr == "all_gates_pass":
                cells.append("✅" if r and r.all_gates_pass else "❌")
            else:
                gate = getattr(r, gate_attr) if r and gate_attr else None
                cells.append(cell(r, fmt, attr, gate))
        if attr == "mean_hold_days":
            gate_label = "≥30d"
        elif attr == "median_realized_gain_pct":
            gate_label = "≥+20%"
        elif attr == "win_rate":
            gate_label = "≥0.50"
        elif attr == "all_gates_pass":
            gate_label = "all 3"
        row_cells = " | ".join(cells)
        lines.append(f"| {label} | {row_cells} | {gate_label} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    features_path = _resolve_path(args.features)

    print(f"exit-variant sweep — Stage 2 of quant signal contract")
    print(f"  features: {features_path.relative_to(_REPO_ROOT)}")
    features = pl.read_parquet(features_path)
    print(f"  loaded {features.height:,} feature rows × {features.width} cols")

    # Regenerate historical entries across [train_cutoff, holdout_end - 60d]
    # so each entry has full forward window for joint validation
    max_feature_date = features["date"].max()
    if isinstance(max_feature_date, str):
        max_feature_date = date.fromisoformat(max_feature_date)
    # Entries must allow `joint_cap_days` forward days, leave 5-day buffer for
    # next-day-open fill + price gaps
    last_eligible_entry = min(args.holdout_end, max_feature_date - timedelta(days=args.joint_cap_days + 5))
    first_eligible_entry = args.train_cutoff + timedelta(days=1)
    print(f"  regenerating historical entries: {first_eligible_entry} → {last_eligible_entry}")
    print(f"    (last eligible entry shifted back to ensure {args.joint_cap_days}d forward window)")
    aggregate_path = _resolve_path(args.walkforward_aggregate)
    track1_dir = _resolve_path(args.track1_dir)
    track4_dir = _resolve_path(args.track4_dir)
    track5_dir = _resolve_path(args.track5_dir)
    entries_holdout = _generate_historical_entries(
        features, aggregate_path, track1_dir, track4_dir, track5_dir,
        args.entry_min_val_lift, args.entry_dedup_days,
        first_eligible_entry, last_eligible_entry,
    )
    n_before_strength = entries_holdout.height
    entries_holdout = entries_holdout.filter(pl.col("signal_strength") >= args.entry_min_strength)
    print(f"  entries: {n_before_strength:,} historical → {entries_holdout.height:,} after signal_strength ≥{args.entry_min_strength} filter")

    # Optional cluster-of-interest filter (Stage 2.1 — addresses the gain-gate
    # binding constraint from Stage 2.0 by restricting entries to the v1
    # walk-forward cluster Phase B v3 deployed against)
    if args.entry_cluster_id is not None:
        cm_path = _resolve_path(args.entry_cluster_membership)
        print(f"  cluster filter: {cm_path.relative_to(_REPO_ROOT)} (cluster_id={args.entry_cluster_id})")
        cm = pl.read_parquet(cm_path).filter(pl.col("cluster_id") == args.entry_cluster_id)
        # cluster-membership has (symbol, date) — date is polars.Date
        # entries_holdout's signal_date is String (per the historical-generation logic).
        # Convert the membership date to String to match for the inner join.
        cm = cm.with_columns(pl.col("date").cast(pl.String).alias("signal_date")).select(["symbol", "signal_date"]).unique()
        n_before_cluster = entries_holdout.height
        entries_holdout = entries_holdout.join(cm, on=["symbol", "signal_date"], how="inner")
        print(f"  entries: {n_before_cluster:,} → {entries_holdout.height:,} after cluster filter ({(entries_holdout.height/max(n_before_cluster,1))*100:.1f}% retained)")
    print(f"  train cutoff: {args.train_cutoff} | holdout end: {args.holdout_end}")
    print(f"  eligibility window: {args.exit_eligibility_days}d | hard cap: {args.joint_cap_days}d")

    out_dir = args.out_dir or (_REPO_ROOT / f"runs/{date.today().isoformat()}-exit_variant_sweep")
    out_dir = _resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir: {out_dir.relative_to(_REPO_ROOT)}")

    results: list[VariantResult] = []
    for vid in args.variants:
        variant = VARIANTS[vid]
        v_out = out_dir / variant.label_id()
        v_out.mkdir(exist_ok=True)
        print(f"\n=== Variant {vid}: {variant.label_id()} ===")
        labeled = prepare_label_set(features, variant)
        stats = label_statistics(labeled, variant)
        print(f"  qualified: {stats['n_qualified_positions']:,}, positives: {stats['n_positive_labels']:,} ({stats['positive_rate']*100:.2f}%)")

        print(f"  training XGB ({args.n_rounds} rounds) ...")
        t_train = time.perf_counter()
        booster, train_auc, val_auc, feature_cols = _train_xgb_for_variant(
            labeled, args.train_cutoff, args.n_rounds
        )
        print(f"  train_auc={train_auc:.4f}  val_auc={val_auc:.4f}  ({time.perf_counter() - t_train:.1f}s)")

        # Save booster
        booster.save_model(str(v_out / "xgb_model.json"))

        # Score holdout
        print(f"  scoring holdout ({args.train_cutoff} → {args.holdout_end}) ...")
        exit_scores = _score_holdout_features(
            booster, features, feature_cols, args.exit_score_percentile,
            args.train_cutoff, args.holdout_end, variant,
        )
        n_emit = int(exit_scores.filter(pl.col("would_emit_exit")).height) if exit_scores.height else 0
        print(f"  scored {exit_scores.height:,} rows, would_emit_exit on {n_emit:,}")

        # Joint validation
        print(f"  simulating paired trades ...")
        pairs = _join_entries_with_exits(
            entries_holdout, exit_scores, features,
            args.exit_eligibility_days, args.joint_cap_days,
        )
        pairs.write_parquet(v_out / "paired_trades.parquet")
        print(f"  {pairs.height:,} pairs written to {(v_out / 'paired_trades.parquet').relative_to(_REPO_ROOT)}")

        result = _compute_variant_result(variant, stats, train_auc, val_auc, pairs)
        results.append(result)

        # Per-variant manifest
        v_manifest = {
            "variant": asdict(variant),
            "training": {"train_auc": train_auc, "val_auc": val_auc, "n_rounds": args.n_rounds},
            "result": asdict(result),
        }
        (v_out / "manifest.json").write_text(json.dumps(v_manifest, indent=2, default=str))
        print(f"  GATES: hold={'✅' if result.gate_mean_hold_passes else '❌'} ({result.mean_hold_days:.1f}d) | "
              f"gain={'✅' if result.gate_median_gain_passes else '❌'} ({result.median_realized_gain_pct:+.2f}%) | "
              f"win={'✅' if result.gate_win_rate_passes else '❌'} ({result.win_rate:.2%}) | "
              f"all={'✅' if result.all_gates_pass else '❌'}")

    # Top-level comparison
    print(f"\n=== COMPARISON ===")
    comparison_df = pl.DataFrame([asdict(r) for r in results])
    comparison_df.write_parquet(out_dir / "comparison.parquet")
    md = _render_comparison_markdown(results)
    (out_dir / "comparison.md").write_text(md)
    print(md)

    sweep_manifest = {
        "run_id": f"{date.today().isoformat()}-exit_variant_sweep",
        "pipeline_step": "exit_variant_sweep",
        "git_commit_of_quant_repo": _git_head_sha(),
        "features_path": str(features_path.relative_to(_REPO_ROOT)),
        "walkforward_aggregate_path": str(aggregate_path.relative_to(_REPO_ROOT)),
        "train_cutoff": args.train_cutoff.isoformat(),
        "holdout_end": args.holdout_end.isoformat(),
        "first_eligible_entry": first_eligible_entry.isoformat(),
        "last_eligible_entry": last_eligible_entry.isoformat(),
        "exit_eligibility_days": args.exit_eligibility_days,
        "joint_cap_days": args.joint_cap_days,
        "exit_score_percentile": args.exit_score_percentile,
        "entry_min_val_lift": args.entry_min_val_lift,
        "entry_dedup_days": args.entry_dedup_days,
        "entry_min_strength": args.entry_min_strength,
        "entry_cluster_membership_path": str(_resolve_path(args.entry_cluster_membership).relative_to(_REPO_ROOT)) if args.entry_cluster_id is not None else None,
        "entry_cluster_id": args.entry_cluster_id,
        "variants_evaluated": args.variants,
        "n_entries_in_holdout": int(entries_holdout.height),
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    (out_dir / "sweep_manifest.json").write_text(json.dumps(sweep_manifest, indent=2))
    print(f"\nWrote sweep_manifest.json + comparison.{{parquet,md}} to {out_dir.relative_to(_REPO_ROOT)}")
    print(f"Wall clock: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
