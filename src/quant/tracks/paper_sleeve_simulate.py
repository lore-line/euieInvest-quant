"""Track B-sleeve — Tier 3 paper sleeve backtest.

Multi-period backtest of the Phase B walk-forward survivor rules as a
Tier 3 paper sleeve. Trades are simulated against the post-cleanup
price_history (open/high/low/close/close_adj/volume) under realistic
Wealthsimple constraints.

See PR #1 issuecomment-4441437671 (server-team Phase B brief) for the
canonical spec.

Inputs
------

- ``runs/<date>-step4_walkforward_validation/rule-validation-aggregate.parquet``
  → filter to ``is_walk_forward_survivor == True``
- ``runs/<date>-step4_walkforward_validation/rule-validation.parquet``
  → per-rule conditions (round-trip via the rule loaders)
- The labeled feature matrix (for rule evaluation; we recompute matches
  per simulation step rather than caching)
- Raw price_history (open/close/high/low/volume + close_adj) — needed
  for fill prices, intraday HL stops, and exit logic. Loaded from
  ``data/snapshots/ohlcv.parquet``.

Sleeve mechanics
----------------

- Capital: $10,000 USD starting sleeve
- Position sizing: 10% per signal (within the brief's 5-15% band)
- Max concurrent positions: 4
- Whole shares only (Wealthsimple constraint)
- Friction: commission-free, 0.10% slippage per leg
- Fill: next-day open after signal date

Exit logic per position (first-trigger wins):
1. **Target** — close_adj reaches `entry_price × 1.20` (+20% off entry)
2. **Stop**  — close_adj drops to `entry_price × 0.92` (-8% off entry)
3. **Time** — 45 trading days elapsed since entry

Signal-to-position mapping
--------------------------

Each (rule, symbol, signal_date) triggers a candidate signal. The
sleeve may reject if:
- 4 positions already open
- Position size for this $10K sleeve would be < 1 share at entry price
- Symbol already held (no overlap)

If multiple rules fire on the same (symbol, date), the highest-
mean_val_lift rule wins (deterministic tie-breaking).

Outputs
-------

- ``signals.parquet`` — one row per simulated trade
- ``sleeve-results.parquet`` — single-row aggregate (sharpe, win_rate,
  max_drawdown, etc.)

Cost
----

CPU-bound. ~2000 surviving rules × ~500K holdout windows of rule
evaluation, plus the chronological position-management loop. Estimate
3-10 minutes wall clock depending on signal density.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant.train import RunStatus
from quant.tracks import make_run_id
from quant.tracks.walkforward_validate import (
    Rule,
    evaluate_rule_on_slice,
    load_phase_a_rules,
)

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Position:
    rule_key: str
    symbol: str
    signal_date: date
    entry_date: date
    entry_price: float  # close_adj on entry, with slippage applied
    position_size_usd: float
    shares: int
    expected_lift: float

    # Filled in on exit
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str | None = None  # "target", "stop", "time", "end_of_period"
    realized_pnl_usd: float | None = None


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B Track B-sleeve — Tier 3 paper sleeve simulator")
    p.add_argument(
        "--walkforward-dir", type=Path,
        default=Path("runs/2026-05-14-step4_walkforward_validation"),
        help="Path to the Track B-walk run dir (must contain rule-validation-aggregate.parquet).",
    )
    p.add_argument(
        "--features", type=Path, default=Path("data/features/features.parquet"),
        help="Labeled feature matrix (for rule signal generation).",
    )
    p.add_argument(
        "--prices", type=Path, default=Path("data/snapshots/ohlcv.parquet"),
        help="Raw price_history with open/high/low/close/close_adj/volume.",
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
        "--cluster-membership", type=Path,
        default=Path("runs/2026-05-14-step3g_embedding_clustering/cluster-membership.parquet"),
        help="Track 7 (original) cluster-membership.parquet — or the walk-forward "
             "equivalent (runs/2026-05-14-step4_walkforward_cluster_id/"
             "cluster-membership-walkforward.parquet) to test the universe filter "
             "without forward-look bias. Used when --universe is cluster-7-*.",
    )
    p.add_argument(
        "--cluster-id", type=int, default=7,
        help="Cluster ID to filter on when --universe is cluster-7-*. "
             "Track 7 original uses cluster 7; the walk-forward cluster ID may "
             "differ (e.g. cluster 8). Match the cluster-membership file you pass.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 3, 30))
    p.add_argument("--sleeve-usd", type=float, default=10_000.0)
    p.add_argument("--position-size-pct", type=float, default=0.10,
                   help="Fraction of sleeve per signal (brief: 0.05-0.15).")
    p.add_argument("--max-concurrent", type=int, default=4,
                   help="Max concurrent positions. Use -1 for unlimited (rule-cohort PnL baseline).")
    p.add_argument("--slippage-pct", type=float, default=0.001, help="Per-leg.")
    # Default exit thresholds (--per-rule-exits=fixed). Label-aware mode
    # overrides these from rule source.
    p.add_argument("--target-pct", type=float, default=0.20)
    p.add_argument("--stop-pct", type=float, default=0.08)
    p.add_argument("--time-decay-days", type=int, default=45)

    # Phase B v2 additions per server-team direction (issuecomment-4451476355).
    p.add_argument(
        "--ranker", choices=["first-fire", "top-lift"], default="top-lift",
        help="Signal selector when multiple signals fire on the same day. "
             "first-fire = chronological (Phase B v1 baseline); "
             "top-lift = sort by expected_lift desc, take top until slots full.",
    )
    p.add_argument(
        "--universe",
        choices=["all", "cluster-7-rows", "cluster-7-symbols", "negative-decay"],
        default="all",
        help="Restrict signal universe. "
             "all = no filter; "
             "cluster-7-rows = (symbol,date) pairs in Track 7's cluster 7 (rigorous); "
             "cluster-7-symbols = symbols that appear in cluster 7 at least once (broad); "
             "negative-decay = rules with lift_decay <= 0 (out-of-sample stronger than train).",
    )
    p.add_argument(
        "--min-val-lift", type=float, default=1.5,
        help="Selectivity gate: drop rules where Track B-walk's min_val_lift < this. "
             "1.5 = default survivor threshold; 2.4 picks the negative-decay-like cohort.",
    )
    p.add_argument(
        "--per-rule-exits", choices=["fixed", "label-aware"], default="fixed",
        help="fixed = use --target-pct/--stop-pct/--time-decay-days for all rules; "
             "label-aware = match the rule's training label horizon (L1: +15/-6/30d, "
             "L2: +20/-8/30d, L3: +30/-12/60d, L4: +15/-6/30d, L5: +20/-8/60d; "
             "Track 1 and Track 5 default to L2-spec).",
    )
    return p.parse_args(argv)


# Label-aware exit parameters per rule source.
# Tracks 1 and 5 trained on the global +20%/30d (L2) target, so they use L2 exits.
# Track 4's per-label rules use their own label's horizon.
_LABEL_EXITS: dict[str, tuple[float, float, int]] = {
    "L1": (0.15, 0.06, 30),  # +15%/-6%/30d
    "L2": (0.20, 0.08, 30),  # +20%/-8%/30d  (default for Tracks 1 + 5)
    "L3": (0.30, 0.12, 60),  # +30%/-12%/60d
    "L4": (0.15, 0.06, 30),  # +15%/-6%/30d (close-to-close variant)
    "L5": (0.20, 0.08, 60),  # +20%/-8%/60d
}


def _exits_for_rule(rule_key: str, source_track: str, source_label_or_regime: str,
                    args: argparse.Namespace) -> tuple[float, float, int]:
    """Return (target_pct, stop_pct, time_decay_days) per rule."""
    if args.per_rule_exits == "fixed":
        return args.target_pct, args.stop_pct, args.time_decay_days
    # label-aware
    if source_track == "step3c" and source_label_or_regime in _LABEL_EXITS:
        return _LABEL_EXITS[source_label_or_regime]
    # Tracks 1 and 5 default to L2 (+20%/30d) — the global Phase A target
    return _LABEL_EXITS["L2"]


def _next_trading_day_index(date_to_idx: dict[date, int], dates: list[date], from_date: date) -> int | None:
    """Return the index of the first date >= from_date, or None if past end."""
    idx = date_to_idx.get(from_date)
    if idx is not None:
        return idx
    # Linear scan — fast enough for the few thousand signals we'll have.
    for i, d in enumerate(dates):
        if d >= from_date:
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step4_paper_sleeve_simulation"
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
        epoch_total=1,
    )
    status.update(state="training", epoch_current=0)

    try:
        print(f"track B-sleeve — Tier 3 paper sleeve simulator")
        wf_dir = args.walkforward_dir if args.walkforward_dir.is_absolute() else (_REPO_ROOT / args.walkforward_dir)
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        prices_path = args.prices if args.prices.is_absolute() else (_REPO_ROOT / args.prices)
        track1_dir = args.track1_dir if args.track1_dir.is_absolute() else (_REPO_ROOT / args.track1_dir)
        track4_dir = args.track4_dir if args.track4_dir.is_absolute() else (_REPO_ROOT / args.track4_dir)
        track5_dir = args.track5_dir if args.track5_dir.is_absolute() else (_REPO_ROOT / args.track5_dir)

        # Load walk-forward survivors + apply selectivity gate.
        agg = pl.read_parquet(wf_dir / "rule-validation-aggregate.parquet")
        baseline_survivors = agg.filter(pl.col("is_walk_forward_survivor"))
        print(f"  walk-forward baseline survivors: {baseline_survivors.height:,}")

        # Selectivity gates:
        # 1. --min-val-lift filter on min_val_lift (worst-window lift)
        survivors = baseline_survivors.filter(pl.col("min_val_lift") >= args.min_val_lift)
        print(f"  after min_val_lift >= {args.min_val_lift}: {survivors.height:,}")
        # 2. --universe = negative-decay further restricts to lift_decay <= 0
        if args.universe == "negative-decay":
            survivors = survivors.filter(pl.col("lift_decay") <= 0)
            print(f"  after universe=negative-decay (lift_decay <= 0): {survivors.height:,}")

        if survivors.height == 0:
            raise RuntimeError(
                f"no rules pass the selectivity gate (min_val_lift >= {args.min_val_lift}). "
                f"Loosen the threshold."
            )

        survivor_keys = set(survivors["rule_key"].to_list())
        survivor_lift = dict(zip(survivors["rule_key"].to_list(), survivors["mean_val_lift"].to_list()))

        # Load all rules + filter to survivors.
        all_rules = load_phase_a_rules(track1_dir, track4_dir, track5_dir)
        rules = [r for r in all_rules if r.rule_key in survivor_keys]
        print(f"  loaded {len(rules):,} survivor rule definitions ({len(all_rules):,} total Phase A rules)")

        # Build a rule_key → (source_track, source_label_or_regime) map for per-rule
        # exit lookup later.
        rule_source: dict[str, tuple[str, str]] = {
            r.rule_key: (r.source_track, r.source_label_or_regime) for r in rules
        }

        # Build the universe filter (applied to signals after generation).
        # cluster-7-rows: (symbol, date) pairs that are in Track 7 cluster 7
        # cluster-7-symbols: any (symbol, date) for symbols that appear in cluster 7 at least once
        # negative-decay: applied via survivor filter above; no per-signal filter here
        cluster_membership_filter: pl.DataFrame | None = None
        cluster_symbols_filter: set[str] | None = None
        if args.universe in ("cluster-7-rows", "cluster-7-symbols"):
            cm_path = args.cluster_membership if args.cluster_membership.is_absolute() else (_REPO_ROOT / args.cluster_membership)
            cm = pl.read_parquet(cm_path).filter(pl.col("cluster_id") == args.cluster_id)
            print(f"  cluster-membership source: {cm_path.relative_to(_REPO_ROOT)} (filtered to cluster_id={args.cluster_id})")
            if args.universe == "cluster-7-rows":
                cluster_membership_filter = cm.select(["symbol", "date"]).unique()
                print(f"  universe filter: cluster-7-rows → {cluster_membership_filter.height:,} (symbol, date) pairs")
            else:  # cluster-7-symbols
                cluster_symbols_filter = set(cm["symbol"].unique().to_list())
                print(f"  universe filter: cluster-7-symbols → {len(cluster_symbols_filter):,} unique symbols")

        # Load features (for rule evaluation in the signal window).
        labeled = pl.read_parquet(features_path)
        sim_features = labeled.filter(
            (pl.col("date") >= args.start) & (pl.col("date") <= args.end)
        )
        print(f"  sim period features: {sim_features.height:,} rows ({args.start} → {args.end})")

        # Generate signals: for each rule, evaluate on the full sim period; emit a signal
        # for each (symbol, date) that matches.
        #
        # We keep everything in polars throughout — earlier draft built a Python
        # list of dicts and converted at the end, which OOM-killed the container
        # at ~41M candidate signals before dedup.  Building per-rule polars
        # frames and concatenating once at the end keeps memory bounded
        # (intermediate frames are at most n_sim_rows = ~1.1M rows × ~3 cols).
        print(f"  generating signals from {len(rules):,} rules ...")
        t_sig = time.perf_counter()
        from quant.tracks.walkforward_validate import _condition_expr
        per_rule_frames: list[pl.DataFrame] = []
        for i, rule in enumerate(rules):
            expr = None
            skip_rule = False
            for cond in rule.conditions:
                feat = cond["feature"]
                if feat not in sim_features.columns:
                    skip_rule = True
                    break
                e = _condition_expr(cond)
                expr = e if expr is None else (expr & e)
            if skip_rule or expr is None:
                continue
            matches = sim_features.filter(expr).select(["symbol", "date"])
            if matches.height == 0:
                continue
            # Add rule metadata as constant columns.
            per_rule_frames.append(matches.with_columns([
                pl.lit(rule.rule_key).alias("rule_key"),
                pl.lit(float(survivor_lift[rule.rule_key])).alias("expected_lift"),
            ]))
            if (i + 1) % 500 == 0:
                running_rows = sum(f.height for f in per_rule_frames)
                print(f"    {i+1:,}/{len(rules):,} rules processed, "
                      f"{running_rows:,} candidate (symbol,date,rule) rows ({time.perf_counter()-t_sig:.1f}s)")

        if not per_rule_frames:
            print("  no rule matches found in the sim period — aborting")
            sig_df = pl.DataFrame(schema={
                "symbol": pl.Utf8, "date": pl.Date,
                "rule_key": pl.Utf8, "expected_lift": pl.Float64,
            })
        else:
            sig_df = pl.concat(per_rule_frames)
            print(f"  total candidate (rule,symbol,date) signals: {sig_df.height:,}")

        # Load prices.
        prices = pl.read_parquet(prices_path).select([
            "symbol", "date", "open", "high", "low", "close", "close_adj", "volume"
        ]).filter(
            (pl.col("date") >= args.start - timedelta(days=10))  # need a few days lookback for fill date
            & (pl.col("date") <= args.end + timedelta(days=args.time_decay_days + 30))  # need lookahead for exits
        ).sort(["symbol", "date"])
        print(f"  price rows in sim window+buffer: {prices.height:,}")

        # Per-symbol price tables — indexed by (symbol, date) → row idx
        # For each symbol, build a date-indexed numpy array for fast lookup.
        symbol_price_table: dict[str, dict[str, Any]] = {}
        for sym in prices["symbol"].unique().to_list():
            sym_prices = prices.filter(pl.col("symbol") == sym).sort("date")
            dates_list = sym_prices["date"].to_list()
            symbol_price_table[sym] = {
                "dates": dates_list,
                "date_to_idx": {d: i for i, d in enumerate(dates_list)},
                "open": np.array(sym_prices["open"].to_list(), dtype=np.float64),
                "high": np.array(sym_prices["high"].to_list(), dtype=np.float64),
                "low": np.array(sym_prices["low"].to_list(), dtype=np.float64),
                "close": np.array(sym_prices["close"].to_list(), dtype=np.float64),
                "close_adj": np.array(sym_prices["close_adj"].to_list(), dtype=np.float64),
            }
        print(f"  built per-symbol price tables for {len(symbol_price_table):,} symbols")

        # Apply universe filter to signals.
        if cluster_membership_filter is not None:
            before = sig_df.height
            sig_df = sig_df.join(cluster_membership_filter, on=["symbol", "date"], how="inner")
            print(f"  after universe=cluster-7-rows filter: {sig_df.height:,} / {before:,} signals retained")
        elif cluster_symbols_filter is not None:
            before = sig_df.height
            sig_df = sig_df.filter(pl.col("symbol").is_in(list(cluster_symbols_filter)))
            print(f"  after universe=cluster-7-symbols filter: {sig_df.height:,} / {before:,} signals retained")

        # Deduplicate signals: highest expected_lift per (symbol, date) wins.
        # (Column is `date` from the features frame; we rename to signal_date below
        # only when materializing the per-trade output.)
        if sig_df.height > 0:
            sig_df = sig_df.sort("expected_lift", descending=True).unique(
                subset=["symbol", "date"], keep="first"
            ).sort("date").rename({"date": "signal_date"})
        print(f"  unique (symbol, signal_date) signals: {sig_df.height:,}")

        # Run the chronological simulation, grouped by signal_date so the ranker
        # can sort all signals firing on the same day together.
        print(f"  simulating sleeve ({args.start} → {args.end}) "
              f"with ranker={args.ranker}, max_concurrent={'unlimited' if args.max_concurrent < 0 else args.max_concurrent} ...")
        open_positions: list[Position] = []
        closed_positions: list[Position] = []
        rejected_no_price = 0
        rejected_concurrent_cap = 0
        rejected_already_held = 0
        rejected_size_too_small = 0
        max_concurrent_effective = float("inf") if args.max_concurrent < 0 else args.max_concurrent

        # Group signals by signal_date so we can apply the ranker per day.
        # Polars group_by + iter_groups keeps memory bounded vs materializing a dict.
        sig_groups = sig_df.group_by("signal_date").agg(
            pl.col("symbol"),
            pl.col("rule_key"),
            pl.col("expected_lift"),
        ).sort("signal_date")

        for grp in sig_groups.iter_rows(named=True):
            signal_date = grp["signal_date"]
            symbols = grp["symbol"]
            rule_keys = grp["rule_key"]
            expected_lifts = grp["expected_lift"]
            day_signals = list(zip(symbols, rule_keys, expected_lifts))

            # 1) Close any open positions whose exit conditions have been met by signal_date.
            still_open: list[Position] = []
            for pos in open_positions:
                exit_info = _check_exit_with_rule_exits(pos, signal_date, symbol_price_table, args, rule_source)
                if exit_info is None:
                    still_open.append(pos)
                else:
                    pos.exit_date, pos.exit_price, pos.exit_reason = exit_info
                    pos.realized_pnl_usd = (pos.exit_price - pos.entry_price) * pos.shares
                    closed_positions.append(pos)
            open_positions = still_open

            # 2) Rank today's signals if applicable.
            if args.ranker == "top-lift":
                day_signals.sort(key=lambda x: x[2], reverse=True)  # by expected_lift desc

            # 3) Try to open each signal in (ranked or chronological-ish) order.
            for symbol, rule_key, expected_lift in day_signals:
                if len(open_positions) >= max_concurrent_effective:
                    # Count remaining as concurrent-capped rejections.
                    rejected_concurrent_cap += 1
                    continue
                if symbol not in symbol_price_table:
                    rejected_no_price += 1
                    continue
                if any(p.symbol == symbol for p in open_positions):
                    rejected_already_held += 1
                    continue
                sym_table = symbol_price_table[symbol]
                # Find fill date — next trading day's open after signal_date.
                fill_idx = None
                for i, d in enumerate(sym_table["dates"]):
                    if d > signal_date:
                        fill_idx = i
                        break
                if fill_idx is None:
                    rejected_no_price += 1
                    continue
                fill_open = float(sym_table["open"][fill_idx])
                if fill_open <= 0:
                    rejected_no_price += 1
                    continue
                entry_price = fill_open * (1.0 + args.slippage_pct)
                position_size_usd = args.sleeve_usd * args.position_size_pct
                shares = int(position_size_usd / entry_price)
                if shares < 1:
                    rejected_size_too_small += 1
                    continue
                actual_position_usd = shares * entry_price
                pos = Position(
                    rule_key=rule_key,
                    symbol=symbol,
                    signal_date=signal_date,
                    entry_date=sym_table["dates"][fill_idx],
                    entry_price=entry_price,
                    position_size_usd=actual_position_usd,
                    shares=shares,
                    expected_lift=expected_lift,
                )
                open_positions.append(pos)

        # Force-close any still-open positions at the end-of-period close.
        end_d = args.end
        for pos in open_positions:
            sym_table = symbol_price_table.get(pos.symbol)
            if sym_table is None:
                continue
            # Find the last trading day <= end_d for this symbol.
            last_idx = None
            for i in range(len(sym_table["dates"]) - 1, -1, -1):
                if sym_table["dates"][i] <= end_d:
                    last_idx = i
                    break
            if last_idx is None:
                continue
            last_close_adj = float(sym_table["close_adj"][last_idx])
            pos.exit_date = sym_table["dates"][last_idx]
            pos.exit_price = last_close_adj * (1.0 - args.slippage_pct)  # slippage on exit
            pos.exit_reason = "end_of_period"
            pos.realized_pnl_usd = (pos.exit_price - pos.entry_price) * pos.shares
            closed_positions.append(pos)

        print(f"  positions opened: {len(closed_positions):,}")
        print(f"  rejections: concurrent_cap={rejected_concurrent_cap:,}, "
              f"already_held={rejected_already_held:,}, "
              f"size_too_small={rejected_size_too_small:,}, "
              f"no_price={rejected_no_price:,}")

        # Build signals.parquet.
        sig_rows = []
        for pos in closed_positions:
            sig_rows.append({
                "rule_key": pos.rule_key,
                "symbol": pos.symbol,
                "signal_date": pos.signal_date,
                "expected_lift": round(pos.expected_lift, 4),
                "position_size_usd": round(pos.position_size_usd, 2),
                "shares": pos.shares,
                "status": "exited",
                "entered_at": datetime.combine(pos.entry_date, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
                "entered_price": round(pos.entry_price, 4),
                "exited_at": datetime.combine(pos.exit_date, datetime.min.time(), tzinfo=timezone.utc).isoformat() if pos.exit_date else None,
                "exited_price": round(pos.exit_price, 4) if pos.exit_price else None,
                "exit_reason": pos.exit_reason,
                "realized_pnl_usd": round(pos.realized_pnl_usd, 2) if pos.realized_pnl_usd else None,
                "notes": "",
            })
        sig_out = pl.DataFrame(sig_rows)
        sig_path = run_dir / "signals.parquet"
        sig_out.write_parquet(sig_path)
        print(f"  wrote {sig_path.relative_to(_REPO_ROOT)}  ({sig_out.height:,} trades)")

        # Aggregate metrics.
        total_pnl = sum(p.realized_pnl_usd for p in closed_positions if p.realized_pnl_usd is not None)
        win_rate = (
            sum(1 for p in closed_positions if p.realized_pnl_usd and p.realized_pnl_usd > 0)
            / max(len(closed_positions), 1)
        )
        avg_holding_days = (
            sum((p.exit_date - p.entry_date).days for p in closed_positions if p.exit_date)
            / max(len(closed_positions), 1)
        )

        # Daily-equity curve for sharpe + max-drawdown.
        # Track cash-and-positions on each trading date by simulating mark-to-market.
        sleeve_curve = _compute_equity_curve(closed_positions, symbol_price_table, args)
        if len(sleeve_curve) > 1:
            returns = np.diff(sleeve_curve) / sleeve_curve[:-1]
            sharpe = float(np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252))
            peak = np.maximum.accumulate(sleeve_curve)
            dd = (sleeve_curve - peak) / peak
            max_drawdown = float(dd.min())
        else:
            sharpe = 0.0
            max_drawdown = 0.0

        exit_reasons = defaultdict(int)
        for p in closed_positions:
            exit_reasons[p.exit_reason or "unknown"] += 1

        agg_row = {
            "framework_version": "1.0",
            "start_date": args.start.isoformat(),
            "end_date": args.end.isoformat(),
            "total_signals": int(sig_out.height),
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "avg_holding_days": round(avg_holding_days, 2),
            "total_pnl_usd": round(total_pnl, 2),
            "final_sleeve_usd": round(args.sleeve_usd + total_pnl, 2),
            "return_pct": round(total_pnl / args.sleeve_usd, 4),
            "results_json": json.dumps({
                "exit_reasons": dict(exit_reasons),
                "rejections": {
                    "concurrent_cap": rejected_concurrent_cap,
                    "already_held": rejected_already_held,
                    "size_too_small": rejected_size_too_small,
                    "no_price": rejected_no_price,
                },
            }),
        }
        agg_out = pl.DataFrame([agg_row])
        agg_path = run_dir / "sleeve-results.parquet"
        agg_out.write_parquet(agg_path)
        print(f"  wrote {agg_path.relative_to(_REPO_ROOT)}")

        # Manifest.
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "walkforward_dir": str(wf_dir.relative_to(_REPO_ROOT)),
            "n_survivor_rules": len(rules),
            "n_candidate_signals_pre_dedup": int(sum(f.height for f in per_rule_frames) if per_rule_frames else 0),
            "n_unique_symbol_date_signals": int(sig_df.height),
            "n_trades": len(closed_positions),
            "config": {
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "sleeve_usd": args.sleeve_usd,
                "position_size_pct": args.position_size_pct,
                "max_concurrent": args.max_concurrent,
                "slippage_pct": args.slippage_pct,
                "target_pct": args.target_pct,
                "stop_pct": args.stop_pct,
                "time_decay_days": args.time_decay_days,
                # Phase B v2 flags
                "ranker": args.ranker,
                "universe": args.universe,
                "min_val_lift": args.min_val_lift,
                "per_rule_exits": args.per_rule_exits,
                # Phase B v3 flag
                "cluster_id": args.cluster_id,
                "cluster_membership_path": str(args.cluster_membership),
            },
            "results": {
                "total_pnl_usd": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_drawdown, 4),
                "avg_holding_days": round(avg_holding_days, 2),
                "exit_reasons": dict(exit_reasons),
            },
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        print()
        print(f"=== TRACK B-SLEEVE RESULT ===")
        print(f"  trades:         {len(closed_positions):,}")
        print(f"  win rate:       {win_rate:.2%}")
        print(f"  total P&L:      ${total_pnl:+,.2f}")
        print(f"  return %:       {total_pnl/args.sleeve_usd:+.2%}")
        print(f"  Sharpe:         {sharpe:.3f}")
        print(f"  max drawdown:   {max_drawdown:.2%}")
        print(f"  avg holding:    {avg_holding_days:.1f} days")
        print(f"  exit reasons:   {dict(exit_reasons)}")
        print(f"  wall clock:     {wall_clock_s/60:.1f} min")
        status.record_checkpoint(epoch=1)
        status.update(state="done", epoch_current=1)
        return 0
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


def _check_exit_with_rule_exits(
    pos: Position,
    current_date: date,
    symbol_price_table: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    rule_source: dict[str, tuple[str, str]],
) -> tuple[date, float, str] | None:
    """Check if position should exit by current_date with per-rule exits.

    Returns (exit_date, exit_price, reason) or None.
    """
    sym_table = symbol_price_table.get(pos.symbol)
    if sym_table is None:
        return None
    entry_idx = sym_table["date_to_idx"].get(pos.entry_date)
    if entry_idx is None:
        return None

    src = rule_source.get(pos.rule_key)
    if src is not None:
        target_pct, stop_pct, time_decay_days = _exits_for_rule(pos.rule_key, src[0], src[1], args)
    else:
        target_pct, stop_pct, time_decay_days = args.target_pct, args.stop_pct, args.time_decay_days

    target = pos.entry_price * (1.0 + target_pct)
    stop = pos.entry_price * (1.0 - stop_pct)
    time_decay_idx = entry_idx + time_decay_days
    for i in range(entry_idx + 1, min(len(sym_table["dates"]), time_decay_idx + 1)):
        d = sym_table["dates"][i]
        if d > current_date:
            break
        h = float(sym_table["high"][i])
        l = float(sym_table["low"][i])
        if l <= stop:
            return (d, stop * (1.0 - args.slippage_pct), "stop")
        if h >= target:
            return (d, target * (1.0 - args.slippage_pct), "target")
    # Time decay check.
    if time_decay_idx < len(sym_table["dates"]):
        decay_date = sym_table["dates"][time_decay_idx]
        if decay_date <= current_date:
            decay_price = float(sym_table["close_adj"][time_decay_idx])
            return (decay_date, decay_price * (1.0 - args.slippage_pct), "time")
    return None


# Keep the original _check_exit as a thin wrapper for backward compat with any
# tests that import it (none currently, but defensive).
def _check_exit(
    pos: Position,
    current_date: date,
    symbol_price_table: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[date, float, str] | None:
    return _check_exit_with_rule_exits(pos, current_date, symbol_price_table, args, {})


def _compute_equity_curve(
    closed_positions: list[Position],
    symbol_price_table: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> np.ndarray:
    """Compute a simple equity curve by summing realized P&Ls in chronological order.

    Not mark-to-market — we'd need to track open positions per day. But for a
    realized-P&L sleeve the order-of-magnitude Sharpe + drawdown numbers are
    informative.
    """
    if not closed_positions:
        return np.array([args.sleeve_usd])
    by_exit = sorted(closed_positions, key=lambda p: p.exit_date if p.exit_date else date(9999, 1, 1))
    curve = [args.sleeve_usd]
    running = args.sleeve_usd
    for pos in by_exit:
        if pos.realized_pnl_usd is None:
            continue
        running += pos.realized_pnl_usd
        curve.append(running)
    return np.array(curve)


if __name__ == "__main__":
    raise SystemExit(main())
