"""P3 Day 1 v0.4 — regime-conditional strategy performance attribution.

Per issue #22 spec + multi-strategy framework (issue #20 + #21 conversations).
Builds the strategy × regime Sharpe/CAGR matrix that drives dynamic capital
allocation across strategies. This is the operational input for doctrine §6
(performance-driven allocation) under doctrine §5.5 (multi-edge regime coverage).

v0.4 scope (consumer-side only):
  - 2a equity_momentum_fast (from latest walkforward signals.parquet)
  - 2b equity_momentum_slow (from latest walkforward signals.parquet)
  - 2c_bull_trend (P2 v0.4 in-memory output)

Server-side-fed strategies (left as TODO stubs with documented schema):
  - 1 Buffett (Stream 1, server-side P&L)
  - 2c_grid (DCA grid, server-side P&L)
  - 3 hype/reddit (Stream 3, server-side P&L)
  - 4 scalping (Stream 4, server-side P&L)

Attribution mechanism: each trade is assigned to the regime that was active
on its entry_date (via P1 v0.4 regime_labels_v1.parquet). Per (strategy,
regime) cell we compute:
  - n_trades
  - n_unique_entry_days (proxy for active days in regime)
  - mean_pnl_pct (per-trade)
  - per_trade_sharpe (mean / std, annualized by trades/year)
  - win_rate_pct
  - max_trade_loss_pct
  - cumulative_total_return_pct (compound)

Output: data/quant_publish/strategy_regime_sharpe_matrix.parquet

Run: PYTHONPATH=src python -m quant.tracks.regime_attribution_matrix
"""
from __future__ import annotations

import io
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


REGIME_LABELS_PATH = Path("data/quant_publish/regime_labels_v1.parquet")
PUBLISH_PATH = Path("data/quant_publish/strategy_regime_sharpe_matrix.parquet")

# Consumer-side strategy P&L sources
STRATEGY_SOURCES = {
    "stream_2a_equity_momentum_fast": Path(
        "D:/quant-runs/2026-05-18-equity_momentum_fast_g15_vc15_walkforward/signals.parquet"
    ),
    "stream_2b_equity_momentum_slow": Path(
        "D:/quant-runs/2026-05-18-equity_momentum_slow_g25_vc15_walkforward/signals.parquet"
    ),
}

# Server-side strategy P&L feed (combined parquet with `strategy_id` discriminator,
# published per issue #22 contract). v0.4.1 ingests stream_2c_grid_inverse_aggressive
# and stream_2c_grid_ungated; future strategies (stream_1_buffett, etc.) will be
# added to this same file by server-team.
SERVER_SIDE_FEED = Path("data/quant_publish/server_strategy_signals.parquet")
# v0.6: per-day portfolio-return time series per strategy, for cross-strategy
# daily-aggregation analysis. Auto-ingests when server team publishes.
# Schema: strategy_id | date | daily_return_pct | open_deal_count | active_capital_pct
SERVER_DAILY_FEED = Path("data/quant_publish/server_strategy_daily.parquet")
# v0.7: multi-strategy harness policy outputs (per (date, policy_id))
# Schema: policy_id | date | daily_return_pct | active_strategy_id | active_capital_pct
SERVER_POLICY_FEED = Path("data/quant_publish/multi_strategy_policies.parquet")

# Server-side strategies not-yet-feeding (informational)
SERVER_SIDE_PENDING = {
    "stream_1_buffett": "deferred — needs FIFO matching against pre-tracking holdings",
    "stream_3_hype":    "doesn't exist yet (paper-only spec)",
    "stream_4_scalping": "doesn't exist yet (collapsed into Stream 2)",
}


def normalize_signals(raw: pl.DataFrame, strategy_id: str) -> pl.DataFrame:
    """Map various raw signals.parquet schemas → unified (strategy_id, entry_date, net_pnl_pct).

    equity_momentum_walkforward signals.parquet has `gross_pnl_pct` per trade.
    Net pnl ≈ gross - friction; for v0.4 we use gross as a uniform proxy
    (friction varies per strategy; consistent comparison requires per-strategy
    friction model, not in scope here).
    """
    if "gross_pnl_pct" in raw.columns and "entry_date" in raw.columns:
        hold_expr = (
            pl.col("realized_horizon_days").cast(pl.Int64).alias("hold_days")
            if "realized_horizon_days" in raw.columns
            else pl.lit(None, dtype=pl.Int64).alias("hold_days")
        )
        return raw.select([
            pl.lit(strategy_id).alias("strategy_id"),
            pl.col("entry_date"),
            pl.col("gross_pnl_pct").alias("net_pnl_pct"),
            hold_expr,
        ])
    raise ValueError(f"unrecognized signals schema for {strategy_id}: cols={raw.columns}")


def load_consumer_side_trades() -> pl.DataFrame:
    """Load + normalize all consumer-side strategy P&L."""
    frames = []
    for sid, path in STRATEGY_SOURCES.items():
        if not path.exists():
            print(f"  [warn] {sid}: signals parquet not at {path}, skipping")
            continue
        raw = pl.read_parquet(path)
        norm = normalize_signals(raw, sid)
        frames.append(norm)
        print(f"  {sid}: {norm.height} trades")
    if not frames:
        return pl.DataFrame(schema={
            "strategy_id": pl.String, "entry_date": pl.Date,
            "net_pnl_pct": pl.Float64, "hold_days": pl.Int64,
        })
    return pl.concat(frames, how="vertical")


def load_server_side_feed() -> pl.DataFrame:
    """Load + normalize server-side strategy P&L from combined parquet.

    Server schema uses Datetime[UTC] for entry/exit dates; we cast to Date
    to match the consumer-side schema for cell-level join consistency.
    """
    if not SERVER_SIDE_FEED.exists():
        print(f"  [warn] server feed not at {SERVER_SIDE_FEED}")
        return pl.DataFrame(schema={
            "strategy_id": pl.String, "entry_date": pl.Date,
            "net_pnl_pct": pl.Float64, "hold_days": pl.Int64,
        })
    raw = pl.read_parquet(SERVER_SIDE_FEED)
    norm = raw.select([
        pl.col("strategy_id"),
        pl.col("entry_date").cast(pl.Date),
        pl.col("net_pnl_pct"),
        pl.col("hold_days").cast(pl.Int64),
    ])
    for sid in norm["strategy_id"].unique().to_list():
        n = norm.filter(pl.col("strategy_id") == sid).height
        print(f"  {sid}: {n} trades (server feed)")
    return norm


def load_p2_bull_trend_trades() -> pl.DataFrame:
    """Recompute P2 v0.4 trades in-memory and normalize.

    Imports the P2 simulator to avoid duplicating the breakout/exit logic.
    """
    from quant.tracks.crypto_bull_trend_v0_4 import (
        fetch_crypto, compute_breakout_entries, simulate_trades,
    )
    ohlcv = fetch_crypto()
    sig = compute_breakout_entries(ohlcv)
    trades = simulate_trades(sig)
    if trades.height == 0:
        return pl.DataFrame(schema={
            "strategy_id": pl.String, "entry_date": pl.Date,
            "net_pnl_pct": pl.Float64, "hold_days": pl.Int64,
        })
    return trades.select([
        pl.lit("stream_2c_bull_trend").alias("strategy_id"),
        pl.col("entry_date"),
        pl.col("net_pnl_pct"),
        pl.col("hold_days").cast(pl.Int64),
    ])


def load_server_daily_feed() -> pl.DataFrame:
    """v0.6: load per-day portfolio-return time series.

    Auto-no-ops if `server_strategy_daily.parquet` doesn't exist (gracefully
    skip per-day analysis until server team publishes the daily feed).

    Schema expected:
        strategy_id | date | daily_return_pct | open_deal_count | active_capital_pct
    """
    if not SERVER_DAILY_FEED.exists():
        return pl.DataFrame(schema={
            "strategy_id": pl.String, "date": pl.Date,
            "daily_return_pct": pl.Float64,
            "open_deal_count": pl.Int64,
            "active_capital_pct": pl.Float64,
        })
    raw = pl.read_parquet(SERVER_DAILY_FEED)
    # Tolerate Datetime[UTC] -> Date cast, same as trade feed
    if "date" in raw.columns and raw["date"].dtype != pl.Date:
        raw = raw.with_columns(pl.col("date").cast(pl.Date))
    return raw.sort(["strategy_id", "date"])


def compute_per_day_per_regime(
    daily: pl.DataFrame, regime_labels: pl.DataFrame,
) -> pl.DataFrame:
    """v0.6: per-(strategy, regime) DAILY-return aggregation.

    Different lens than per-trade attribution:
      - Per-trade: which trades worked, per regime that was active at entry
      - Per-day:   what was the daily portfolio return on each regime-active day

    Per-day is the right lens for capital-allocation simulators (combine
    multiple strategies' daily returns into composite portfolio).
    """
    if daily.height == 0:
        return pl.DataFrame()
    rl = regime_labels.select(["date", "regime_label"])
    joined = daily.join(rl, on="date", how="left")

    rows = []
    for (strategy_id, regime), sub in joined.group_by(["strategy_id", "regime_label"]):
        sid = strategy_id if isinstance(strategy_id, str) else str(strategy_id)
        reg = regime if isinstance(regime, str) else (str(regime) if regime is not None else "unlabeled")
        rets = sub["daily_return_pct"].drop_nulls().to_list()
        if len(rets) < 5:
            continue  # too few days for meaningful aggregation
        arr = np.array(rets)
        cap = sub["active_capital_pct"].drop_nulls()
        mean_cap = float(cap.mean()) if cap.len() > 0 else 0.0
        rows.append({
            "strategy_id": sid,
            "regime_label": reg,
            "n_active_days": len(rets),
            "mean_daily_return_pct": float(arr.mean()),
            "annualized_return_pct": float(arr.mean() * 252),
            "daily_volatility_pct": float(arr.std()),
            "daily_sharpe": float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0,
            "max_daily_loss_pct": float(arr.min()),
            "mean_active_capital_pct": mean_cap,
            "active_capital_weighted_return_pct": float(arr.mean() * mean_cap),
        })
    return pl.DataFrame(rows).sort(["strategy_id", "regime_label"])


def load_multi_strategy_policies() -> pl.DataFrame:
    """v0.7: load multi-strategy harness policy outputs.

    Auto-no-ops if `multi_strategy_policies.parquet` doesn't exist.
    """
    if not SERVER_POLICY_FEED.exists():
        return pl.DataFrame(schema={
            "policy_id": pl.String, "date": pl.Date,
            "daily_return_pct": pl.Float64,
            "active_strategy_id": pl.String,
            "active_capital_pct": pl.Float64,
        })
    raw = pl.read_parquet(SERVER_POLICY_FEED)
    if "date" in raw.columns and raw["date"].dtype != pl.Date:
        raw = raw.with_columns(pl.col("date").cast(pl.Date))
    return raw.sort(["policy_id", "date"])


def compute_policy_summary(policy_df: pl.DataFrame) -> pl.DataFrame:
    """v0.7: per-policy CAGR + daily Sharpe + max-DD + transition count.

    For the BTC-rotation thesis validation, the key comparison is:
      baseline_inverse_aggressive       → +47.18% canonical
      inverse_aggressive_plus_btc_rotation → expected +55-75% per
                                             v0.6 daily matrix math
    """
    if policy_df.height == 0:
        return pl.DataFrame()
    rows = []
    for policy_id, sub in policy_df.group_by("policy_id"):
        rets = sub["daily_return_pct"].drop_nulls().to_list()
        if len(rets) < 30:
            continue
        arr = np.array(rets)
        # CAGR from compounding daily
        cumulative = float(np.prod(1 + arr / 100.0))
        n_days = len(rets)
        cagr_pct = (cumulative ** (252.0 / n_days) - 1.0) * 100.0
        # Daily Sharpe
        sharpe = (float(arr.mean()) / float(arr.std()) * np.sqrt(252)
                  if arr.std() > 0 else 0.0)
        # Max drawdown on cumulative equity curve
        equity = np.cumprod(1 + arr / 100.0)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100.0
        max_dd = float(drawdown.min())
        # Transition count
        if "active_strategy_id" in sub.columns:
            strategies = sub["active_strategy_id"].to_list()
            transitions = sum(1 for i in range(1, len(strategies))
                              if strategies[i] != strategies[i-1])
        else:
            transitions = 0
        rows.append({
            "policy_id": str(policy_id) if not isinstance(policy_id, str) else policy_id,
            "n_days": n_days,
            "cagr_pct": cagr_pct,
            "daily_sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "n_transitions": transitions,
            "annual_transitions": float(transitions * 252 / max(n_days, 1)),
            "total_return_pct": (cumulative - 1.0) * 100.0,
        })
    return pl.DataFrame(rows).sort("cagr_pct", descending=True)


def per_trade_sharpe(net_pnls: list[float], holds: list[int]) -> float:
    """Per-trade Sharpe with sqrt(252/mean_hold_days) annualization.

    Per consumer-side `sharpe-metric-scoping.md` doctrine — appropriate for
    parallel discovery / multi-strategy comparison (not capital-recycled serial).
    """
    if len(net_pnls) < 2:
        return 0.0
    arr = np.array(net_pnls)
    if arr.std() == 0:
        return 0.0
    mean_hold = max(np.mean(holds), 1.0)
    annual_factor = np.sqrt(252.0 / mean_hold)
    return float(arr.mean() / arr.std() * annual_factor)


def compute_matrix(trades: pl.DataFrame, regime_labels: pl.DataFrame) -> pl.DataFrame:
    """Build the strategy × regime cell-by-cell."""
    rl = regime_labels.select(["date", "regime_label"]).rename({"date": "entry_date"})
    joined = trades.join(rl, on="entry_date", how="left")

    rows = []
    for (strategy_id, regime), sub in joined.group_by(["strategy_id", "regime_label"]):
        # group_by returns Series for keys; coerce
        sid = strategy_id if isinstance(strategy_id, str) else str(strategy_id)
        reg = regime if isinstance(regime, str) else (str(regime) if regime is not None else "unlabeled")
        net = sub["net_pnl_pct"].drop_nulls().to_list()
        holds = sub["hold_days"].drop_nulls().to_list() or [1] * sub.height
        if not net:
            continue

        # NOTE: removed `cumulative_total_pct` — it's misleading for cells with
        # different n_trades (compound of 2500 trades at +11% mean = 10^19,
        # not a real return). Per-trade metrics are the comparable ones.
        # Use `mean_pnl_pct × n_trades` mentally for additive-capital-allocation
        # CAGR estimate.
        mean_pnl = float(np.mean(net))
        mean_hold = float(np.mean(holds)) if holds else 1.0
        # comparable_yield_pct: cross-strategy-class comparator that isn't
        # Sharpe-inflation-vulnerable. Approximates "annualized yield if you
        # cycled this trade pattern continuously."
        comparable_yield_pct = mean_pnl * (252.0 / max(mean_hold, 1.0))
        # sharpe_class: flag whether Sharpe value is comparable cross-class
        is_tp_clustered = bool(
            sid.startswith("stream_2c_grid") or "dca" in sid.lower()
        )
        rows.append({
            "strategy_id": sid,
            "regime_label": reg,
            "n_trades": sub.height,
            "n_unique_entry_days": int(sub["entry_date"].n_unique()),
            "mean_pnl_pct": mean_pnl,
            "median_pnl_pct": float(np.median(net)),
            "per_trade_sharpe": per_trade_sharpe(net, holds),
            "sharpe_class": "tp_clustered" if is_tp_clustered else "realistic",
            "comparable_yield_pct": comparable_yield_pct,
            "win_rate_pct": float((np.array(net) > 0).mean() * 100),
            "max_trade_loss_pct": float(min(net)),
            "mean_hold_days": mean_hold,
            "confidence": "high" if sub.height >= 30 else ("medium" if sub.height >= 10 else "low"),
        })
    return pl.DataFrame(rows).sort(["strategy_id", "regime_label"])


def main() -> None:
    print("=== P3 v0.4 — regime-conditional strategy attribution matrix ===\n")

    print("[1/4] loading consumer-side strategy trades...")
    consumer_trades = load_consumer_side_trades()
    print(f"      {consumer_trades.height} trades across {consumer_trades['strategy_id'].n_unique()} strategies")

    print("\n[2/4] computing P2 v0.4 bull-trend trades in-memory...")
    p2_trades = load_p2_bull_trend_trades()
    print(f"      {p2_trades.height} P2 trades")

    print("\n[3/4] loading server-side P&L feed...")
    server_trades = load_server_side_feed()

    all_trades = pl.concat([consumer_trades, p2_trades, server_trades], how="vertical")
    print(f"\n      grand total: {all_trades.height} trades across "
          f"{all_trades['strategy_id'].n_unique()} strategies")

    print("\n[4/4] loading P1 v0.4 regime labels + computing matrix...")
    regime_labels = pl.read_parquet(REGIME_LABELS_PATH)
    matrix = compute_matrix(all_trades, regime_labels)
    print(f"      matrix: {matrix.height} cells (strategy × regime)")

    # Pretty-print
    print("\n=== STRATEGY × REGIME ATTRIBUTION MATRIX ===\n")
    for sid in matrix["strategy_id"].unique().to_list():
        print(f"\n{sid}")
        sub = matrix.filter(pl.col("strategy_id") == sid).sort("regime_label")
        print(f"  {'regime':25s} {'n':>5s} {'win%':>6s} {'mean%':>7s} "
              f"{'sharpe':>7s} {'cls':>12s} {'yield%/yr':>10s} {'conf':>6s}")
        for r in sub.iter_rows(named=True):
            print(f"  {r['regime_label']:25s} {r['n_trades']:5d} "
                  f"{r['win_rate_pct']:6.1f} {r['mean_pnl_pct']:+7.2f} "
                  f"{r['per_trade_sharpe']:+7.3f} {r['sharpe_class']:>12s} "
                  f"{r['comparable_yield_pct']:+10.1f} {r['confidence']:>6s}")

    # Server-side gaps still pending
    print("\n=== SERVER-SIDE PENDING (not yet in feed) ===")
    for sid, note in SERVER_SIDE_PENDING.items():
        print(f"  {sid:35s}  {note}")

    # Publish
    PUBLISH_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_parquet(PUBLISH_PATH)
    print(f"\n=== wrote {PUBLISH_PATH} ({matrix.height} rows) ===")

    # v0.6: per-day aggregation (auto-skips if daily feed absent)
    print("\n=== v0.6 per-day aggregation ===")
    daily = load_server_daily_feed()
    if daily.height == 0:
        print(f"  [waiting] {SERVER_DAILY_FEED} not yet on publish surface")
        print(f"  When server team publishes the daily-curve parquet, this re-run")
        print(f"  auto-emits a per-day per-regime matrix at the schema:")
        print(f"    strategy_id | regime_label | n_active_days | mean_daily_return_pct |")
        print(f"    annualized_return_pct | daily_volatility_pct | daily_sharpe |")
        print(f"    max_daily_loss_pct | mean_active_capital_pct | active_capital_weighted_return_pct")
    else:
        print(f"  loaded {daily.height} daily-return rows for "
              f"{daily['strategy_id'].n_unique()} strategies")
        daily_matrix = compute_per_day_per_regime(daily, regime_labels)
        if daily_matrix.height > 0:
            print(f"\n  per-day matrix: {daily_matrix.height} cells")
            for r in daily_matrix.iter_rows(named=True):
                print(f"    {r['strategy_id']:35s} {r['regime_label']:25s}  "
                      f"n_days={r['n_active_days']:4d}  "
                      f"annual={r['annualized_return_pct']:+7.2f}%  "
                      f"sharpe={r['daily_sharpe']:+6.2f}  "
                      f"cap_wt_ret={r['active_capital_weighted_return_pct']:+6.3f}%/day")
            daily_publish = Path("data/quant_publish/strategy_regime_daily_matrix.parquet")
            daily_matrix.write_parquet(daily_publish)
            print(f"\n  wrote {daily_publish}")

    # v0.7: multi-strategy harness policy summary (auto-skips if absent)
    print("\n=== v0.7 multi-strategy policy summary ===")
    policies = load_multi_strategy_policies()
    if policies.height == 0:
        print(f"  [waiting] {SERVER_POLICY_FEED} not yet on publish surface")
        print(f"  When server-team harness publishes policy outputs, this re-run")
        print(f"  auto-emits per-policy CAGR/Sharpe/MaxDD/transition stats.")
        print(f"  Expected first policies: baseline_ungated_dca, baseline_inverse_aggressive,")
        print(f"                           inverse_aggressive_plus_btc_steady_bull_rotation")
        print(f"  BTC-rotation target per v0.6 matrix math: +55-75% CAGR")
    else:
        print(f"  loaded {policies.height} policy-daily rows for "
              f"{policies['policy_id'].n_unique()} policies")
        policy_summary = compute_policy_summary(policies)
        if policy_summary.height > 0:
            print(f"\n  per-policy summary:")
            print(f"  {'policy_id':50s} {'n_days':>7s} {'CAGR%':>8s} {'Sharpe':>7s} "
                  f"{'MaxDD%':>7s} {'trans/yr':>9s}")
            for r in policy_summary.iter_rows(named=True):
                print(f"  {r['policy_id']:50s} {r['n_days']:7d} {r['cagr_pct']:+8.2f} "
                      f"{r['daily_sharpe']:+7.2f} {r['max_drawdown_pct']:+7.2f} "
                      f"{r['annual_transitions']:9.1f}")
            policy_publish = Path("data/quant_publish/multi_strategy_policy_summary.parquet")
            policy_summary.write_parquet(policy_publish)
            print(f"\n  wrote {policy_publish}")

    # Multi-strategy framework summary — use comparable_yield_pct (cross-class-safe)
    # instead of Sharpe (TP-clustered classes inflate to +50, dwarfs realistic
    # momentum cells even when they have higher per-trade economics).
    print("\n=== MULTI-STRATEGY ALLOCATION HINTS (ranked by comparable_yield_pct) ===")
    for regime in matrix["regime_label"].unique().drop_nulls().to_list():
        sub = matrix.filter((pl.col("regime_label") == regime)
                            & (pl.col("confidence") != "low")
                            ).sort("comparable_yield_pct", descending=True)
        if sub.height == 0:
            continue
        print(f"\n  {regime}:")
        for r in sub.head(3).iter_rows(named=True):
            print(f"    {r['strategy_id']:35s} yield={r['comparable_yield_pct']:+8.1f}%/yr  "
                  f"sharpe={r['per_trade_sharpe']:+7.2f} ({r['sharpe_class']:>12s})  n={r['n_trades']}")


if __name__ == "__main__":
    main()
