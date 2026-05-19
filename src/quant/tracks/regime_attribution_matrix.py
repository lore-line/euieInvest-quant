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

# Server-side strategy P&L feeds we'd consume if available
# (TODO: server-team publishes these to their `data/quant_publish/` analog)
SERVER_SIDE_STUBS = {
    "stream_1_buffett":            "data/quant_publish/buffett_signals.parquet (TODO)",
    "stream_2c_grid":              "data/quant_publish/dca_grid_signals.parquet (TODO)",
    "stream_2c_grid_inverse_aggr": "data/quant_publish/dca_grid_inverse_aggressive_signals.parquet (TODO)",
    "stream_3_hype":               "data/quant_publish/hype_signals.parquet (TODO)",
    "stream_4_scalping":           "data/quant_publish/scalping_signals.parquet (TODO)",
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
        rows.append({
            "strategy_id": sid,
            "regime_label": reg,
            "n_trades": sub.height,
            "n_unique_entry_days": int(sub["entry_date"].n_unique()),
            "mean_pnl_pct": float(np.mean(net)),
            "median_pnl_pct": float(np.median(net)),
            "per_trade_sharpe": per_trade_sharpe(net, holds),
            "win_rate_pct": float((np.array(net) > 0).mean() * 100),
            "max_trade_loss_pct": float(min(net)),
            "mean_hold_days": float(np.mean(holds)),
            "confidence": "high" if sub.height >= 30 else ("medium" if sub.height >= 10 else "low"),
        })
    return pl.DataFrame(rows).sort(["strategy_id", "regime_label"])


def main() -> None:
    print("=== P3 v0.4 — regime-conditional strategy attribution matrix ===\n")

    print("[1/3] loading consumer-side strategy trades...")
    consumer_trades = load_consumer_side_trades()
    print(f"      {consumer_trades.height} trades across {consumer_trades['strategy_id'].n_unique()} strategies")

    print("\n[2/3] computing P2 v0.4 bull-trend trades in-memory...")
    p2_trades = load_p2_bull_trend_trades()
    print(f"      {p2_trades.height} P2 trades")
    all_trades = pl.concat([consumer_trades, p2_trades], how="vertical")
    print(f"      total: {all_trades.height} trades, "
          f"{all_trades['strategy_id'].n_unique()} strategies")

    print("\n[3/3] loading P1 v0.4 regime labels + computing matrix...")
    regime_labels = pl.read_parquet(REGIME_LABELS_PATH)
    matrix = compute_matrix(all_trades, regime_labels)
    print(f"      matrix: {matrix.height} cells (strategy × regime)")

    # Pretty-print
    print("\n=== STRATEGY × REGIME ATTRIBUTION MATRIX ===\n")
    for sid in matrix["strategy_id"].unique().to_list():
        print(f"\n{sid}")
        sub = matrix.filter(pl.col("strategy_id") == sid).sort("regime_label")
        print(f"  {'regime':25s} {'n':>5s} {'win%':>6s} {'mean%':>7s} {'med%':>7s} "
              f"{'sharpe':>7s} {'maxloss':>8s} {'conf':>6s}")
        for r in sub.iter_rows(named=True):
            print(f"  {r['regime_label']:25s} {r['n_trades']:5d} "
                  f"{r['win_rate_pct']:6.1f} {r['mean_pnl_pct']:+7.2f} "
                  f"{r['median_pnl_pct']:+7.2f} {r['per_trade_sharpe']:+7.3f} "
                  f"{r['max_trade_loss_pct']:+8.2f} {r['confidence']:>6s}")

    # Server-side gaps
    print("\n=== SERVER-SIDE P&L FEEDS NEEDED FOR FULL MATRIX ===")
    for sid, path_note in SERVER_SIDE_STUBS.items():
        print(f"  {sid:35s}  {path_note}")
    print("\n  Consumer-side cannot compute these cells without the server-team")
    print("  publishing trade-level P&L parquets (same schema as our equity_momentum")
    print("  signals.parquet — `strategy_id | entry_date | exit_date | net_pnl_pct |")
    print("  hold_days`).")

    # Publish
    PUBLISH_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_parquet(PUBLISH_PATH)
    print(f"\n=== wrote {PUBLISH_PATH} ({matrix.height} rows) ===")

    # Multi-strategy framework summary
    print("\n=== MULTI-STRATEGY ALLOCATION HINTS (from matrix) ===")
    for regime in matrix["regime_label"].unique().drop_nulls().to_list():
        sub = matrix.filter((pl.col("regime_label") == regime)
                            & (pl.col("confidence") != "low")).sort("per_trade_sharpe", descending=True)
        if sub.height == 0:
            continue
        best = sub.row(0, named=True)
        print(f"  {regime:25s} → best consumer-side strategy: {best['strategy_id']:35s} "
              f"sharpe={best['per_trade_sharpe']:+6.3f} n={best['n_trades']}")


if __name__ == "__main__":
    main()
