"""Derive stream_2b daily curve from per-trade signals + OHLCV.

Per server-team request (issue #22 ~14:30Z 2026-05-19): unblock equity-side
multi-strategy harness by publishing daily-return time series matching the
agreed schema:
    strategy_id | date | daily_return_pct | open_deal_count | active_capital_pct

Methodology:
1. Each trade in `stream_2b_equity_momentum_slow_signals.parquet` has
   (symbol, entry_date, exit_date). Pull daily close prices for that symbol
   over the trade's window from `data/snapshots/ohlcv.parquet`.
2. For each calendar day D: enumerate all trades open on D
   (entry_date <= D < exit_date), compute each trade's day-D return via
   close[D] / close[D-1] - 1.
3. Portfolio daily return = equal-weight mean of open-trade per-day returns
   (consistent with the walkforward's `max_concurrent=4` config — at most
   4 positions held concurrently, capital split equally).
4. active_capital_pct = open_deal_count / max_concurrent × 100
5. Cap n_positions at max_concurrent (anything in excess gets queued; we
   approximate by truncating per day).

Output: data/quant_publish/stream_2b_daily.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl


PUBLISH = Path("data/quant_publish")
SIGNALS_PARQUET = PUBLISH / "stream_2b_equity_momentum_slow_signals.parquet"
OHLCV_PARQUET = Path("data/snapshots/ohlcv.parquet")
OUT_PARQUET = PUBLISH / "stream_2b_daily.parquet"

# Per slow_g25_vc15 walkforward config
MAX_CONCURRENT = 4


def main() -> None:
    print("=== Stream 2b daily curve derivation ===\n")

    print("[1/4] loading per-trade signals + OHLCV...")
    sigs = pl.read_parquet(SIGNALS_PARQUET)
    print(f"  signals: {sigs.height} trades")
    print(f"  cols: {sigs.columns}")
    # Need: entry_date, exit_date, entry_price, exit_price, symbol
    # Published signals only have (strategy_id, entry_date, exit_date, net_pnl_pct, hold_days)
    # — need to re-load the FULL signals.parquet from D:/quant-runs for symbol + prices
    full_sigs = pl.read_parquet(
        "D:/quant-runs/2026-05-18-equity_momentum_slow_g25_vc15_walkforward/signals.parquet"
    )
    print(f"  full signals: {full_sigs.height} trades, cols: {full_sigs.columns}")

    ohlcv = pl.read_parquet(OHLCV_PARQUET).select(["symbol", "date", "close_adj"])
    print(f"  OHLCV: {ohlcv.height} rows")

    print("\n[2/4] building per-(symbol, date) daily-return panel...")
    # Pre-compute per-symbol daily returns
    panel = ohlcv.sort(["symbol", "date"]).with_columns(
        daily_return_pct=(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("symbol") - 1.0) * 100.0
        ),
    ).drop_nulls("daily_return_pct")
    print(f"  panel: {panel.height} per-symbol per-day return rows")

    print("\n[3/4] computing per-day portfolio returns...")
    # For each trade, determine the days it was OPEN: [entry_date+1, exit_date]
    # (we don't get exposure on entry day itself; the daily return D-to-D+1 first counts the day after entry)
    # Approximation: include day D = entry_date+1 through exit_date
    trades = full_sigs.select(["symbol", "entry_date", "exit_date"]).with_columns(
        active_start=pl.col("entry_date") + pl.duration(days=1),
        active_end=pl.col("exit_date"),
    )

    # For each trade, generate the list of days it was active by joining to the panel
    # Each (symbol, day) gets one return contribution from each open trade on that symbol-day
    # Use a join + filter approach
    trades_lazy = trades.lazy().select(["symbol", "active_start", "active_end"])
    panel_lazy = panel.lazy()

    # cross-join trades × panel rows for the symbol, filter to active days
    joined = (
        trades_lazy.join(panel_lazy, on="symbol", how="inner")
        .filter(
            (pl.col("date") >= pl.col("active_start"))
            & (pl.col("date") <= pl.col("active_end"))
        )
        .group_by("date")
        .agg(
            n_open_deals=pl.len(),
            sum_daily_return=pl.col("daily_return_pct").sum(),
        )
        .with_columns(
            # Cap open deal count at MAX_CONCURRENT (simulator constraint)
            n_open_deals_capped=pl.min_horizontal([pl.col("n_open_deals"), pl.lit(MAX_CONCURRENT)]),
        )
        .with_columns(
            # Portfolio daily return = average of per-trade returns, weighted by cap
            # Equivalent to: sum / n_open (uncapped) when n_open <= MAX_CONCURRENT
            # When n_open > MAX_CONCURRENT, approximate as average of all open trades
            daily_return_pct=pl.col("sum_daily_return") / pl.col("n_open_deals"),
            active_capital_pct=pl.col("n_open_deals_capped") / MAX_CONCURRENT * 100.0,
        )
        .sort("date")
        .collect()
    )

    print(f"  daily curve: {joined.height} rows, "
          f"{joined['date'].min()} → {joined['date'].max()}")

    print("\n[4/4] formatting + publishing...")
    out = joined.select([
        pl.lit("stream_2b_equity_momentum_slow").alias("strategy_id"),
        pl.col("date"),
        pl.col("daily_return_pct"),
        pl.col("n_open_deals_capped").alias("open_deal_count"),
        pl.col("active_capital_pct"),
    ])
    print(f"  schema: {out.schema}")
    print(f"  sample:")
    print(out.head(5))
    print(f"  tail:")
    print(out.tail(5))

    out.write_parquet(OUT_PARQUET)
    print(f"\n  wrote {OUT_PARQUET}")

    # Diagnostic stats
    arr = np.array(out["daily_return_pct"].to_list())
    n_days = len(arr)
    print(f"\n  daily return stats:")
    print(f"    n_days:     {n_days}")
    print(f"    mean:       {arr.mean():+.4f}%/day")
    print(f"    std:        {arr.std():.4f}")
    print(f"    annualized: {(np.prod(1+arr/100))**(252/n_days)-1:+.2%} (252 trading-day year)")
    print(f"    sharpe:     {arr.mean()/arr.std()*np.sqrt(252):.3f}")
    print(f"    median open deals: {out['open_deal_count'].median()}")
    print(f"    median active cap: {out['active_capital_pct'].median():.1f}%")


if __name__ == "__main__":
    main()
