"""Gap and range features.

All four functions are implemented as of the /ohlcv ``open`` migration
(see CLAUDE.md §4). ``open``, ``high``, ``low``, ``close`` are all
split-adjusted on a consistent basis, so per-bar features mixing them
are not distorted by split days.

Spec: CLAUDE.md §7 (gaps.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["gap_pct", "range_expansion", "body_range_ratio", "inside_bar"]


def gap_pct(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``gap_pct = (open[t] - close[t-1]) / close[t-1]`` per symbol.

    Null on the first row per symbol (no prior close).
    """
    out = df.sort(["symbol", "date"])
    prev_close = pl.col("close").shift(1).over("symbol")
    return out.with_columns(
        ((pl.col("open") - prev_close) / prev_close).alias("gap_pct")
    )


def range_expansion(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """Add ``range_expansion_{lookback}d`` per symbol.

    ``TR[t] / mean(TR[t-lookback+1 .. t-1])`` — today's True Range as a
    multiple of recent average. Values > 1 mean today's range is wider
    than the recent norm; > 2 is a notable expansion.
    """
    out = df.sort(["symbol", "date"])
    prev_close = pl.col("close").shift(1)
    hl = pl.col("high") - pl.col("low")
    hpc = (pl.col("high") - prev_close).abs()
    lpc = (pl.col("low") - prev_close).abs()
    tr = pl.max_horizontal([hl, hpc, lpc]).over("symbol")
    out = out.with_columns(tr.alias("_tr"))
    # Lookback mean excludes today (shift by 1)
    prior_mean = (
        pl.col("_tr")
        .shift(1)
        .rolling_mean(window_size=lookback, min_samples=lookback)
        .over("symbol")
    )
    return out.with_columns(
        (pl.col("_tr") / prior_mean).alias(f"range_expansion_{lookback}d")
    ).drop("_tr")


def body_range_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``body_range_ratio = |close - open| / (high - low)`` per symbol.

    Values near 1 mean a marubozu-style strong-trend bar; near 0 means a
    doji-style indecision bar. Null when ``high == low`` (no range).
    """
    out = df.sort(["symbol", "date"])
    span = pl.col("high") - pl.col("low")
    body = (pl.col("close") - pl.col("open")).abs()
    return out.with_columns(
        pl.when(span > 0)
        .then(body / span)
        .otherwise(None)
        .alias("body_range_ratio")
    )


def inside_bar(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_inside_bar`` per symbol.

    True when ``high[t] < high[t-1]`` AND ``low[t] > low[t-1]`` — today's
    range is fully contained within yesterday's. Null on the first row
    per symbol (no prior bar).
    """
    out = df.sort(["symbol", "date"])
    prev_high = pl.col("high").shift(1).over("symbol")
    prev_low = pl.col("low").shift(1).over("symbol")
    return out.with_columns(
        ((pl.col("high") < prev_high) & (pl.col("low") > prev_low)).alias(
            "is_inside_bar"
        )
    )
