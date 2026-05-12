"""Gap and range features.

The current snapshot schema (CLAUDE.md §4) does NOT include an ``open``
column. Features that intrinsically need ``open`` (overnight gap, body
ratio) remain scaffolded with NotImplementedError + a clear pointer to
the upstream change needed; features that only need ``high``, ``low``,
and ``close`` are implemented.

Spec: CLAUDE.md §7 (gaps.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["gap_pct", "range_expansion", "body_range_ratio", "inside_bar"]


def gap_pct(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``gap_pct = (open[t] - close[t-1]) / close[t-1]`` per symbol.

    BLOCKED: ``open`` is not present in the current snapshot schema.
    Either ask the euieInvest server team to add ``open`` to
    ``/api/v1/ohlcv`` (additive change, non-breaking per contract §2),
    or drop this feature from the v1 feature set.
    """
    raise NotImplementedError(
        "src/quant/features/gaps.py: gap_pct — requires 'open' column not "
        "present in the current snapshot. Request the server team add "
        "'open' to /api/v1/ohlcv (additive, non-breaking per contract §2)."
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

    BLOCKED: same as ``gap_pct`` — needs ``open``.
    """
    raise NotImplementedError(
        "src/quant/features/gaps.py: body_range_ratio — requires 'open' "
        "column not present in the current snapshot. See gap_pct() doc."
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
