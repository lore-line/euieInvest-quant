"""Volume features: SMA multiples, OBV slope, accumulation/distribution.

All functions:
- Take a polars DataFrame with at least ``symbol``, ``date``, ``close``,
  ``high``, ``low``, ``volume`` columns
- Return the same df sorted by ``(symbol, date)`` with new feature
  columns appended

Spec: CLAUDE.md §7 (volume.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["vol_mult", "obv_slope", "accumulation_distribution"]


def vol_mult(
    df: pl.DataFrame, windows: tuple[int, ...] = (5, 10, 30, 60)
) -> pl.DataFrame:
    """Add ``vol_mult_{N}`` = volume / SMA{N}(volume) per symbol.

    A value > 1 means today's volume exceeds its N-day average; a value
    near 2 means a 2× volume day.
    """
    out = df.sort(["symbol", "date"])
    cols = [
        (
            pl.col("volume").cast(pl.Float64)
            / pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window_size=n, min_samples=n)
            .over("symbol")
        ).alias(f"vol_mult_{n}")
        for n in windows
    ]
    return out.with_columns(cols)


def obv_slope(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Add ``obv_slope_{lookback}d`` per symbol — OBV change over lookback.

    OBV (On-Balance Volume) cumulates signed volume:
    ``OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]``.

    Slope is the normalized change:
    ``slope = (OBV[t] - OBV[t-lookback]) / mean_volume_over_lookback``,
    which keeps the scale roughly comparable across symbols of different
    average liquidity.
    """
    out = df.sort(["symbol", "date"])
    close_diff = pl.col("close").diff().over("symbol")
    sign = (
        pl.when(close_diff > 0)
        .then(1.0)
        .when(close_diff < 0)
        .then(-1.0)
        .otherwise(0.0)
    )
    signed_vol = (sign * pl.col("volume").cast(pl.Float64)).alias("_signed_vol")
    out = out.with_columns(signed_vol)
    out = out.with_columns(
        pl.col("_signed_vol").cum_sum().over("symbol").alias("_obv")
    )
    obv_change = pl.col("_obv") - pl.col("_obv").shift(lookback).over("symbol")
    mean_vol = (
        pl.col("volume")
        .cast(pl.Float64)
        .rolling_mean(window_size=lookback, min_samples=lookback)
        .over("symbol")
    )
    return out.with_columns(
        (obv_change / (mean_vol * lookback)).alias(f"obv_slope_{lookback}d")
    ).drop(["_signed_vol", "_obv"])


def accumulation_distribution(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``ad_line`` (Accumulation/Distribution Line) per symbol.

    Money-flow multiplier × volume, then cumulated:

    - ``mfm = ((close - low) - (high - close)) / (high - low)`` (0 if
      high == low to avoid div-by-zero)
    - ``mfv = mfm * volume``
    - ``ad_line = cumsum(mfv)`` per symbol

    Positive A/D trend means accumulation (closing strength on volume);
    negative means distribution.
    """
    out = df.sort(["symbol", "date"])
    h = pl.col("high")
    l = pl.col("low")
    c = pl.col("close")
    range_hl = h - l
    mfm = pl.when(range_hl > 0).then(((c - l) - (h - c)) / range_hl).otherwise(0.0)
    mfv = mfm * pl.col("volume").cast(pl.Float64)
    return out.with_columns(mfv.cum_sum().over("symbol").alias("ad_line"))
