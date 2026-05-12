"""Price-action features: SMA distances/slopes, band positions, N-day extremes.

All functions:
- Take a polars DataFrame with at least ``symbol``, ``date``, ``close``,
  ``high``, and ``low`` columns
- Return the same df sorted by ``(symbol, date)`` with new feature
  columns appended
- Use TRAILING windows only — never reference future rows

Spec: CLAUDE.md §7 (price.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["sma_distance", "sma_slope", "band_position", "n_day_high_low"]


def sma_distance(
    df: pl.DataFrame, windows: tuple[int, ...] = (10, 20, 50, 200)
) -> pl.DataFrame:
    """Add ``close_over_sma_{N}`` for each N. Per-symbol, date-sorted.

    ``close_over_sma_N = close[t] / SMA_N(close)[t]``. Values > 1 mean
    price is above its N-day average; < 1 means below. Null until the
    rolling window fills.
    """
    out = df.sort(["symbol", "date"])
    cols = [
        (
            pl.col("close")
            / pl.col("close")
            .rolling_mean(window_size=n, min_samples=n)
            .over("symbol")
        ).alias(f"close_over_sma_{n}")
        for n in windows
    ]
    return out.with_columns(cols)


def sma_slope(
    df: pl.DataFrame, window: int = 50, lookback: int = 5
) -> pl.DataFrame:
    """Add ``sma{window}_slope_{lookback}d`` per symbol.

    Slope = ``(SMA_w[t] - SMA_w[t-lookback]) / SMA_w[t-lookback]``.
    Positive means the moving average is rising over the lookback.
    """
    out = df.sort(["symbol", "date"])
    sma = (
        pl.col("close")
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
    )
    sma_prev = sma.shift(lookback).over("symbol")
    return out.with_columns(
        ((sma - sma_prev) / sma_prev).alias(f"sma{window}_slope_{lookback}d")
    )


def band_position(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add Bollinger band position per symbol.

    ``bb_position_{window} = (close - SMA_w) / (2 * STD_w)``. Values near
    0 mean close is at the SMA; ±1 means at the ±2σ band.

    Null when the rolling window is perfectly flat (std == 0), which
    avoids 0/0 = NaN from polars float division. Flat windows happen on
    halted or illiquid names — they're rare but real, and a NaN in the
    feature matrix poisons downstream xgboost training. Prefer null,
    which xgboost handles natively.
    """
    out = df.sort(["symbol", "date"])
    mean = (
        pl.col("close")
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
    )
    std = (
        pl.col("close")
        .rolling_std(window_size=window, min_samples=window)
        .over("symbol")
    )
    return out.with_columns(
        pl.when(std > 0)
        .then((pl.col("close") - mean) / (2 * std))
        .otherwise(None)
        .alias(f"bb_position_{window}")
    )


def n_day_high_low(
    df: pl.DataFrame, windows: tuple[int, ...] = (20, 60, 252)
) -> pl.DataFrame:
    """Add ``pct_of_{N}d_high`` and ``pct_of_{N}d_low`` per symbol.

    - ``pct_of_N_high = close[t] / max(high[t-N+1..t])`` → near 1 means
      at the N-day high
    - ``pct_of_N_low = close[t] / min(low[t-N+1..t])`` → near 1 means
      at the N-day low
    """
    out = df.sort(["symbol", "date"])
    cols = []
    for n in windows:
        cols.append(
            (
                pl.col("close")
                / pl.col("high")
                .rolling_max(window_size=n, min_samples=n)
                .over("symbol")
            ).alias(f"pct_of_{n}d_high")
        )
        cols.append(
            (
                pl.col("close")
                / pl.col("low")
                .rolling_min(window_size=n, min_samples=n)
                .over("symbol")
            ).alias(f"pct_of_{n}d_low")
        )
    return out.with_columns(cols)
