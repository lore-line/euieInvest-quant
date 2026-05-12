"""Price-action features: SMA distances/slopes, band positions, N-day extremes.

Spec: CLAUDE.md §7 price.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["sma_distance", "sma_slope", "band_position", "n_day_high_low"]


def sma_distance(
    df: pl.DataFrame, windows: tuple[int, ...] = (10, 20, 50, 200)
) -> pl.DataFrame:
    """Add ``close_over_sma_{N}`` for each N. Per-symbol, date-sorted."""
    raise NotImplementedError(
        "src/quant/features/price.py: sma_distance — compute close/SMA{N} "
        "ratios per symbol for N in (10,20,50,200); see CLAUDE.md §7 price.py."
    )


def sma_slope(
    df: pl.DataFrame, window: int = 50, lookback: int = 5
) -> pl.DataFrame:
    """Add ``sma{window}_slope_{lookback}d`` per symbol."""
    raise NotImplementedError(
        "src/quant/features/price.py: sma_slope — compute SMA slope over "
        "lookback days per symbol; see CLAUDE.md §7 price.py."
    )


def band_position(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add Bollinger band-position features (close relative to ±2σ band)."""
    raise NotImplementedError(
        "src/quant/features/price.py: band_position — compute Bollinger band "
        "position per symbol; see CLAUDE.md §7 price.py."
    )


def n_day_high_low(
    df: pl.DataFrame, windows: tuple[int, ...] = (20, 60, 252)
) -> pl.DataFrame:
    """Add ``pct_of_{N}d_high`` and ``pct_of_{N}d_low`` per symbol."""
    raise NotImplementedError(
        "src/quant/features/price.py: n_day_high_low — compute price relative "
        "to rolling N-day high/low per symbol; see CLAUDE.md §7 price.py."
    )
