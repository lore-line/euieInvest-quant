"""Volume features: SMA multiples, OBV slope, accumulation/distribution.

Spec: CLAUDE.md §7 volume.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["vol_mult", "obv_slope", "accumulation_distribution"]


def vol_mult(
    df: pl.DataFrame, windows: tuple[int, ...] = (5, 10, 30, 60)
) -> pl.DataFrame:
    """Add ``vol_mult_{N}`` = volume / SMA{N}(volume) per symbol."""
    raise NotImplementedError(
        "src/quant/features/volume.py: vol_mult — compute volume / SMA{N}(volume) "
        "for N in (5,10,30,60) per symbol; see CLAUDE.md §7 volume.py."
    )


def obv_slope(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Add ``obv_slope_{lookback}d`` per symbol (OBV regressed over lookback)."""
    raise NotImplementedError(
        "src/quant/features/volume.py: obv_slope — compute On-Balance-Volume "
        "slope over lookback days per symbol; see CLAUDE.md §7 volume.py."
    )


def accumulation_distribution(df: pl.DataFrame) -> pl.DataFrame:
    """Add the Accumulation/Distribution Line per symbol."""
    raise NotImplementedError(
        "src/quant/features/volume.py: accumulation_distribution — compute "
        "the A/D Line per symbol; see CLAUDE.md §7 volume.py."
    )
