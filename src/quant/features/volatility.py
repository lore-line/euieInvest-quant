"""Volatility features: ATR%, Bollinger squeeze, NR4/NR7, HV ratios.

Spec: CLAUDE.md §7 volatility.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["atr_pct", "bb_squeeze", "nr4_nr7", "hv_ratio"]


def atr_pct(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    """Add ``atr_pct_{window}`` = ATR(window) / close per symbol."""
    raise NotImplementedError(
        "src/quant/features/volatility.py: atr_pct — compute ATR(window)/close "
        "per symbol; see CLAUDE.md §7 volatility.py."
    )


def bb_squeeze(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add a Bollinger-squeeze ratio (band width / SMA) per symbol."""
    raise NotImplementedError(
        "src/quant/features/volatility.py: bb_squeeze — compute Bollinger band "
        "width / SMA per symbol; see CLAUDE.md §7 volatility.py."
    )


def nr4_nr7(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_nr4`` and ``is_nr7`` inside-bar flags per symbol."""
    raise NotImplementedError(
        "src/quant/features/volatility.py: nr4_nr7 — flag NR4 and NR7 inside-bars "
        "per symbol; see CLAUDE.md §7 volatility.py."
    )


def hv_ratio(
    df: pl.DataFrame, short_window: int = 10, long_window: int = 60
) -> pl.DataFrame:
    """Add ``hv_ratio_{short}_{long}`` = HV(short) / HV(long) per symbol."""
    raise NotImplementedError(
        "src/quant/features/volatility.py: hv_ratio — compute historical "
        "volatility ratio (short/long) per symbol; see CLAUDE.md §7 volatility.py."
    )
