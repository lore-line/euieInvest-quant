"""Momentum features: RSI, MACD, ROC, consecutive runs.

Spec: CLAUDE.md §7 momentum.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["rsi", "macd", "roc", "consecutive_run"]


def rsi(df: pl.DataFrame, windows: tuple[int, ...] = (2, 5, 14)) -> pl.DataFrame:
    """Add ``rsi_{N}`` for each N per symbol (Wilder's smoothing)."""
    raise NotImplementedError(
        "src/quant/features/momentum.py: rsi — compute Wilder's RSI for "
        "N in (2,5,14) per symbol; see CLAUDE.md §7 momentum.py."
    )


def macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Add ``macd_line``, ``macd_signal``, ``macd_hist`` per symbol."""
    raise NotImplementedError(
        "src/quant/features/momentum.py: macd — compute MACD line/signal/hist "
        "per symbol; see CLAUDE.md §7 momentum.py."
    )


def roc(
    df: pl.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 60)
) -> pl.DataFrame:
    """Add ``roc_{N}`` = close.pct_change(N) per symbol."""
    raise NotImplementedError(
        "src/quant/features/momentum.py: roc — compute rate-of-change for "
        "N in (5,10,20,60) per symbol; see CLAUDE.md §7 momentum.py."
    )


def consecutive_run(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``consec_up`` and ``consec_down`` run-lengths per symbol."""
    raise NotImplementedError(
        "src/quant/features/momentum.py: consecutive_run — compute up/down "
        "run-lengths per symbol; see CLAUDE.md §7 momentum.py."
    )
