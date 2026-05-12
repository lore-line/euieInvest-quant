"""Gap and range features: gaps, expansions, body ratios, inside bars.

Spec: CLAUDE.md §7 gaps.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["gap_pct", "range_expansion", "body_range_ratio", "inside_bar"]


def gap_pct(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``gap_pct`` = (open[t] - close[t-1]) / close[t-1] per symbol.

    Note: ``open`` is not present in the current snapshot schema; this
    function will need to source ``open`` from a future feature build or
    fall back to overnight ``low``-anchored proxies. See CLAUDE.md §7 gaps.py.
    """
    raise NotImplementedError(
        "src/quant/features/gaps.py: gap_pct — compute overnight gap %; "
        "requires open column (not in current snapshot — see docstring)."
    )


def range_expansion(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """Add ``range_expansion_{lookback}d`` = TR(t) / mean(TR over lookback)."""
    raise NotImplementedError(
        "src/quant/features/gaps.py: range_expansion — compute TR vs lookback "
        "mean per symbol; see CLAUDE.md §7 gaps.py."
    )


def body_range_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``body_range_ratio`` = |close - open| / (high - low) per symbol.

    Requires ``open`` (see ``gap_pct`` note).
    """
    raise NotImplementedError(
        "src/quant/features/gaps.py: body_range_ratio — compute |close-open|/"
        "(high-low); requires open column (not in current snapshot)."
    )


def inside_bar(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_inside_bar`` flag per symbol (high[t]<high[t-1] & low[t]>low[t-1])."""
    raise NotImplementedError(
        "src/quant/features/gaps.py: inside_bar — flag inside bars per symbol; "
        "see CLAUDE.md §7 gaps.py."
    )
