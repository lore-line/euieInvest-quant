"""Behavioral / regime features: recency of prior runs, market regime, cap bucket.

Spec: CLAUDE.md §7 behavioral.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["days_since_last_20pct", "market_regime", "cap_bucket"]


def days_since_last_20pct(df: pl.DataFrame, lookahead: int = 30) -> pl.DataFrame:
    """Add ``days_since_last_20pct`` per symbol — recency of last winner-grade run.

    Uses ``compute_forward_winner_labels`` on past data, then computes the
    number of trading days since the most recent ``is_winner = True`` event.
    """
    raise NotImplementedError(
        "src/quant/features/behavioral.py: days_since_last_20pct — compute "
        "days since most recent +20%/30d winner event per symbol; "
        "see CLAUDE.md §7 behavioral.py."
    )


def market_regime(spy: pl.DataFrame) -> pl.DataFrame:
    """Return a per-date regime label for SPY (e.g. uptrend/downtrend/chop)."""
    raise NotImplementedError(
        "src/quant/features/behavioral.py: market_regime — classify SPY regime "
        "per date; see CLAUDE.md §7 behavioral.py."
    )


def cap_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``cap_bucket`` (mega/large/mid/small/micro) per symbol.

    Phase 1 note: market-cap data is not in the current snapshot. Source
    from a side-table or external join in a later pass.
    """
    raise NotImplementedError(
        "src/quant/features/behavioral.py: cap_bucket — assign mega/large/mid/"
        "small/micro per symbol; requires market-cap side-table (not in "
        "current snapshot)."
    )
