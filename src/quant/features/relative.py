"""Relative-strength features: vs SPY, vs sector, vs peer cluster.

Spec: CLAUDE.md §7 relative.py.
"""
from __future__ import annotations

import polars as pl

__all__ = ["rel_strength_spy", "rel_strength_sector", "peer_zscore"]


def rel_strength_spy(
    df: pl.DataFrame, spy: pl.DataFrame, lookback: int = 20
) -> pl.DataFrame:
    """Add ``rs_spy_{lookback}d`` per symbol (symbol return / SPY return)."""
    raise NotImplementedError(
        "src/quant/features/relative.py: rel_strength_spy — compute symbol vs "
        "SPY relative return over lookback days; see CLAUDE.md §7 relative.py."
    )


def rel_strength_sector(
    df: pl.DataFrame, peer_groups: dict[str, list[str]], lookback: int = 20
) -> pl.DataFrame:
    """Add ``rs_sector_{lookback}d`` per symbol (symbol return / sector return)."""
    raise NotImplementedError(
        "src/quant/features/relative.py: rel_strength_sector — compute symbol "
        "vs sector mean return; see CLAUDE.md §7 relative.py."
    )


def peer_zscore(
    df: pl.DataFrame, peer_groups: dict[str, list[str]], column: str
) -> pl.DataFrame:
    """Add ``{column}_peer_z`` per symbol (z-score within peer group, per date)."""
    raise NotImplementedError(
        "src/quant/features/relative.py: peer_zscore — compute z-score within "
        "peer group per date; see CLAUDE.md §7 relative.py."
    )
