"""Temporal train/val/holdout splits for time-series data.

Hard rule (CLAUDE.md §8): no random splits, no peeking at the holdout
while iterating.
"""
from __future__ import annotations

from datetime import date

import polars as pl

__all__ = ["split_by_date"]


def split_by_date(
    df: pl.DataFrame,
    train_end: date,
    val_end: date,
    date_col: str = "date",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Partition ``df`` into ``(train, val, holdout)`` by date.

    - Train:   ``date <= train_end``
    - Val:     ``train_end < date <= val_end``
    - Holdout: ``date >  val_end``

    Asserts every row lands in exactly one bucket and that
    ``train_end < val_end``.
    """
    assert train_end < val_end, (
        f"train_end ({train_end}) must be strictly before val_end ({val_end})"
    )
    train = df.filter(pl.col(date_col) <= train_end)
    val = df.filter((pl.col(date_col) > train_end) & (pl.col(date_col) <= val_end))
    holdout = df.filter(pl.col(date_col) > val_end)
    assert train.height + val.height + holdout.height == df.height, (
        "split lost or duplicated rows"
    )
    return train, val, holdout
