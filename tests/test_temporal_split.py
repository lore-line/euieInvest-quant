"""Tests for split_by_date."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant.backtest.temporal import split_by_date


def test_disjoint_and_complete() -> None:
    dates = [
        date(2023, 1, 1),
        date(2023, 6, 1),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 6, 1),
        date(2024, 12, 31),
        date(2025, 1, 1),
        date(2025, 6, 1),
    ]
    df = pl.DataFrame({"date": dates, "x": list(range(len(dates)))})

    train, val, holdout = split_by_date(
        df, train_end=date(2023, 12, 31), val_end=date(2024, 12, 31)
    )

    assert train.height + val.height + holdout.height == df.height
    assert train.height > 0
    assert val.height > 0
    assert holdout.height > 0

    assert train["date"].max() == date(2023, 12, 31)
    assert val["date"].min() == date(2024, 1, 1)
    assert val["date"].max() == date(2024, 12, 31)
    assert holdout["date"].min() == date(2025, 1, 1)


def test_rejects_bad_boundaries() -> None:
    df = pl.DataFrame({"date": [date(2024, 1, 1)], "x": [1]})
    with pytest.raises(AssertionError):
        split_by_date(df, train_end=date(2024, 6, 1), val_end=date(2024, 1, 1))


def test_custom_date_col() -> None:
    df = pl.DataFrame(
        {
            "ts": [date(2023, 1, 1), date(2024, 6, 1), date(2025, 6, 1)],
            "v": [1, 2, 3],
        }
    )
    train, val, holdout = split_by_date(
        df,
        train_end=date(2023, 12, 31),
        val_end=date(2024, 12, 31),
        date_col="ts",
    )
    assert train.height == 1
    assert val.height == 1
    assert holdout.height == 1
