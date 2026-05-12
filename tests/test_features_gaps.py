"""Tests for quant.features.gaps."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant.features.gaps import (
    body_range_ratio,
    gap_pct,
    inside_bar,
    range_expansion,
)


def _build(
    highs: list[float], lows: list[float], closes: list[float] | None = None
) -> pl.DataFrame:
    n = len(highs)
    assert len(lows) == n
    c = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    return pl.DataFrame(
        {
            "symbol": ["A"] * n,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "close": c,
            "high": highs,
            "low": lows,
            "volume": [1000] * n,
        }
    )


def test_gap_pct_raises_until_open_lands() -> None:
    df = _build([10.0], [9.0])
    with pytest.raises(NotImplementedError, match="open"):
        gap_pct(df)


def test_body_range_ratio_raises_until_open_lands() -> None:
    df = _build([10.0], [9.0])
    with pytest.raises(NotImplementedError, match="open"):
        body_range_ratio(df)


def test_range_expansion_above_one_on_widening_bar() -> None:
    # Constant narrow range, then a single wide bar
    highs = [10.5] * 10 + [12.0]
    lows = [9.5] * 10 + [9.0]
    df = _build(highs, lows)
    out = range_expansion(df, lookback=5)
    last = out["range_expansion_5d"].tail(1).item()
    assert last > 1.0


def test_inside_bar_flagged() -> None:
    # Day 0: high=11, low=9. Day 1: high=10, low=10 → inside.
    df = _build([11.0, 10.0], [9.0, 10.0])
    out = inside_bar(df)
    assert out["is_inside_bar"].to_list() == [None, True]


def test_inside_bar_not_flagged_when_breakout() -> None:
    # Day 0: high=11, low=9. Day 1: high=12, low=10 → high breached upward.
    df = _build([11.0, 12.0], [9.0, 10.0])
    out = inside_bar(df)
    assert out["is_inside_bar"].to_list() == [None, False]
