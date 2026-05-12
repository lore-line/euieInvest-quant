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
    highs: list[float],
    lows: list[float],
    closes: list[float] | None = None,
    opens: list[float] | None = None,
) -> pl.DataFrame:
    n = len(highs)
    assert len(lows) == n
    c = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    o = opens if opens is not None else c
    return pl.DataFrame(
        {
            "symbol": ["A"] * n,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "close": c,
            "high": highs,
            "low": lows,
            "volume": [1000] * n,
            "open": o,
        }
    )


def test_gap_pct_up_and_down() -> None:
    # Day 0: close=10. Day 1: open=11 → +10% gap. Day 2: open=10 from prev close=11 → -9.09% gap.
    df = _build(
        highs=[10.5, 11.5, 10.5],
        lows=[9.5, 10.5, 9.5],
        closes=[10.0, 11.0, 10.0],
        opens=[10.0, 11.0, 10.0],
    )
    out = gap_pct(df)
    g = out["gap_pct"].to_list()
    assert g[0] is None
    assert g[1] == pytest.approx(0.10)
    assert g[2] == pytest.approx(-1.0 / 11.0)


def test_body_range_ratio_marubozu_vs_doji() -> None:
    # Marubozu: open=9, close=11, high=11, low=9 → body=range → 1.0
    # Doji: open=close=10, high=11, low=9 → body=0 → 0.0
    df = _build(
        highs=[11.0, 11.0],
        lows=[9.0, 9.0],
        closes=[11.0, 10.0],
        opens=[9.0, 10.0],
    )
    out = body_range_ratio(df)
    r = out["body_range_ratio"].to_list()
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(0.0)


def test_body_range_ratio_null_when_no_range() -> None:
    df = _build(highs=[10.0], lows=[10.0], closes=[10.0], opens=[10.0])
    out = body_range_ratio(df)
    assert out["body_range_ratio"].to_list() == [None]


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
