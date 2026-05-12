"""Tests for quant.features.price."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant.features.price import (
    band_position,
    n_day_high_low,
    sma_distance,
    sma_slope,
)


def _build(symbols_and_closes: dict[str, list[float]]) -> pl.DataFrame:
    rows = []
    for sym, closes in symbols_and_closes.items():
        for i, c in enumerate(closes):
            rows.append(
                {
                    "symbol": sym,
                    "date": date(2024, 1, 1) + timedelta(days=i),
                    "close": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "volume": 1000,
                }
            )
    return pl.DataFrame(rows)


def test_sma_distance_flat_series_is_one() -> None:
    df = _build({"A": [10.0] * 30})
    out = sma_distance(df, windows=(10,))
    vals = out["close_over_sma_10"].to_list()
    assert all(v is None for v in vals[:9])
    assert all(abs(v - 1.0) < 1e-9 for v in vals[9:])


def test_sma_distance_ramp_above_one() -> None:
    closes = [float(c) for c in range(1, 31)]
    df = _build({"A": closes})
    out = sma_distance(df, windows=(10,))
    # At row 9: close=10, SMA10=mean(1..10)=5.5; ratio=10/5.5
    val = out["close_over_sma_10"][9]
    assert abs(val - 10 / 5.5) < 1e-9


def test_sma_distance_per_symbol_isolated() -> None:
    df = _build({"A": [10.0] * 15, "B": [100.0] * 15})
    out = sma_distance(df, windows=(10,)).sort(["symbol", "date"])
    a_tail = out.filter(pl.col("symbol") == "A").tail(6)["close_over_sma_10"].to_list()
    b_tail = out.filter(pl.col("symbol") == "B").tail(6)["close_over_sma_10"].to_list()
    assert all(abs(v - 1.0) < 1e-9 for v in a_tail)
    assert all(abs(v - 1.0) < 1e-9 for v in b_tail)


def test_sma_slope_zero_on_flat() -> None:
    df = _build({"A": [10.0] * 60})
    out = sma_slope(df, window=20, lookback=5)
    vals = [v for v in out["sma20_slope_5d"].to_list() if v is not None]
    assert len(vals) > 0
    assert all(abs(v) < 1e-9 for v in vals)


def test_sma_slope_positive_on_ramp() -> None:
    closes = [float(c) for c in range(1, 61)]
    df = _build({"A": closes})
    out = sma_slope(df, window=20, lookback=5)
    vals = [v for v in out["sma20_slope_5d"].to_list() if v is not None]
    assert all(v > 0 for v in vals)


def test_band_position_zero_at_mean() -> None:
    """Two-step alternation around a constant mean → bb_position oscillates near 0."""
    closes = [10.0, 10.1] * 30
    df = _build({"A": closes})
    out = band_position(df, window=20)
    vals = [v for v in out["bb_position_20"].to_list() if v is not None]
    assert max(vals) < 2.0
    assert min(vals) > -2.0
    # Mean should be ~0 since the series oscillates symmetrically
    assert abs(sum(vals) / len(vals)) < 0.5


def test_n_day_high_low_at_high_when_close_near_high() -> None:
    closes = [float(c) for c in range(1, 21)]
    df = _build({"A": closes})  # high = close * 1.01
    out = n_day_high_low(df, windows=(10,))
    # Once the window fills, pct_of_10d_high should be close/max(high in window)
    # For monotone increasing, max(high) = high[t] = close[t]*1.01, so ratio = 1/1.01
    val = out["pct_of_10d_high"][15]
    assert abs(val - 1 / 1.01) < 1e-9


def test_n_day_high_low_per_symbol() -> None:
    df = _build({"A": [10.0] * 25, "B": [100.0] * 25})
    out = n_day_high_low(df, windows=(10,)).sort(["symbol", "date"])
    a = out.filter(pl.col("symbol") == "A")["pct_of_10d_high"].to_list()
    b = out.filter(pl.col("symbol") == "B")["pct_of_10d_high"].to_list()
    # Both flat → close/high = 10/10.1 = 100/101 = 1/1.01
    assert all(
        v is None or abs(v - 1 / 1.01) < 1e-9 for v in a + b
    )
