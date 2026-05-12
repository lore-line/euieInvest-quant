"""Tests for quant.features.volume."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant.features.volume import (
    accumulation_distribution,
    obv_slope,
    vol_mult,
)


def _build(
    sym: str, closes: list[float], volumes: list[int]
) -> pl.DataFrame:
    assert len(closes) == len(volumes)
    rows = [
        {
            "symbol": sym,
            "date": date(2024, 1, 1) + timedelta(days=i),
            "close": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "volume": v,
        }
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]
    return pl.DataFrame(rows)


def test_vol_mult_one_on_flat_volume() -> None:
    df = _build("A", [10.0] * 20, [1000] * 20)
    out = vol_mult(df, windows=(5,))
    vals = [v for v in out["vol_mult_5"].to_list() if v is not None]
    assert all(abs(v - 1.0) < 1e-9 for v in vals)


def test_vol_mult_spike_above_one() -> None:
    closes = [10.0] * 20
    volumes = [1000] * 19 + [5000]
    df = _build("A", closes, volumes)
    out = vol_mult(df, windows=(5,))
    # Last row's vol_mult_5 = 5000 / mean(last 5 volumes incl today) =
    # 5000 / ((1000*4 + 5000)/5) = 5000 / 1800 ≈ 2.78
    assert abs(out["vol_mult_5"].tail(1).item() - 5000 / 1800) < 1e-9


def test_vol_mult_null_on_zero_volume_window() -> None:
    """When the rolling volume mean is exactly 0 (delisted name with
    all-zero recent volume), vol_mult must be null — NOT inf. inf in
    the feature matrix poisons xgboost training. Regression-tests the
    rolling_mean>0 guard in vol_mult.
    """
    df = _build("A", [10.0] * 20, [0] * 20)
    out = vol_mult(df, windows=(5,))
    vals = out["vol_mult_5"].to_list()

    # First 4 rows null from insufficient window
    assert all(v is None for v in vals[:4])
    # Rows 4..19 have a fully-zero-volume window → null (not inf, not NaN)
    flat_vals = vals[4:]
    assert all(v is None for v in flat_vals), (
        f"zero-volume window should produce null, got {flat_vals}"
    )
    assert out.height == 20


def test_obv_slope_positive_on_rising_close() -> None:
    closes = [float(c) for c in range(1, 31)]
    df = _build("A", closes, [1000] * 30)
    out = obv_slope(df, lookback=10)
    vals = [v for v in out["obv_slope_10d"].to_list() if v is not None]
    # OBV climbs steadily on monotone up → slope positive
    assert all(v > 0 for v in vals)


def test_obv_slope_zero_on_flat_close() -> None:
    df = _build("A", [10.0] * 30, [1000] * 30)
    out = obv_slope(df, lookback=10)
    vals = [v for v in out["obv_slope_10d"].to_list() if v is not None]
    # No close moves → OBV doesn't change → slope = 0
    assert all(abs(v) < 1e-9 for v in vals)


def test_accumulation_distribution_positive_on_close_at_high() -> None:
    """When close == high every day, the A/D line should be strictly increasing."""
    # Force close = high via build helper's high = close*1.01? No, we need close == high.
    # Build manually.
    rows = []
    base = date(2024, 1, 1)
    for i in range(20):
        c = 10.0
        rows.append(
            {
                "symbol": "A",
                "date": base + timedelta(days=i),
                "close": c,
                "high": c,        # close == high
                "low": c * 0.95,
                "volume": 1000,
            }
        )
    df = pl.DataFrame(rows)
    out = accumulation_distribution(df)
    ad = out["ad_line"].to_list()
    # Each step contributes +volume (full bull MFM=+1 when close=high)
    assert ad[-1] > ad[0]


def test_accumulation_distribution_negative_on_close_at_low() -> None:
    rows = []
    base = date(2024, 1, 1)
    for i in range(20):
        c = 10.0
        rows.append(
            {
                "symbol": "A",
                "date": base + timedelta(days=i),
                "close": c,
                "high": c * 1.05,
                "low": c,         # close == low
                "volume": 1000,
            }
        )
    df = pl.DataFrame(rows)
    out = accumulation_distribution(df)
    ad = out["ad_line"].to_list()
    assert ad[-1] < ad[0]
