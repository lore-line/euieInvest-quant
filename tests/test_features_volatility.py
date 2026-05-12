"""Tests for quant.features.volatility."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant.features.volatility import atr_pct, bb_squeeze, hv_ratio, nr4_nr7


def _build(closes: list[float], highs: list[float] | None = None,
           lows: list[float] | None = None) -> pl.DataFrame:
    n = len(closes)
    h = highs if highs is not None else [c * 1.01 for c in closes]
    l = lows if lows is not None else [c * 0.99 for c in closes]
    return pl.DataFrame(
        {
            "symbol": ["A"] * n,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "close": closes,
            "high": h,
            "low": l,
            "volume": [1000] * n,
        }
    )


def test_atr_pct_constant_on_constant_range() -> None:
    closes = [10.0] * 30
    df = _build(closes)  # high=10.1, low=9.9 → TR ≈ 0.2 every day
    out = atr_pct(df, window=14)
    vals = [v for v in out["atr_pct_14"].to_list() if v is not None]
    # Each non-null value should be ~ 0.2 / 10 = 0.02
    assert all(abs(v - 0.02) < 1e-6 for v in vals)


def test_bb_squeeze_smaller_when_low_variance() -> None:
    quiet = [10.0, 10.05] * 15
    noisy = [10.0, 11.0] * 15
    quiet_out = bb_squeeze(_build(quiet), window=20)
    noisy_out = bb_squeeze(_build(noisy), window=20)
    q_last = quiet_out["bb_squeeze_20"].tail(1).item()
    n_last = noisy_out["bb_squeeze_20"].tail(1).item()
    assert q_last < n_last


def test_nr4_flagged_on_smallest_range() -> None:
    # Days 0-2 have wide TR; day 3 has the smallest → is_nr4=True at day 3
    closes = [10.0, 10.0, 10.0, 10.0]
    highs = [11.0, 11.0, 11.0, 10.05]
    lows = [9.0, 9.0, 9.0, 9.95]
    df = _build(closes, highs, lows)
    out = nr4_nr7(df)
    # Only row 3 has window of 4 — and its TR is min
    vals = out["is_nr4"].to_list()
    assert vals[3] is True
    # Earlier rows are null (insufficient window)
    assert vals[2] is None


def test_hv_ratio_above_one_when_recent_more_volatile() -> None:
    # Quiet phase then noisy phase
    quiet = [10.0 + 0.001 * (i % 2) for i in range(60)]
    noisy_tail = [10.0 + 0.5 * (i % 2) for i in range(20)]
    closes = quiet + noisy_tail
    df = _build(closes)
    out = hv_ratio(df, short_window=10, long_window=60)
    # Last row's short-window HV >> long-window HV
    last = out["hv_ratio_10_60"].tail(1).item()
    assert last > 1.0
