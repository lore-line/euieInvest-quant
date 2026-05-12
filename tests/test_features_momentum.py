"""Tests for quant.features.momentum."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant.features.momentum import consecutive_run, macd, roc, rsi


def _build(closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "symbol": ["A"] * n,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "close": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "volume": [1000] * n,
        }
    )


def test_rsi_100_on_monotone_up() -> None:
    closes = [float(c) for c in range(1, 31)]
    df = _build(closes)
    out = rsi(df, windows=(14,))
    val = out["rsi_14"].tail(1).item()
    # All gains, no losses → RSI = 100
    assert val == 100.0


def test_rsi_0_on_monotone_down() -> None:
    closes = [float(c) for c in range(30, 0, -1)]
    df = _build(closes)
    out = rsi(df, windows=(14,))
    val = out["rsi_14"].tail(1).item()
    # All losses, no gains → RSI = 0
    assert val == 0.0


def test_macd_columns_present() -> None:
    closes = [10.0 + 0.1 * i for i in range(100)]
    df = _build(closes)
    out = macd(df, fast=12, slow=26, signal=9)
    assert {"macd_line", "macd_signal", "macd_hist"}.issubset(set(out.columns))


def test_roc_correct_value() -> None:
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    df = _build(closes)
    out = roc(df, windows=(2,))
    # roc_2 at row 5: 15/13 - 1 ≈ 0.1538
    val = out["roc_2"].tail(1).item()
    assert abs(val - (15 / 13 - 1)) < 1e-9


def test_consecutive_run_counts_up_days() -> None:
    closes = [10.0, 11.0, 12.0, 13.0, 12.5]
    df = _build(closes)
    out = consecutive_run(df)
    up = out["consec_up"].to_list()
    down = out["consec_down"].to_list()
    # Row 0: delta=null → up=0, down=0
    # Row 1: close went up → up=1, down=0
    # Row 2: up → up=2, down=0
    # Row 3: up → up=3, down=0
    # Row 4: down → up=0, down=1
    assert up == [0, 1, 2, 3, 0]
    assert down == [0, 0, 0, 0, 1]
