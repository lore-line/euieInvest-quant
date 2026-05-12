"""Tests for quant.features.behavioral."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant.features.behavioral import (
    cap_bucket,
    days_since_last_20pct,
    market_regime,
)


def _build(sym: str, closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "symbol": [sym] * n,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "close": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "volume": [1000] * n,
        }
    )


def test_days_since_last_20pct_resets_on_event() -> None:
    # Flat at 10 for 30 days, then jumps to 13 (+30% vs the prior min) → event,
    # then flat at 13 for 5 days → days_since increments 0, 1, 2, ...
    closes = [10.0] * 30 + [13.0] * 5
    df = _build("A", closes)
    out = days_since_last_20pct(df, lookahead=30, threshold=0.20)
    vals = out["days_since_last_20pct"].to_list()
    # The first 30 rows have no winner-grade move → null
    assert all(v is None for v in vals[:30])
    # Row 30: today's close 13 vs min(close[1..29])=10 → 1.3 ≥ 1.2 → event → 0
    assert vals[30] == 0
    # Subsequent rows count up
    assert vals[31] == 1
    assert vals[32] == 2


def test_days_since_last_20pct_null_when_no_winner_seen() -> None:
    closes = [10.0] * 40
    df = _build("A", closes)
    out = days_since_last_20pct(df, lookahead=30, threshold=0.20)
    vals = out["days_since_last_20pct"].to_list()
    # Flat series → no event ever → all rows after the lookback window are still null
    assert all(v is None for v in vals)


def test_market_regime_classifies_uptrend() -> None:
    # 250 days of monotone increase → SMA50 < close, SMA200 < SMA50 → uptrend
    closes = [float(c) for c in range(1, 251)]
    spy = _build("SPY", closes)
    out = market_regime(spy)
    last = out["market_regime"].tail(1).item()
    assert last == "uptrend"


def test_market_regime_classifies_downtrend() -> None:
    closes = [float(c) for c in range(250, 0, -1)]
    spy = _build("SPY", closes)
    out = market_regime(spy)
    last = out["market_regime"].tail(1).item()
    assert last == "downtrend"


def test_cap_bucket_still_raises() -> None:
    df = _build("A", [10.0])
    with pytest.raises(NotImplementedError, match="market-cap"):
        cap_bucket(df)
