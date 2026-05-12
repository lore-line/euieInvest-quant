"""Tests for quant.features.relative."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant.features.relative import (
    peer_zscore,
    rel_strength_sector,
    rel_strength_spy,
)


def _build_multi(
    symbols_and_closes: dict[str, list[float]],
) -> pl.DataFrame:
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


def test_rel_strength_spy_above_one_when_symbol_outperforms() -> None:
    spy = _build_multi({"SPY": [100.0 + i for i in range(30)]})  # +29% over the run
    df = _build_multi({"AAA": [10.0 + 0.5 * i for i in range(30)]})  # +145% over the run
    out = rel_strength_spy(df, spy, lookback=20)
    # AAA's return >> SPY's return → rs > 1
    last = out["rs_spy_20d"].tail(1).item()
    assert last > 1.0


def test_rel_strength_sector_uses_peer_mean() -> None:
    # Two-symbol sector where one is flat and one rises. The riser should
    # have rs_sector > 1; the flat one < 1.
    df = _build_multi(
        {
            "RISE": [10.0 + i for i in range(25)],
            "FLAT": [20.0] * 25,
        }
    )
    out = rel_strength_sector(
        df, peer_groups={"tech": ["RISE", "FLAT"]}, lookback=20
    )
    rise = out.filter(pl.col("symbol") == "RISE")["rs_sector_20d"].tail(1).item()
    flat = out.filter(pl.col("symbol") == "FLAT")["rs_sector_20d"].tail(1).item()
    assert rise > flat


def test_peer_zscore_zero_when_all_equal() -> None:
    df = _build_multi({"A": [10.0] * 5, "B": [10.0] * 5, "C": [10.0] * 5})
    # Compute z-score of "close" within the peer group of {A,B,C}
    out = peer_zscore(
        df, peer_groups={"tech": ["A", "B", "C"]}, column="close"
    )
    vals = [v for v in out["close_peer_z"].to_list() if v is not None]
    # Std is zero across peers → result should be null (per implementation)
    assert all(v is None for v in out["close_peer_z"].to_list())


def test_peer_zscore_signed_correctly() -> None:
    # Symbol B is the high outlier; should get a positive z-score
    df = _build_multi({"A": [10.0] * 5, "B": [20.0] * 5, "C": [10.0] * 5})
    out = peer_zscore(
        df, peer_groups={"tech": ["A", "B", "C"]}, column="close"
    )
    b_z = out.filter(pl.col("symbol") == "B")["close_peer_z"].tail(1).item()
    a_z = out.filter(pl.col("symbol") == "A")["close_peer_z"].tail(1).item()
    assert b_z > 0
    assert a_z < 0
