"""Tests for compute_forward_winner_labels."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant.labels import compute_forward_winner_labels


def _build_symbol(symbol: str, closes: list[float]) -> pl.DataFrame:
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "date": [base + timedelta(days=i) for i in range(len(closes))],
            "close": closes,
        }
    )


def test_ramp_within_lookahead_is_winner() -> None:
    closes = [100.0] * 5 + [100.0 + i for i in range(1, 26)] + [125.0] * 5
    assert len(closes) == 35
    df = _build_symbol("RAMP", closes)
    out = compute_forward_winner_labels(df, lookahead=30, threshold=0.20).sort("date")
    assert "is_winner" in out.columns

    is_winner = out["is_winner"].to_list()
    # Row 0: forward 30 closes peak at 125 → 125/100 = 1.25 ≥ 1.20 → True
    assert is_winner[0] is True
    # Last 30 rows: insufficient forward data → null
    assert out.tail(30)["is_winner"].null_count() == 30


def test_flat_series_no_winners() -> None:
    closes = [50.0] * 60
    df = _build_symbol("FLAT", closes)
    out = compute_forward_winner_labels(df, lookahead=30, threshold=0.20).sort("date")
    is_winner = out["is_winner"].to_list()
    # First 30 rows: all False (flat → ratio 1.0 < 1.20)
    assert all(v is False for v in is_winner[:30])
    # Last 30 rows: null
    assert all(v is None for v in is_winner[30:])


def test_per_symbol_grouping() -> None:
    """Forward-windowing must not bleed across symbols."""
    closes_a = [10.0] * 35
    closes_b = [100.0] + [200.0] * 34
    df = pl.concat([_build_symbol("A", closes_a), _build_symbol("B", closes_b)])
    out = compute_forward_winner_labels(df, lookahead=30, threshold=0.20)

    a = out.filter(pl.col("symbol") == "A").sort("date")
    b = out.filter(pl.col("symbol") == "B").sort("date")

    a_wins = a["is_winner"].to_list()
    b_wins = b["is_winner"].to_list()

    # A is flat → no True values
    assert not any(v is True for v in a_wins)
    # B's first row should be a winner (forward closes are 200 → 200/100 = 2.0)
    assert b_wins[0] is True
    # Last 30 rows null for both symbols
    assert all(v is None for v in a_wins[5:])
    assert all(v is None for v in b_wins[5:])


def test_invalid_args() -> None:
    df = _build_symbol("X", [10.0] * 35)
    with pytest.raises(ValueError):
        compute_forward_winner_labels(df, lookahead=0)
    with pytest.raises(ValueError):
        compute_forward_winner_labels(df, threshold=0.0)
