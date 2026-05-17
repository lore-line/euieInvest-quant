"""Tests for ``quant.tracks.sustained_winner_label``.

Pins the cohort label per the user's verbatim spec — server team
coordination response (2026-05-17 03:08 UTC). Any future tweak to the
threshold / horizon / price-floor must update these tests in lockstep
or the discovery output is invalid.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant.tracks.sustained_winner_label import (
    SPECS,
    SustainedWinnerSpec,
    compute_sustained_winner_label,
    label_statistics,
)


# ---------- Spec catalog ----------

def test_spec_catalog_has_two_variants() -> None:
    assert set(SPECS.keys()) == {"standard", "strict"}


def test_standard_spec_matches_user_directive() -> None:
    """User: 'find every symbol that grew ≥20% over a 30d period that
    is over $1 share price'. Server-team added the +10% day-30 endpoint
    constraint to reject flash-and-fade."""
    s = SPECS["standard"]
    assert s.touch_threshold_pct == 20.0
    assert s.endpoint_threshold_pct == 10.0
    assert s.horizon_days == 30
    assert s.min_entry_price_usd == 1.00


def test_strict_spec_requires_endpoint_at_full_touch() -> None:
    """Strict: day-30 endpoint must also be ≥+20% (full 20% still held)."""
    s = SPECS["strict"]
    assert s.endpoint_threshold_pct == 20.0


def test_label_column_names_distinguish_variants() -> None:
    assert SPECS["standard"].label_column() == "is_sustained_winner_standard"
    assert SPECS["strict"].label_column() == "is_sustained_winner_strict"


# ---------- compute_sustained_winner_label ----------

def _make_path(symbol: str, start: date, prices: list[float]) -> list[dict]:
    """Construct a small synthetic price series for a symbol."""
    return [
        {"symbol": symbol, "date": start + timedelta(days=i),
         "close_adj": p, "open": p, "high": p, "low": p, "close": p, "volume": 1000}
        for i, p in enumerate(prices)
    ]


def test_label_true_when_both_thresholds_satisfied_standard() -> None:
    """Entry $100 → max touch $125 (day 5) → endpoint $112 (day 30).
    Standard spec: touch ≥+20% (25% ✓), endpoint ≥+10% (12% ✓) → True."""
    prices = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0] + [115.0] * 24 + [112.0]
    # Index: 0=entry, 1..30 = forward window, last (index 30) = day 30 endpoint
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    label_col = "is_sustained_winner_standard"
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row[label_col][0] is True


def test_label_false_when_touch_threshold_not_met() -> None:
    """Entry $100 → max touch $118 (only 18%, < 20%) → endpoint $110.
    Standard spec: touch fails → False even though endpoint clears."""
    prices = [100.0] + [110.0] * 4 + [118.0] + [110.0] * 24 + [110.0]
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is False


def test_label_false_when_endpoint_reverts_flash_and_fade() -> None:
    """Entry $100 → touches $130 (day 5) but day-30 endpoint = $108.
    Standard spec: touch ✓ (30%), endpoint FAILS (8% < 10%) → False.

    This is the flash-and-fade rejection that distinguishes the new
    label from the old transient-touch label."""
    prices = [100.0, 110.0, 120.0, 125.0, 128.0, 130.0] + [109.0] * 24 + [108.0]
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is False


def test_label_true_near_threshold() -> None:
    """Tiny epsilon above both thresholds → True. (Note: real-world prices
    never land exactly on +20.000…% or +10.000…% so we don't test exact
    boundary — floating-point math makes 100 * 120/100 ≠ 20.0 exactly and
    behavior at the literal boundary is undefined-by-IEEE-754.)"""
    prices = [100.0] + [120.5] * 29 + [110.5]  # touch=20.5%, endpoint=10.5%
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is True


def test_label_filters_penny_stocks() -> None:
    """Entry close < $1.00 → label = null regardless of forward path."""
    prices = [0.50] + [0.70] * 29 + [0.60]  # entry below $1, would otherwise touch +40%
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is None


def test_label_null_when_forward_window_exceeds_data() -> None:
    """Last 30 rows per symbol have no forward window → null label."""
    prices = [100.0] * 15  # only 15 rows; horizon 30 → all rows are null-labeled
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    assert out["is_sustained_winner_standard"].is_null().all()


def test_label_independent_per_symbol() -> None:
    """AAA's forward window must not leak into BBB's computation."""
    rows = (
        _make_path("AAA", date(2026, 1, 1), [100.0] * 31)  # flat → label=False
        + _make_path("BBB", date(2026, 1, 1), [100.0] + [130.0] * 30)  # winner → True
    )
    df = pl.DataFrame(rows)
    out = compute_sustained_winner_label(df, SPECS["standard"])
    aaa_entry = out.filter((pl.col("symbol") == "AAA") & (pl.col("date") == date(2026, 1, 1)))
    bbb_entry = out.filter((pl.col("symbol") == "BBB") & (pl.col("date") == date(2026, 1, 1)))
    assert aaa_entry["is_sustained_winner_standard"][0] is False
    assert bbb_entry["is_sustained_winner_standard"][0] is True


def test_strict_variant_requires_full_endpoint() -> None:
    """Standard says True (endpoint +10%) but strict says False (endpoint < +20%)."""
    prices = [100.0, 125.0] + [120.0] * 28 + [115.0]  # touch 25%, endpoint 15%
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    standard_out = compute_sustained_winner_label(df, SPECS["standard"])
    strict_out = compute_sustained_winner_label(df, SPECS["strict"])
    entry_date = date(2026, 1, 1)
    assert standard_out.filter(pl.col("date") == entry_date)["is_sustained_winner_standard"][0] is True
    assert strict_out.filter(pl.col("date") == entry_date)["is_sustained_winner_strict"][0] is False


def test_label_emits_diagnostic_columns() -> None:
    """forward_max_pct + forward_endpoint_pct must be on the output."""
    prices = [100.0, 110.0, 120.0] + [115.0] * 27 + [113.0]
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    assert "forward_max_pct" in out.columns
    assert "forward_endpoint_pct" in out.columns
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["forward_max_pct"][0] == pytest.approx(20.0, abs=0.01)
    assert entry["forward_endpoint_pct"][0] == pytest.approx(13.0, abs=0.01)


# ---------- label_statistics ----------

def test_statistics_winner_rate() -> None:
    df = pl.DataFrame({
        "spec": ["test"] * 4,
        "is_sustained_winner_standard": [True, True, False, False],
        "forward_max_pct": [25.0, 30.0, 5.0, -5.0],
        "forward_endpoint_pct": [15.0, 12.0, 2.0, -10.0],
    })
    stats = label_statistics(df, SPECS["standard"])
    assert stats["n_labelable_rows"] == 4
    assert stats["n_winners"] == 2
    assert stats["winner_rate"] == 0.5


def test_statistics_empty_input() -> None:
    empty = pl.DataFrame(schema={
        "is_sustained_winner_standard": pl.Boolean,
        "forward_max_pct": pl.Float64,
        "forward_endpoint_pct": pl.Float64,
    })
    stats = label_statistics(empty, SPECS["standard"])
    assert stats["n_labelable_rows"] == 0
    assert stats["winner_rate"] == 0.0
