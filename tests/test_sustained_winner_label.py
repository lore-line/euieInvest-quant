"""Tests for ``quant.tracks.sustained_winner_label``.

Pins the cohort label per the user's verbatim spec + server-team final
20-trading-day unit (PR #1 issuecomment 2026-05-17 03:23). Any future
tweak to the threshold / horizon / price-floor must update these tests
in lockstep or the discovery output is invalid.
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
    sweep_specs,
)


# ---------- Spec catalog ----------

def test_spec_catalog_has_two_variants() -> None:
    assert set(SPECS.keys()) == {"standard", "strict"}


def test_standard_spec_matches_user_directive() -> None:
    """User: 'find every symbol that grew ≥20% over a 30d period that
    is over $1 share price'. Server-team final spec: 20 TRADING days
    horizon (calendar→trading conversion + clean indexing). Endpoint
    sustained ≥+10% at day 20 (= g/2 with g=20)."""
    s = SPECS["standard"]
    assert s.touch_threshold_pct == 20.0
    assert s.endpoint_threshold_pct == 10.0
    assert s.horizon_days == 20  # trading days, not calendar
    assert s.min_entry_price_usd == 1.00


def test_strict_spec_requires_endpoint_at_full_touch() -> None:
    """Strict: day-20 endpoint must also be ≥+20% (full 20% still held)."""
    s = SPECS["strict"]
    assert s.endpoint_threshold_pct == 20.0
    assert s.horizon_days == 20


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
    """Entry $100 → max touch $125 (day 5) → endpoint $112 (day 20).
    Standard spec: touch ≥+20% (25% ✓), endpoint ≥+10% (12% ✓) → True.

    Total prices: 21 rows (entry + 20 forward days)."""
    prices = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0] + [115.0] * 14 + [112.0]
    assert len(prices) == 21
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    label_col = "is_sustained_winner_standard"
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row[label_col][0] is True


def test_label_false_when_touch_threshold_not_met() -> None:
    """Entry $100 → max touch $118 (only 18%, < 20%) → endpoint $110.
    Standard spec: touch fails → False even though endpoint clears."""
    prices = [100.0] + [110.0] * 4 + [118.0] + [110.0] * 14 + [110.0]
    assert len(prices) == 21
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is False


def test_label_false_when_endpoint_reverts_flash_and_fade() -> None:
    """Entry $100 → touches $130 (day 5) but day-20 endpoint = $108.
    Standard spec: touch ✓ (30%), endpoint FAILS (8% < 10%) → False.

    This is the flash-and-fade rejection that distinguishes the new
    label from the old transient-touch label."""
    prices = [100.0, 110.0, 120.0, 125.0, 128.0, 130.0] + [109.0] * 14 + [108.0]
    assert len(prices) == 21
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is False


def test_label_true_near_threshold() -> None:
    """Tiny epsilon above both thresholds → True. (Note: real-world prices
    never land exactly on +20.000…% or +10.000…% so we don't test exact
    boundary — floating-point math makes 100 * 120/100 ≠ 20.0 exactly and
    behavior at the literal boundary is undefined-by-IEEE-754.)"""
    prices = [100.0] + [120.5] * 19 + [110.5]  # touch=20.5%, endpoint=10.5%
    assert len(prices) == 21
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is True


def test_label_filters_penny_stocks() -> None:
    """Entry close < $1.00 → label = null regardless of forward path."""
    prices = [0.50] + [0.70] * 19 + [0.60]  # entry below $1, would otherwise touch +40%
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    entry_row = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry_row["is_sustained_winner_standard"][0] is None


def test_label_null_when_forward_window_exceeds_data() -> None:
    """Last 20 rows per symbol have no forward window → null label."""
    prices = [100.0] * 15  # only 15 rows; horizon 20 → all rows are null-labeled
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    assert out["is_sustained_winner_standard"].is_null().all()


def test_label_independent_per_symbol() -> None:
    """AAA's forward window must not leak into BBB's computation."""
    rows = (
        _make_path("AAA", date(2026, 1, 1), [100.0] * 21)  # flat → label=False
        + _make_path("BBB", date(2026, 1, 1), [100.0] + [130.0] * 20)  # winner → True
    )
    df = pl.DataFrame(rows)
    out = compute_sustained_winner_label(df, SPECS["standard"])
    aaa_entry = out.filter((pl.col("symbol") == "AAA") & (pl.col("date") == date(2026, 1, 1)))
    bbb_entry = out.filter((pl.col("symbol") == "BBB") & (pl.col("date") == date(2026, 1, 1)))
    assert aaa_entry["is_sustained_winner_standard"][0] is False
    assert bbb_entry["is_sustained_winner_standard"][0] is True


def test_strict_variant_requires_full_endpoint() -> None:
    """Standard says True (endpoint +10%) but strict says False (endpoint < +20%).
    Entry $100, max touch $125 (day 1), endpoint $115."""
    prices = [100.0, 125.0] + [120.0] * 18 + [115.0]  # touch 25%, endpoint 15%
    assert len(prices) == 21
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    standard_out = compute_sustained_winner_label(df, SPECS["standard"])
    strict_out = compute_sustained_winner_label(df, SPECS["strict"])
    entry_date = date(2026, 1, 1)
    assert standard_out.filter(pl.col("date") == entry_date)["is_sustained_winner_standard"][0] is True
    assert strict_out.filter(pl.col("date") == entry_date)["is_sustained_winner_strict"][0] is False


def test_label_emits_diagnostic_columns() -> None:
    """forward_max_pct + forward_endpoint_pct must be on the output.
    Entry $100, peak $120 (day 2), endpoint $113."""
    prices = [100.0, 110.0, 120.0] + [115.0] * 17 + [113.0]
    assert len(prices) == 21
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_sustained_winner_label(df, SPECS["standard"])
    assert "forward_max_pct" in out.columns
    assert "forward_endpoint_pct" in out.columns
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["forward_max_pct"][0] == pytest.approx(20.0, abs=0.01)
    assert entry["forward_endpoint_pct"][0] == pytest.approx(13.0, abs=0.01)


# ---------- sweep_specs ----------

def test_sweep_default_yields_20_specs_descending() -> None:
    """Default sweep is 20% → 1% with 1% step = 20 specs, descending so
    the discovery loop can break at the first clearing g."""
    specs = sweep_specs()
    assert len(specs) == 20
    assert specs[0].touch_threshold_pct == 20.0
    assert specs[-1].touch_threshold_pct == 1.0
    # Descending
    for i in range(1, len(specs)):
        assert specs[i].touch_threshold_pct < specs[i-1].touch_threshold_pct


def test_sweep_endpoint_scales_with_g() -> None:
    """Default endpoint_ratio=0.5 → endpoint = g/2 at every threshold."""
    specs = sweep_specs()
    for s in specs:
        assert s.endpoint_threshold_pct == s.touch_threshold_pct * 0.5


def test_sweep_horizon_locked_to_20_trading_days() -> None:
    """Server-team final spec — all specs use 20 trading days."""
    specs = sweep_specs()
    for s in specs:
        assert s.horizon_days == 20


def test_sweep_name_format_for_pattern_naming() -> None:
    """Names use 2-digit zero-padded format (gNN) to slot into the
    platform-side pattern naming sw1_g{NN}_{rule_id}."""
    specs = sweep_specs()
    expected_names = [f"g{i:02d}" for i in range(20, 0, -1)]
    assert [s.name for s in specs] == expected_names


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
