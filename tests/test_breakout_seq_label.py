"""Tests for `quant.tracks.breakout_seq_label`.

Pins the cohort label + realized-return mechanic per server-team
commission (PR #1 issuecomment-4469607665 2026-05-17 06:29Z).
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant.tracks.breakout_seq_label import (
    PATTERN_PREFIX,
    SPEC_DEFAULT,
    BreakoutSeqSpec,
    compute_breakout_seq_label,
    compute_realized_returns_60td,
    label_statistics,
)


# ---------- Spec ----------

def test_default_spec_matches_commission() -> None:
    """Server-team commission: g=20%, horizon=60td, min_price=$1."""
    s = SPEC_DEFAULT
    assert s.name == "g20"
    assert s.touch_threshold_pct == 20.0
    assert s.horizon_days == 60
    assert s.min_entry_price_usd == 1.00


def test_pattern_naming_format() -> None:
    """Per server spec: bsq60_g20_rule_{rule_id}"""
    s = SPEC_DEFAULT
    assert s.pattern(42) == "bsq60_g20_rule_42"
    assert s.pattern("abc") == "bsq60_g20_rule_abc"
    assert PATTERN_PREFIX == "bsq60"  # the g{NN} part is contributed by spec.name


def test_label_column_name() -> None:
    assert SPEC_DEFAULT.label_column() == "is_breakout_seq_g20"


# ---------- compute_breakout_seq_label ----------

def _make_path(symbol: str, start: date, prices: list[float]) -> list[dict]:
    return [
        {"symbol": symbol, "date": start + timedelta(days=i),
         "close_adj": p, "open": p, "high": p, "low": p, "close": p, "volume": 1000}
        for i, p in enumerate(prices)
    ]


def test_label_true_when_peak_meets_threshold() -> None:
    """Entry $100 → peak $120.5 (day 30 of 60) → True.
    (Using +20.5% not +20.0% to avoid IEEE 754 exact-boundary noise.)"""
    # 61 rows: entry + 60 forward days. Peak hits +20.5% mid-window.
    prices = [100.0] + [110.0] * 29 + [120.5] + [115.0] * 30
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_breakout_seq_label(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["is_breakout_seq_g20"][0] is True
    assert entry["forward_max_pct_60td"][0] == pytest.approx(20.5, abs=0.01)


def test_label_false_when_peak_below_threshold() -> None:
    """Entry $100 → peak $118 (only 18%) → False."""
    prices = [100.0] + [110.0] * 29 + [118.0] + [110.0] * 30
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_breakout_seq_label(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["is_breakout_seq_g20"][0] is False


def test_label_true_even_if_peak_late_in_window() -> None:
    """Peak at day 60 (last day) should still count — NO sustained constraint."""
    prices = [100.0] + [105.0] * 59 + [125.0]
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_breakout_seq_label(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["is_breakout_seq_g20"][0] is True


def test_label_true_even_if_endpoint_reverts() -> None:
    """Peak hits +20% at day 5, then reverts to BELOW entry at day 60.
    Sustained_winner would say FALSE; breakout_seq says TRUE — that's
    the whole point of the new cohort."""
    prices = [100.0, 110.0, 115.0, 118.0, 121.0, 130.0] + [80.0] * 54 + [70.0]
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_breakout_seq_label(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["is_breakout_seq_g20"][0] is True


def test_label_filters_penny_stocks() -> None:
    prices = [0.50] + [0.70] * 60  # would touch +40% but entry < $1
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_breakout_seq_label(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["is_breakout_seq_g20"][0] is None


def test_label_null_when_forward_window_exceeds_data() -> None:
    prices = [100.0] * 40  # only 40 rows; need 60 forward
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_breakout_seq_label(df, SPEC_DEFAULT)
    assert out["is_breakout_seq_g20"].is_null().all()


# ---------- compute_realized_returns_60td ----------

def test_realized_returns_target_hit_exits_at_first_hit() -> None:
    """Target hit at day 5 → exit at day 5 with realized gain ~+25%
    and hold = 5 days. Subsequent peaks ignored (we already exited)."""
    prices = [100.0, 105.0, 110.0, 115.0, 118.0, 125.0] + [150.0] * 55
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_realized_returns_60td(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["bsq_exit_reason"][0] == "target_hit"
    assert entry["bsq_hold_trading_days"][0] == 5
    assert entry["bsq_realized_gain_pct"][0] == pytest.approx(25.0, abs=0.01)


def test_realized_returns_day60_cap_when_no_target_hit() -> None:
    """Never crosses +20% → exit at day 60 close."""
    prices = [100.0] + [110.0] * 59 + [115.0]
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_realized_returns_60td(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["bsq_exit_reason"][0] == "day60_cap"
    assert entry["bsq_hold_trading_days"][0] == 60
    assert entry["bsq_realized_gain_pct"][0] == pytest.approx(15.0, abs=0.01)


def test_realized_returns_day60_cap_with_loss() -> None:
    """Stock goes down — day 60 cap exits at a loss."""
    prices = [100.0] + [95.0] * 59 + [85.0]
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_realized_returns_60td(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["bsq_exit_reason"][0] == "day60_cap"
    assert entry["bsq_realized_gain_pct"][0] == pytest.approx(-15.0, abs=0.01)


def test_realized_returns_target_hit_at_day1() -> None:
    """Earliest possible target hit: day 1 close already at +20%."""
    prices = [100.0, 120.5] + [200.0] * 59
    assert len(prices) == 61
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_realized_returns_60td(df, SPEC_DEFAULT)
    entry = out.filter(pl.col("date") == date(2026, 1, 1))
    assert entry["bsq_exit_reason"][0] == "target_hit"
    assert entry["bsq_hold_trading_days"][0] == 1
    assert entry["bsq_realized_gain_pct"][0] == pytest.approx(20.5, abs=0.01)


def test_realized_returns_null_when_forward_short() -> None:
    """Insufficient forward data → realized columns are null."""
    prices = [100.0] * 30  # need 60 forward
    df = pl.DataFrame(_make_path("AAA", date(2026, 1, 1), prices))
    out = compute_realized_returns_60td(df, SPEC_DEFAULT)
    # All rows should have null realized_gain (no forward window)
    assert out["bsq_realized_gain_pct"].is_null().all()


# ---------- label_statistics ----------

def test_label_statistics_basic() -> None:
    df = pl.DataFrame({
        "is_breakout_seq_g20": [True, True, True, False, False],
        "forward_max_pct_60td": [25.0, 30.0, 22.0, 5.0, -5.0],
    })
    stats = label_statistics(df, SPEC_DEFAULT)
    assert stats["n_labelable_rows"] == 5
    assert stats["n_winners"] == 3
    assert stats["winner_rate"] == 0.6
