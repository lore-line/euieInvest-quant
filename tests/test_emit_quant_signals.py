"""Tests for ``quant.tracks.emit_quant_signals``.

Pins the contract-conforming behavior of the entry signal emitter
(per docs/quant-signal-contract-v1.md on the trading-platform side,
lore-line/euieInvest@ac11d69) — the platform-side ingest validates
every row, so we mirror those validation rules client-side here.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.tracks.emit_quant_signals import (
    _CONTRACT_SCHEMA,
    DEFAULT_DEDUP_WINDOW_DAYS,
    DEFAULT_EXPECTED_HORIZON_DAYS,
    DEFAULT_EXPECTED_RETURN_PCT,
    LIFT_STRENGTH_DIVISOR,
    _apply_dedup,
    _build_signal_rows,
    _next_sequence,
    _validate_signal_row,
)
from quant.tracks.walkforward_validate import Rule


# ---------- _validate_signal_row ----------

def _good_row(**overrides) -> dict:
    base = {
        "signal_id": "2026-05-16-001_AAPL_2026-05-16_ENTRY_L1_42",
        "symbol": "AAPL",
        "signal_date": "2026-05-16",
        "signal_type": "ENTRY",
        "signal_strength": 0.85,
        "pattern": "L1_42",
        "expected_horizon_days": 30,
        "expected_return_pct": 20.0,
        "conditions_json": "[]",
    }
    base.update(overrides)
    return base


def test_validate_accepts_good_row() -> None:
    valid, err = _validate_signal_row(_good_row())
    assert valid is True
    assert err is None


def test_validate_rejects_strength_below_0() -> None:
    valid, err = _validate_signal_row(_good_row(signal_strength=-0.1))
    assert valid is False
    assert "signal_strength" in err
    assert "out of" in err


def test_validate_rejects_strength_above_1() -> None:
    valid, err = _validate_signal_row(_good_row(signal_strength=1.5))
    assert valid is False
    assert "signal_strength" in err


def test_validate_accepts_strength_at_bounds() -> None:
    for s in [0.0, 1.0]:
        valid, err = _validate_signal_row(_good_row(signal_strength=s))
        assert valid is True, f"strength={s} should be valid"


def test_validate_rejects_bad_signal_type() -> None:
    valid, err = _validate_signal_row(_good_row(signal_type="SELL"))
    assert valid is False
    assert "signal_type" in err


def test_validate_accepts_entry_and_exit_only() -> None:
    for st in ["ENTRY", "EXIT"]:
        valid, _ = _validate_signal_row(_good_row(signal_type=st))
        assert valid is True, f"signal_type={st} should be valid"


def test_validate_rejects_lowercase_entry() -> None:
    """Contract says signal_type is case-sensitive enum."""
    valid, err = _validate_signal_row(_good_row(signal_type="entry"))
    assert valid is False


def test_validate_rejects_empty_signal_id() -> None:
    valid, err = _validate_signal_row(_good_row(signal_id=""))
    assert valid is False
    assert "signal_id" in err


def test_validate_rejects_bad_date_format() -> None:
    valid, err = _validate_signal_row(_good_row(signal_date="05/16/2026"))
    assert valid is False
    assert "signal_date" in err


def test_validate_rejects_empty_symbol() -> None:
    valid, err = _validate_signal_row(_good_row(symbol=""))
    assert valid is False
    assert "symbol" in err


# ---------- _apply_dedup ----------

def test_dedup_empty_passthrough() -> None:
    empty = pl.DataFrame(schema={"symbol": pl.String, "date": pl.Date, "rule_key": pl.String})
    out = _apply_dedup(empty, 30)
    assert out.height == 0


def test_dedup_keeps_first_firing_within_window() -> None:
    """A rule firing 5 days in a row should emit on day 1 only."""
    raw = pl.DataFrame({
        "symbol": ["AAPL"] * 5,
        "date": [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3), date(2026, 5, 4), date(2026, 5, 5)],
        "rule_key": ["L1_42"] * 5,
    })
    out = _apply_dedup(raw, dedup_window_days=30)
    assert out.height == 1
    assert out["date"][0] == date(2026, 5, 1)


def test_dedup_re_emits_after_window_passes() -> None:
    """A rule that fires day 1 and day 31 should emit BOTH (window=30 means
    re-emit allowed when gap > 30, so day 32 would be the first re-emit)."""
    raw = pl.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "date": [date(2026, 5, 1), date(2026, 5, 31), date(2026, 6, 1)],
        "rule_key": ["L1_42"] * 3,
    })
    out = _apply_dedup(raw, dedup_window_days=30)
    # day 31 - day 1 = 30 days (NOT > 30) → drop
    # day 1 day 1, day 32+ allowed
    assert out.height == 2
    out_dates = sorted(out["date"].to_list())
    assert out_dates == [date(2026, 5, 1), date(2026, 6, 1)]


def test_dedup_independent_per_symbol() -> None:
    """Different symbols with the same rule key should both emit."""
    raw = pl.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [date(2026, 5, 1), date(2026, 5, 1)],
        "rule_key": ["L1_42", "L1_42"],
    })
    out = _apply_dedup(raw, 30)
    assert out.height == 2
    assert set(out["symbol"].to_list()) == {"AAPL", "MSFT"}


def test_dedup_independent_per_rule_key() -> None:
    """Same symbol with different rules should emit both."""
    raw = pl.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [date(2026, 5, 1), date(2026, 5, 1)],
        "rule_key": ["L1_42", "L2_99"],
    })
    out = _apply_dedup(raw, 30)
    assert out.height == 2


# ---------- _build_signal_rows ----------

def test_build_rows_filters_to_emission_dates() -> None:
    """Dedup window goes back further than emission window — only emission
    dates should appear in the output."""
    deduped = pl.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [date(2026, 4, 1), date(2026, 5, 16)],
        "rule_key": ["L1_42", "L1_42"],
    })
    rule_definitions = {
        "L1_42": Rule(
            rule_key="L1_42", source_track="step3c", source_label_or_regime="L1",
            conditions=[{"feature": "atr_pct_14", "op": ">", "threshold": 0.05}],
            train_lift=2.5, train_precision=0.45,
        )
    }
    survivor_lift = {"L1_42": 2.1}
    out = _build_signal_rows(deduped, survivor_lift, rule_definitions, "2026-05-16-001",
                              emission_dates={date(2026, 5, 16)})
    assert out.height == 1
    assert out["signal_date"][0] == "2026-05-16"


def test_build_rows_signal_id_format() -> None:
    deduped = pl.DataFrame({
        "symbol": ["AAPL"], "date": [date(2026, 5, 16)], "rule_key": ["L1_42"],
    })
    rule_definitions = {
        "L1_42": Rule(rule_key="L1_42", source_track="step3c", source_label_or_regime="L1",
                       conditions=[], train_lift=2.5, train_precision=0.45)
    }
    out = _build_signal_rows(deduped, {"L1_42": 2.1}, rule_definitions, "2026-05-16-001",
                              {date(2026, 5, 16)})
    # signal_id includes pattern to preserve uniqueness when multiple rules
    # fire on the same (symbol, date) — extends the spec's recommended format.
    assert out["signal_id"][0] == "2026-05-16-001_AAPL_2026-05-16_ENTRY_L1_42"


def test_build_rows_signal_strength_formula() -> None:
    """signal_strength = min(1.0, lift / 3.0) per the contract spec."""
    deduped = pl.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "date": [date(2026, 5, 16)] * 3,
        "rule_key": ["R1", "R2", "R3"],
    })
    rule_definitions = {k: Rule(rule_key=k, source_track="step3a", source_label_or_regime="",
                                  conditions=[], train_lift=0.0, train_precision=0.0) for k in ["R1", "R2", "R3"]}
    survivor_lift = {"R1": 1.5, "R2": 3.0, "R3": 5.0}  # → 0.5, 1.0, 1.0 (clamped)
    out = _build_signal_rows(deduped, survivor_lift, rule_definitions, "X", {date(2026, 5, 16)})
    by_pattern = {r["pattern"]: r["signal_strength"] for r in out.iter_rows(named=True)}
    assert by_pattern["R1"] == pytest.approx(0.5)
    assert by_pattern["R2"] == pytest.approx(1.0)
    assert by_pattern["R3"] == pytest.approx(1.0)  # clamped


def test_build_rows_conditions_json_roundtrips() -> None:
    """conditions_json must be a JSON string of the rule's condition list."""
    conditions = [
        {"feature": "atr_pct_14", "op": ">", "threshold": 0.05},
        {"feature": "rsi_14", "op": "<", "threshold": 70},
    ]
    deduped = pl.DataFrame({
        "symbol": ["AAPL"], "date": [date(2026, 5, 16)], "rule_key": ["L1_42"],
    })
    rule_definitions = {
        "L1_42": Rule(rule_key="L1_42", source_track="step3c", source_label_or_regime="L1",
                       conditions=conditions, train_lift=2.5, train_precision=0.45)
    }
    out = _build_signal_rows(deduped, {"L1_42": 2.1}, rule_definitions, "X", {date(2026, 5, 16)})
    import json
    parsed = json.loads(out["conditions_json"][0])
    assert parsed == conditions


def test_build_rows_horizon_and_return_defaults() -> None:
    deduped = pl.DataFrame({
        "symbol": ["AAPL"], "date": [date(2026, 5, 16)], "rule_key": ["L1_42"],
    })
    rule_definitions = {
        "L1_42": Rule(rule_key="L1_42", source_track="step3c", source_label_or_regime="L1",
                       conditions=[], train_lift=2.5, train_precision=0.45)
    }
    out = _build_signal_rows(deduped, {"L1_42": 2.1}, rule_definitions, "X", {date(2026, 5, 16)})
    assert out["expected_horizon_days"][0] == DEFAULT_EXPECTED_HORIZON_DAYS == 30
    assert out["expected_return_pct"][0] == DEFAULT_EXPECTED_RETURN_PCT == 20.0


def test_build_rows_schema_matches_contract() -> None:
    deduped = pl.DataFrame({
        "symbol": ["AAPL"], "date": [date(2026, 5, 16)], "rule_key": ["L1_42"],
    })
    rule_definitions = {
        "L1_42": Rule(rule_key="L1_42", source_track="step3c", source_label_or_regime="L1",
                       conditions=[], train_lift=2.5, train_precision=0.45)
    }
    out = _build_signal_rows(deduped, {"L1_42": 2.1}, rule_definitions, "X", {date(2026, 5, 16)})
    assert dict(out.schema) == _CONTRACT_SCHEMA


def test_build_rows_empty_when_no_emission_match() -> None:
    """If deduped has no rows matching emission_dates, return empty frame."""
    deduped = pl.DataFrame({
        "symbol": ["AAPL"], "date": [date(2026, 1, 1)], "rule_key": ["L1_42"],
    })
    rule_definitions = {"L1_42": Rule("L1_42", "step3c", "L1", [], 2.5, 0.45)}
    out = _build_signal_rows(deduped, {"L1_42": 2.1}, rule_definitions, "X", {date(2026, 5, 16)})
    assert out.height == 0
    assert dict(out.schema) == _CONTRACT_SCHEMA


# ---------- _next_sequence ----------

def test_next_sequence_empty_runs_root(tmp_path: Path) -> None:
    seq = _next_sequence(tmp_path / "missing", "2026-05-16")
    assert seq == 1


def test_next_sequence_no_matching_dirs(tmp_path: Path) -> None:
    (tmp_path / "2026-05-15-001").mkdir()
    (tmp_path / "2026-05-14-step4_walkforward_validation").mkdir()
    seq = _next_sequence(tmp_path, "2026-05-16")
    assert seq == 1


def test_next_sequence_increments_from_existing(tmp_path: Path) -> None:
    (tmp_path / "2026-05-16-001").mkdir()
    (tmp_path / "2026-05-16-002").mkdir()
    seq = _next_sequence(tmp_path, "2026-05-16")
    assert seq == 3


def test_next_sequence_ignores_step_suffix_dirs(tmp_path: Path) -> None:
    """Existing dirs with step suffix (e.g. 2026-05-16-001-step_foo/) should
    NOT count as the new contract-format pattern."""
    (tmp_path / "2026-05-16-001").mkdir()
    (tmp_path / "2026-05-16-001-step_quant_signal_emission").mkdir()
    (tmp_path / "2026-05-16-002").mkdir()
    seq = _next_sequence(tmp_path, "2026-05-16")
    assert seq == 3


# ---------- constants are wired to contract ----------

def test_lift_strength_divisor_matches_contract() -> None:
    """Contract specifies signal_strength = min(1.0, lift / 3.0)."""
    assert LIFT_STRENGTH_DIVISOR == 3.0


def test_default_dedup_matches_failure_mode_recovery() -> None:
    """Contract failure-mode says 'within a 30-day window'."""
    assert DEFAULT_DEDUP_WINDOW_DAYS == 30


def test_contract_schema_field_names_match_spec() -> None:
    expected = {
        "signal_id", "symbol", "signal_date", "signal_type",
        "signal_strength", "pattern", "expected_horizon_days",
        "expected_return_pct", "conditions_json",
    }
    assert set(_CONTRACT_SCHEMA.keys()) == expected
