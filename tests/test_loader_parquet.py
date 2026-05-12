"""Tests for the parquet-preferred loader path.

Builds tiny parquet/json files in tmp_path and points the loader at
the directory via ``EUIEINVEST_SNAPSHOT_DIR``. No SQLite involvement
unless explicitly testing the fallback.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture()
def parquet_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pl.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "close": [10.0, 10.4, 20.0],
            "high": [10.5, 10.6, 20.5],
            "low": [9.9, 10.2, 19.9],
            "volume": [1000, 1100, 2000],
        }
    ).write_parquet(tmp_path / "ohlcv.parquet")

    (tmp_path / "peer_groups.json").write_text(
        json.dumps({"tech": ["AAA", "BBB"], "energy": ["CCC"]})
    )

    pl.DataFrame(
        {
            "id": [1],
            "symbol": ["AAA"],
            "flag_date": [date(2024, 1, 2)],
            "tier": ["3"],
            "status": ["open"],
        }
    ).write_parquet(tmp_path / "anomaly_flags.parquet")

    monkeypatch.setenv("EUIEINVEST_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.delenv("EUIEINVEST_SNAPSHOT", raising=False)
    return tmp_path


def test_load_ohlcv_from_parquet(parquet_snapshot: Path) -> None:
    from quant.data.loader import load_ohlcv

    df = load_ohlcv()
    assert df.height == 3
    assert df.schema["date"] == pl.Date
    assert set(df.columns) == {"symbol", "date", "close", "high", "low", "volume"}


def test_load_ohlcv_filter_from_parquet(parquet_snapshot: Path) -> None:
    from quant.data.loader import load_ohlcv

    df = load_ohlcv("AAA")
    assert df.height == 2
    assert df["symbol"].unique().to_list() == ["AAA"]


def test_load_peer_groups_from_json(parquet_snapshot: Path) -> None:
    from quant.data.loader import load_peer_groups

    g = load_peer_groups()
    assert g["tech"] == ["AAA", "BBB"]
    assert g["energy"] == ["CCC"]


def test_load_anomaly_flags_from_parquet(parquet_snapshot: Path) -> None:
    from quant.data.loader import load_anomaly_flags

    df = load_anomaly_flags()
    assert df.height == 1
    assert df["symbol"][0] == "AAA"
    assert df["tier"][0] == "3"


def test_parquet_preferred_over_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If both parquet and SQLite exist, parquet wins."""
    db = tmp_path / "euieinvest.db"
    con = sqlite3.connect(db)
    try:
        con.executescript(
            """
            CREATE TABLE price_history (
                symbol TEXT, date TEXT, close REAL,
                high REAL, low REAL, volume INTEGER
            );
            INSERT INTO price_history VALUES ('SQL', '2024-01-02', 1.0, 1.0, 1.0, 1);
            """
        )
        con.commit()
    finally:
        con.close()

    pl.DataFrame(
        {
            "symbol": ["PQ"],
            "date": [date(2024, 1, 2)],
            "close": [99.0],
            "high": [99.0],
            "low": [99.0],
            "volume": [99],
        }
    ).write_parquet(tmp_path / "ohlcv.parquet")

    monkeypatch.setenv("EUIEINVEST_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("EUIEINVEST_SNAPSHOT", str(db))

    from quant.data.loader import load_ohlcv

    df = load_ohlcv()
    assert df["symbol"][0] == "PQ", "parquet should win over SQLite when both exist"


def test_sqlite_fallback_when_parquet_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If parquet is missing and SQLite is present, SQLite is used."""
    db = tmp_path / "euieinvest.db"
    con = sqlite3.connect(db)
    try:
        con.executescript(
            """
            CREATE TABLE price_history (
                symbol TEXT, date TEXT, close REAL,
                high REAL, low REAL, volume INTEGER
            );
            INSERT INTO price_history VALUES ('LEGACY', '2024-01-02', 5.0, 5.0, 5.0, 5);
            """
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setenv("EUIEINVEST_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("EUIEINVEST_SNAPSHOT", str(db))

    from quant.data.loader import load_ohlcv

    df = load_ohlcv()
    assert df["symbol"][0] == "LEGACY"
