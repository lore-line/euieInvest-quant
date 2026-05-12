"""Tests for quant.data.loader using a synthetic SQLite fixture.

Never touches the actual snapshot — builds a tiny database in tmp_path
with the real schema and points EUIEINVEST_SNAPSHOT at it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture()
def tiny_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "euieinvest.db"
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE price_history (
                symbol TEXT,
                date TEXT,
                close REAL,
                high REAL,
                low REAL,
                volume INTEGER
            );
            CREATE TABLE peer_groups (
                group_name TEXT,
                symbol TEXT
            );
            CREATE TABLE anomaly_flags (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                flag_date TEXT,
                fire_date TEXT,
                pivot_price REAL,
                vol_mult REAL,
                rsi REAL,
                sma20 REAL,
                sma50 REAL,
                peer_group TEXT,
                tier TEXT,
                status TEXT
            );
            INSERT INTO price_history VALUES
                ('AAA','2024-01-02',10.0,10.5,9.9,1000),
                ('AAA','2024-01-03',10.4,10.6,10.2,1100),
                ('BBB','2024-01-02',20.0,20.5,19.9,2000);
            INSERT INTO peer_groups VALUES
                ('tech','AAA'),
                ('tech','BBB'),
                ('energy','CCC');
            INSERT INTO anomaly_flags
                (symbol,flag_date,fire_date,pivot_price,vol_mult,rsi,sma20,sma50,peer_group,tier,status)
            VALUES
                ('AAA','2024-01-02','2024-01-03',10.0,2.5,55.0,9.5,9.0,'tech','3','open');
            """
        )
        con.commit()
    finally:
        con.close()
    monkeypatch.setenv("EUIEINVEST_SNAPSHOT", str(db_path))
    return db_path


def test_load_ohlcv_all(tiny_snapshot: Path) -> None:
    from quant.data.loader import load_ohlcv

    df = load_ohlcv()
    assert df.height == 3
    assert set(df.columns) == {"symbol", "date", "close", "high", "low", "volume"}
    assert df.schema["date"] == pl.Date


def test_load_ohlcv_filtered(tiny_snapshot: Path) -> None:
    from quant.data.loader import load_ohlcv

    df = load_ohlcv("AAA")
    assert df.height == 2
    assert df["symbol"].unique().to_list() == ["AAA"]


def test_load_peer_groups(tiny_snapshot: Path) -> None:
    from quant.data.loader import load_peer_groups

    groups = load_peer_groups()
    assert groups["tech"] == ["AAA", "BBB"]
    assert groups["energy"] == ["CCC"]


def test_load_anomaly_flags(tiny_snapshot: Path) -> None:
    from quant.data.loader import load_anomaly_flags

    flags = load_anomaly_flags()
    assert flags.height == 1
    assert flags["symbol"][0] == "AAA"
    assert "tier" in flags.columns


def test_read_only_mode(tiny_snapshot: Path) -> None:
    """Verify the snapshot is opened read-only — writes must error."""
    from quant.data.loader import _connect_ro

    con = _connect_ro()
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute(
                "INSERT INTO price_history VALUES ('XXX','2024-01-04',1.0,1.0,1.0,1)"
            )
    finally:
        con.close()


def test_missing_snapshot_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant.data.loader import load_ohlcv

    monkeypatch.setenv("EUIEINVEST_SNAPSHOT", str(tmp_path / "does-not-exist.db"))
    with pytest.raises(FileNotFoundError, match="pull-snapshot"):
        load_ohlcv()
