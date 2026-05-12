"""Read-only loader for the euieInvest snapshot.

Prefers parquet/JSON files written by ``scripts/pull-via-api.py`` to
``data/snapshots/`` (the API data plane). Falls back to a local SQLite
snapshot at ``data/snapshots/euieinvest.db`` during the API cutover
period, so older deployments keep working until the server side ships
its /api/v1 endpoints.

Configuration
-------------

``EUIEINVEST_SNAPSHOT_DIR``
    Directory containing the parquet/JSON cache. Defaults to
    ``<repo-root>/data/snapshots``. Tests override this to point at a
    ``tmp_path`` fixture.
``EUIEINVEST_SNAPSHOT``
    Path to the legacy SQLite file. Used only when the corresponding
    parquet/JSON file is absent. Defaults to
    ``<snapshot-dir>/euieinvest.db``.

The legacy SQLite path will be removed after the API is verified live
in production for ≥ 1 week — see plans/api-data-plane.md PR #6.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import polars as pl

__all__ = [
    "load_anomaly_flags",
    "load_ohlcv",
    "load_peer_groups",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SNAPSHOT_DIR = _REPO_ROOT / "data" / "snapshots"


def _snapshot_dir() -> Path:
    env = os.environ.get("EUIEINVEST_SNAPSHOT_DIR")
    return Path(env) if env else _DEFAULT_SNAPSHOT_DIR


def _legacy_sqlite_path() -> Path:
    env = os.environ.get("EUIEINVEST_SNAPSHOT")
    return Path(env) if env else _snapshot_dir() / "euieinvest.db"


def _connect_ro() -> sqlite3.Connection:
    """Open the legacy SQLite snapshot read-only.

    Public for tests; will be deleted with the SQLite fallback once
    the API path is verified in prod (plans/api-data-plane.md PR #6).
    """
    path = _legacy_sqlite_path()
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot not found at {path}. "
            "Run scripts/pull-via-api.py (preferred) or "
            "scripts/pull-snapshot.{sh,ps1} (legacy) to refresh the cache."
        )
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_ohlcv(symbol: str | None = None) -> pl.DataFrame:
    """Load OHLCV rows. Prefers ``data/snapshots/ohlcv.parquet``."""
    parquet = _snapshot_dir() / "ohlcv.parquet"
    if parquet.exists():
        df = pl.read_parquet(parquet)
        if symbol is not None:
            df = df.filter(pl.col("symbol") == symbol)
        return df
    return _load_ohlcv_from_sqlite(symbol)


def _load_ohlcv_from_sqlite(symbol: str | None) -> pl.DataFrame:
    con = _connect_ro()
    try:
        if symbol is None:
            cur = con.execute(
                "SELECT symbol, date, close, high, low, volume FROM price_history"
            )
        else:
            cur = con.execute(
                "SELECT symbol, date, close, high, low, volume FROM price_history "
                "WHERE symbol = ?",
                (symbol,),
            )
        rows = cur.fetchall()
    finally:
        con.close()
    df = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "date": pl.Utf8,
            "close": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "volume": pl.Int64,
        },
        orient="row",
    )
    return df.with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))


def load_peer_groups() -> dict[str, list[str]]:
    """Load peer groups. Prefers ``data/snapshots/peer_groups.json``."""
    json_path = _snapshot_dir() / "peer_groups.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return _load_peer_groups_from_sqlite()


def _load_peer_groups_from_sqlite() -> dict[str, list[str]]:
    con = _connect_ro()
    try:
        rows = con.execute(
            "SELECT group_name, symbol FROM peer_groups ORDER BY group_name, symbol"
        ).fetchall()
    finally:
        con.close()
    out: dict[str, list[str]] = {}
    for group, symbol in rows:
        out.setdefault(group, []).append(symbol)
    return out


def load_anomaly_flags() -> pl.DataFrame:
    """Load anomaly flags. Prefers ``data/snapshots/anomaly_flags.parquet``."""
    parquet = _snapshot_dir() / "anomaly_flags.parquet"
    if parquet.exists():
        return pl.read_parquet(parquet)
    return _load_anomaly_flags_from_sqlite()


def _load_anomaly_flags_from_sqlite() -> pl.DataFrame:
    con = _connect_ro()
    try:
        cur = con.execute("SELECT * FROM anomaly_flags")
        column_names = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()
    return pl.DataFrame(rows, schema=column_names, orient="row")
