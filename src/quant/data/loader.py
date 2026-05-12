"""Read-only loader for the euieInvest SQLite snapshot.

The snapshot file lives at ``data/snapshots/euieinvest.db`` relative to the
repo root. Override the location via the ``EUIEINVEST_SNAPSHOT`` env var
(used by the test suite to point at synthetic SQLite fixtures).
"""
from __future__ import annotations

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
_DEFAULT_SNAPSHOT = _REPO_ROOT / "data" / "snapshots" / "euieinvest.db"


def _resolve_snapshot_path() -> Path:
    env = os.environ.get("EUIEINVEST_SNAPSHOT")
    return Path(env) if env else _DEFAULT_SNAPSHOT


def _connect_ro() -> sqlite3.Connection:
    path = _resolve_snapshot_path()
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot not found at {path}. "
            "Run scripts/pull-snapshot.sh (WSL/Linux) or "
            "scripts/pull-snapshot.ps1 (PowerShell) to rsync it from claudehost."
        )
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_ohlcv(symbol: str | None = None) -> pl.DataFrame:
    """Load OHLCV rows from ``price_history``.

    Parameters
    ----------
    symbol:
        Optional ticker filter. If ``None``, returns the full table.

    Returns
    -------
    pl.DataFrame
        Columns: ``symbol``, ``date`` (pl.Date), ``close``, ``high``, ``low``,
        ``volume``.
    """
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
    """Return ``{group_name: [symbol, ...]}`` from the ``peer_groups`` table."""
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
    """Load the full ``anomaly_flags`` table for baseline comparison."""
    con = _connect_ro()
    try:
        cur = con.execute("SELECT * FROM anomaly_flags")
        column_names = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()
    return pl.DataFrame(rows, schema=column_names, orient="row")
