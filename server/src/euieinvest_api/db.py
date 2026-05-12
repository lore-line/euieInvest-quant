"""Read-only SQLite connection helper.

The trading platform owns the DB and writes to it via better-sqlite3 in WAL
mode. We open with `mode=ro` so this process can never mutate the file, and
SQLite's WAL allows concurrent readers without blocking the writer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "euieinvest.db"


def db_path() -> Path:
    override = os.environ.get("EUIEINVEST_DB_PATH")
    return Path(override) if override else _DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(f"SQLite DB not found at {path}")
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> object:
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None
