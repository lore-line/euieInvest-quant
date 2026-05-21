"""Build a local SQLite intraday DB on heaven-pc from monthly parquet snapshots.

Mirrors server-side /home/euie/code/euieInvest/data/euieinvest.db `intraday_history`
schema so the harness can run via `--db-path` (when server team ships --data-source)
or by stubbing the hardcoded DB_PATH.

Inputs:
  data/intraday_archive/intraday-YYYY-MM.parquet  (rsync'd from claudehost)

Output:
  data/intraday_archive/euieinvest.db
    table: intraday_history(symbol TEXT, timestamp TEXT, interval_min INTEGER,
                            open REAL, high REAL, low REAL, close REAL, volume REAL)
    index: (symbol, interval_min, timestamp)

Idempotent: drops + recreates table on every run.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
ARCHIVE_DIR = REPO / "data" / "intraday_archive"
DB_PATH = ARCHIVE_DIR / "euieinvest.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS intraday_history (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    interval_min INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, interval_min, timestamp)
);
"""
INDEX = "CREATE INDEX IF NOT EXISTS idx_intraday_lookup ON intraday_history (symbol, interval_min, timestamp);"


def main() -> int:
    parquets = sorted(ARCHIVE_DIR.glob("intraday-*.parquet"))
    if not parquets:
        print(f"ERROR: no parquets in {ARCHIVE_DIR}", file=sys.stderr)
        return 1
    print(f"[build-db] found {len(parquets)} monthly parquets in {ARCHIVE_DIR}",
          file=sys.stderr)
    total_mb = sum(p.stat().st_size for p in parquets) / 1024 / 1024
    print(f"[build-db] total parquet size: {total_mb:.1f} MB", file=sys.stderr)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[build-db] removed existing {DB_PATH}", file=sys.stderr)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(SCHEMA)
    conn.commit()

    total_rows = 0
    for i, p in enumerate(parquets, 1):
        df = pd.read_parquet(p)
        # Coerce timestamp to ISO-string text matching the original DB schema.
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        # Reorder cols to match schema (defensive).
        cols = ["symbol", "timestamp", "interval_min", "open", "high", "low", "close", "volume"]
        df = df[cols]
        df.to_sql("intraday_history", conn, if_exists="append", index=False,
                  method="multi", chunksize=10000)
        total_rows += len(df)
        print(f"[build-db] {i}/{len(parquets)} {p.name}: +{len(df):,} rows "
              f"(total {total_rows:,})", file=sys.stderr)

    conn.execute(INDEX)
    conn.commit()
    print(f"[build-db] created index on (symbol, interval_min, timestamp)",
          file=sys.stderr)

    # Verify.
    n = conn.execute("SELECT COUNT(*) FROM intraday_history").fetchone()[0]
    symbols = conn.execute("SELECT DISTINCT symbol FROM intraday_history ORDER BY symbol").fetchall()
    intervals = conn.execute("SELECT DISTINCT interval_min FROM intraday_history ORDER BY interval_min").fetchall()
    date_range = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM intraday_history"
    ).fetchone()
    db_size_mb = DB_PATH.stat().st_size / 1024 / 1024

    conn.close()

    print(f"\n[ok] wrote {n:,} rows -> {DB_PATH.relative_to(REPO)}  ({db_size_mb:.1f} MB)")
    print(f"     symbols: {[s[0] for s in symbols]}")
    print(f"     intervals: {[i[0] for i in intervals]}")
    print(f"     date range: {date_range[0]} -> {date_range[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
