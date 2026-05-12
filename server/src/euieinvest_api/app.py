"""euieInvest data API — v1.

Five read-only endpoints under /api/v1/, exposing price_history, peer_groups,
and anomaly_flags to the euieInvest-quant consumer. See docs/api-contract.md
in the euieInvest-quant repo for the wire-format spec.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from io import BytesIO
from typing import Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from . import __version__
from .db import connect, scalar
from .peer_hash import peer_groups_hash

SERVICE_VERSION = "2026.05.12"
MAX_SYMBOLS = 5000
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
NDJSON_MEDIA_TYPE = "application/x-ndjson"
PROBLEM_MEDIA_TYPE = "application/problem+json"

app = FastAPI(
    title="euieInvest data API",
    version=__version__,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_conn: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


# -- /api/v1/health -----------------------------------------------------------


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "euieInvest",
        "service_version": SERVICE_VERSION,
        "api_version": "1",
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# -- /api/v1/snapshot-cursor --------------------------------------------------


def _load_peer_groups(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT group_name, symbol FROM peer_groups ORDER BY group_name, symbol"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["group_name"], []).append(r["symbol"])
    return out


@app.get("/api/v1/snapshot-cursor")
def snapshot_cursor() -> dict:
    conn = db()
    groups = _load_peer_groups(conn)
    return {
        "ohlcv": {
            "max_date": scalar(conn, "SELECT MAX(date) FROM price_history"),
            "row_count": int(scalar(conn, "SELECT COUNT(*) FROM price_history") or 0),
            "symbol_count": int(
                scalar(conn, "SELECT COUNT(DISTINCT symbol) FROM price_history") or 0
            ),
        },
        "anomaly_flags": {
            "max_id": int(scalar(conn, "SELECT COALESCE(MAX(id), 0) FROM anomaly_flags") or 0),
            "max_flag_date": scalar(conn, "SELECT MAX(flag_date) FROM anomaly_flags"),
            "row_count": int(scalar(conn, "SELECT COUNT(*) FROM anomaly_flags") or 0),
        },
        "peer_groups": {
            "hash": peer_groups_hash(groups),
            "group_count": len(groups),
            "entry_count": sum(len(v) for v in groups.values()),
        },
    }


# -- /api/v1/ohlcv ------------------------------------------------------------


def _tabular_response(df: pl.DataFrame, fmt: str) -> Response:
    if fmt == "parquet":
        buf = BytesIO()
        df.write_parquet(buf)
        return Response(buf.getvalue(), media_type=PARQUET_MEDIA_TYPE)
    if fmt == "ndjson":
        return Response(df.write_ndjson(), media_type=NDJSON_MEDIA_TYPE)
    raise HTTPException(
        status_code=400,
        detail=f"unsupported format '{fmt}'; supported: parquet, ndjson",
    )


_OHLCV_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "date": pl.Utf8,
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "volume": pl.Int64,
    "open": pl.Float64,
    "close_adj": pl.Float64,
}


@app.get("/api/v1/ohlcv")
def ohlcv(
    since: date | None = None,
    until: date | None = None,
    symbols: str | None = None,
    format: Literal["parquet", "ndjson"] = "parquet",
) -> Response:
    where: list[str] = []
    params: list[object] = []

    if since:
        where.append("date >= ?")
        params.append(since.isoformat())
    if until:
        where.append("date <= ?")
        params.append(until.isoformat())
    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if len(sym_list) > MAX_SYMBOLS:
            raise HTTPException(
                status_code=400,
                detail=f"symbols list exceeds max of {MAX_SYMBOLS}",
            )
        where.append(f"symbol IN ({','.join('?' * len(sym_list))})")
        params.extend(sym_list)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT symbol, date, close, high, low, volume, open, close_adj "
        f"FROM price_history{clause}"
    )
    rows = db().execute(sql, params).fetchall()

    df = pl.DataFrame(
        {
            "symbol": [r["symbol"] for r in rows],
            "date": [r["date"] for r in rows],
            "close": [r["close"] for r in rows],
            "high": [r["high"] for r in rows],
            "low": [r["low"] for r in rows],
            "volume": [r["volume"] for r in rows],
            "open": [r["open"] for r in rows],
            "close_adj": [r["close_adj"] for r in rows],
        },
        schema=_OHLCV_SCHEMA,
    ).with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))

    return _tabular_response(df, format)


# -- /api/v1/peer-groups ------------------------------------------------------


@app.get("/api/v1/peer-groups")
def peer_groups() -> dict[str, list[str]]:
    return _load_peer_groups(db())


# -- /api/v1/symbols ----------------------------------------------------------
# Contract §5.6. Per-symbol static-ish metadata. Backs the consumer's
# cap_bucket, market_regime, and survivorship-filter features.
#
# `last_seen` is computed live from price_history.MAX(date) so it never
# diverges from cursor.ohlcv.max_date for active names. `status` is derived:
# "active" if last_seen is within 7 calendar days of the universe max_date
# (covers 5 trading days + weekend), else "delisted".
#
# `shares_outstanding`, `sector`, `listing_date` come from `symbol_metadata`,
# populated by fetch-symbol-metadata.mjs. Nulls flow through honestly per
# spec (ETFs lack sector; pre-listed symbols lack listing_date; etc).

_DELISTED_THRESHOLD_DAYS = 7


@app.get("/api/v1/symbols")
def symbols() -> dict[str, dict[str, object]]:
    conn = db()
    rows = conn.execute(
        """
        SELECT ph.symbol,
               MAX(ph.date) AS last_seen,
               sm.shares_outstanding,
               sm.sector,
               sm.listing_date
        FROM price_history ph
        LEFT JOIN symbol_metadata sm ON sm.symbol = ph.symbol
        GROUP BY ph.symbol
        """
    ).fetchall()

    universe_max = conn.execute(
        "SELECT MAX(date) FROM price_history"
    ).fetchone()[0]
    universe_max_dt = (
        date.fromisoformat(universe_max) if universe_max else None
    )

    out: dict[str, dict[str, object]] = {}
    for r in rows:
        last_seen_str = r["last_seen"]
        last_seen_dt = date.fromisoformat(last_seen_str) if last_seen_str else None

        status = "delisted"
        if universe_max_dt and last_seen_dt:
            age = (universe_max_dt - last_seen_dt).days
            status = "active" if age <= _DELISTED_THRESHOLD_DAYS else "delisted"

        out[r["symbol"]] = {
            "status": status,
            "last_seen": last_seen_str,
            "shares_outstanding": r["shares_outstanding"],
            "sector": r["sector"],
            "listing_date": r["listing_date"],
        }
    return out


# -- /api/v1/anomaly-flags ----------------------------------------------------


_ANOMALY_TIMESTAMP_COLS = (
    "fired_at",
    "entry_at",
    "invalidated_at",
    "exited_at",
)
_ANOMALY_DATE_COLS = ("flag_date", "dismissed_until")


def _anomaly_schema(conn: sqlite3.Connection) -> dict[str, pl.DataType]:
    """Map every column in anomaly_flags to a polars dtype.

    Required columns (per contract §5.5) get their typed dtype. Future columns
    fall through as pl.Utf8 — additive evolution, contract §2.
    """
    info = conn.execute("PRAGMA table_info(anomaly_flags)").fetchall()
    typed = {
        "id": pl.Int64,
        "symbol": pl.Utf8,
        "flag_date": pl.Utf8,
        "fired_at": pl.Utf8,
        "pivot_price": pl.Float64,
        "vol_mult": pl.Float64,
        "rsi": pl.Float64,
        "sma20": pl.Float64,
        "sma50": pl.Float64,
        "peer_group": pl.Utf8,
        "tier": pl.Utf8,
        "status": pl.Utf8,
        "position_units": pl.Float64,
        "entry_price": pl.Float64,
        "entry_at": pl.Utf8,
        "peak_price": pl.Float64,
        "trailing_stop": pl.Float64,
        "invalidated_at": pl.Utf8,
        "exited_at": pl.Utf8,
        "exit_reason": pl.Utf8,
        "dismissed_until": pl.Utf8,
        "notes": pl.Utf8,
    }
    schema: dict[str, pl.DataType] = {}
    for col in info:
        name = col["name"]
        schema[name] = typed.get(name, pl.Utf8)
    return schema


@app.get("/api/v1/anomaly-flags")
def anomaly_flags(
    since: str | None = Query(default=None),
    status: str | None = Query(default=None),
    format: Literal["parquet", "ndjson"] = "parquet",
) -> Response:
    conn = db()
    schema = _anomaly_schema(conn)
    columns = list(schema.keys())

    where: list[str] = []
    params: list[object] = []

    if since:
        # Contract §5.5: since filter is a timestamp; we apply to flag_date.
        # Accept full ISO 8601 timestamp or plain YYYY-MM-DD; SQLite TEXT
        # comparison on ISO dates is lexicographic and correct.
        cutoff_date = since.split("T", 1)[0]
        try:
            date.fromisoformat(cutoff_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"invalid since= value '{since}' (expected ISO date or timestamp)",
            )
        where.append("flag_date >= ?")
        params.append(cutoff_date)

    if status:
        status_list = [s.strip() for s in status.split(",") if s.strip()]
        if status_list:
            where.append(f"status IN ({','.join('?' * len(status_list))})")
            params.extend(status_list)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT {','.join(columns)} FROM anomaly_flags{clause}"
    rows = conn.execute(sql, params).fetchall()

    data: dict[str, list] = {c: [r[c] for r in rows] for c in columns}
    df = pl.DataFrame(data, schema=schema)

    casts = []
    for col in _ANOMALY_DATE_COLS:
        if col in schema:
            casts.append(
                pl.col(col).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            )
    for col in _ANOMALY_TIMESTAMP_COLS:
        if col in schema:
            casts.append(
                pl.col(col).str.to_datetime(time_zone="UTC", strict=False, time_unit="us")
            )
    if casts:
        df = df.with_columns(casts)

    return _tabular_response(df, format)


# -- Error handler (RFC 7807) -------------------------------------------------


def _problem_response(status_code: int, title: str, detail: object, path: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": f"https://euieinvest.invalid/errors/{status_code}",
            "title": title,
            "status": status_code,
            "detail": detail if isinstance(detail, (str, list, dict)) else str(detail),
            "instance": path,
        },
        media_type=PROBLEM_MEDIA_TYPE,
    )


@app.exception_handler(HTTPException)
async def problem_json_handler(request: Request, exc: HTTPException) -> JSONResponse:
    title = str(exc.detail)[:60] if exc.detail else "request failed"
    return _problem_response(exc.status_code, title, exc.detail, request.url.path)


@app.exception_handler(RequestValidationError)
async def validation_problem_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _problem_response(
        400, "invalid request parameters", exc.errors(), request.url.path
    )
