"""Contract tests against a live euieInvest data API server.

Skipped unless ``EUIEINVEST_API_BASE_URL`` is set. To run:

    EUIEINVEST_API_BASE_URL=http://localhost:8443 uv run pytest tests/contract/

These tests intentionally hit the network and require a running server
implementing the contract in docs/api-contract.md. They are NOT part of
the default test run.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from io import BytesIO

import httpx
import polars as pl
import pytest


_BASE_URL_ENV = "EUIEINVEST_API_BASE_URL"


def _api_base() -> str:
    base = os.environ[_BASE_URL_ENV].rstrip("/")
    return base + "/api/v1"


pytestmark = pytest.mark.skipif(
    not os.environ.get(_BASE_URL_ENV),
    reason=f"{_BASE_URL_ENV} not set — these tests require a live server",
)


# -- /api/v1/health -----------------------------------------------------------


def test_health_returns_200_with_schema() -> None:
    r = httpx.get(f"{_api_base()}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert body["service"] == "euieInvest"
    assert body["api_version"] == "1"
    assert "service_version" in body
    assert "as_of" in body


# -- /api/v1/snapshot-cursor --------------------------------------------------


def test_snapshot_cursor_has_required_keys() -> None:
    r = httpx.get(f"{_api_base()}/snapshot-cursor", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert {"ohlcv", "anomaly_flags", "peer_groups"} <= set(body.keys())

    assert {"max_date", "row_count", "symbol_count"} <= set(body["ohlcv"].keys())
    assert isinstance(body["ohlcv"]["row_count"], int)
    assert isinstance(body["ohlcv"]["symbol_count"], int)

    assert {"max_id", "row_count"} <= set(body["anomaly_flags"].keys())
    assert isinstance(body["anomaly_flags"]["max_id"], int)

    assert {"hash", "group_count", "entry_count"} <= set(body["peer_groups"].keys())
    assert body["peer_groups"]["hash"].startswith("sha256:")
    assert len(body["peer_groups"]["hash"]) == len("sha256:") + 64


# -- /api/v1/ohlcv ------------------------------------------------------------


def test_ohlcv_returns_parquet_with_correct_schema() -> None:
    r = httpx.get(f"{_api_base()}/ohlcv", params={"format": "parquet"}, timeout=120)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/vnd.apache.parquet"

    df = pl.read_parquet(BytesIO(r.content))

    # Required columns — contract §5.3. Subset semantics (not exact-match)
    # so the server can add columns additively per §2 versioning policy.
    required_cols = {"symbol", "date", "close", "high", "low", "volume"}
    assert required_cols.issubset(set(df.columns)), (
        f"missing required ohlcv columns: {required_cols - set(df.columns)}"
    )

    # Required dtypes
    assert df.schema["symbol"] == pl.Utf8
    assert df.schema["date"] == pl.Date
    assert df.schema["close"] == pl.Float64
    assert df.schema["high"] == pl.Float64
    assert df.schema["low"] == pl.Float64
    assert df.schema["volume"] == pl.Int64

    # Forward-looking dtype checks for additive columns we expect to land:
    # - `open`: split-adjusted open price (consistent basis with close/high/low)
    # - `close_adj`: split + dividend-adjusted close (Yahoo's adjclose)
    # When the server ships these as part of the contract §5.3 response, the
    # types are pinned here so a future-typed regression fails loudly.
    if "open" in df.columns:
        assert df.schema["open"] == pl.Float64, (
            "open must be Float64 (split-adjusted, consistent with close)"
        )
    if "close_adj" in df.columns:
        assert df.schema["close_adj"] == pl.Float64, (
            "close_adj must be Float64 (split + dividend-adjusted)"
        )

    assert df.height > 0, "live server should have non-empty ohlcv data"


def test_ohlcv_filters_by_since() -> None:
    cutoff = date(2025, 1, 1)
    r = httpx.get(
        f"{_api_base()}/ohlcv",
        params={"since": cutoff.isoformat()},
        timeout=120,
    )
    assert r.status_code == 200
    df = pl.read_parquet(BytesIO(r.content))
    if df.height > 0:
        assert (df["date"] >= cutoff).all(), (
            "ohlcv with since=2025-01-01 returned older rows"
        )


def test_ohlcv_filters_by_symbols() -> None:
    # Pick a symbol that should exist in any prod snapshot
    r = httpx.get(
        f"{_api_base()}/ohlcv",
        params={"symbols": "AAPL"},
        timeout=60,
    )
    assert r.status_code == 200
    df = pl.read_parquet(BytesIO(r.content))
    if df.height > 0:
        assert df["symbol"].unique().to_list() == ["AAPL"]


def test_ohlcv_rejects_oversized_symbols_list() -> None:
    long_list = ",".join(f"X{i}" for i in range(5001))
    r = httpx.get(
        f"{_api_base()}/ohlcv", params={"symbols": long_list}, timeout=10
    )
    assert r.status_code == 400
    assert "problem+json" in r.headers.get("content-type", "")


# -- /api/v1/peer-groups ------------------------------------------------------


def test_peer_groups_returns_mapping() -> None:
    r = httpx.get(f"{_api_base()}/peer-groups", timeout=10)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")

    body = r.json()
    assert isinstance(body, dict)
    for group, symbols in body.items():
        assert isinstance(group, str)
        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)


def test_peer_groups_hash_in_cursor_matches_canonicalization() -> None:
    """The hash in /snapshot-cursor must be reproducible from /peer-groups."""
    cursor = httpx.get(f"{_api_base()}/snapshot-cursor", timeout=10).json()
    peer_groups = httpx.get(f"{_api_base()}/peer-groups", timeout=10).json()

    # Canonicalize per contract §5.2: sorted keys + sorted inner arrays
    canonical = {k: sorted(v) for k, v in peer_groups.items()}
    body = json.dumps(canonical, sort_keys=True)
    expected_hash = "sha256:" + hashlib.sha256(body.encode()).hexdigest()

    assert cursor["peer_groups"]["hash"] == expected_hash, (
        "cursor's peer_groups.hash does not match canonicalized /peer-groups"
    )


# -- /api/v1/anomaly-flags ----------------------------------------------------


def test_anomaly_flags_returns_parquet_with_required_columns() -> None:
    r = httpx.get(
        f"{_api_base()}/anomaly-flags", params={"format": "parquet"}, timeout=60
    )
    assert r.status_code == 200
    df = pl.read_parquet(BytesIO(r.content))
    required_cols = {
        "id",
        "symbol",
        "flag_date",
        "fired_at",
        "pivot_price",
        "vol_mult",
        "rsi",
        "sma20",
        "sma50",
        "peer_group",
        "tier",
        "status",
    }
    assert required_cols.issubset(set(df.columns)), (
        f"anomaly-flags missing required columns: "
        f"{required_cols - set(df.columns)}"
    )


# -- /api/v1/symbols ----------------------------------------------------------


def test_symbols_returns_per_ticker_metadata() -> None:
    """Per contract §5.6."""
    r = httpx.get(f"{_api_base()}/symbols", timeout=30)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert isinstance(body, dict)
    assert len(body) > 0
    required_keys = {
        "status", "last_seen", "shares_outstanding", "sector", "listing_date"
    }
    for sym, meta in body.items():
        assert isinstance(sym, str)
        assert required_keys <= set(meta.keys()), (
            f"{sym} missing keys: {required_keys - set(meta.keys())}"
        )
        assert meta["status"] in {"active", "delisted"}
        assert date.fromisoformat(meta["last_seen"])
        if meta["shares_outstanding"] is not None:
            assert isinstance(meta["shares_outstanding"], int)
            assert meta["shares_outstanding"] > 0


# -- Cross-endpoint invariants ------------------------------------------------


def test_cursor_ohlcv_max_date_matches_ohlcv_max() -> None:
    """cursor.ohlcv.max_date must equal max(ohlcv.date)."""
    cursor = httpx.get(f"{_api_base()}/snapshot-cursor", timeout=10).json()
    r = httpx.get(f"{_api_base()}/ohlcv", timeout=180)
    df = pl.read_parquet(BytesIO(r.content))

    actual_max = df["date"].max()
    reported_max = date.fromisoformat(cursor["ohlcv"]["max_date"])
    assert actual_max == reported_max, (
        f"cursor.ohlcv.max_date ({reported_max}) != "
        f"max(ohlcv.date) ({actual_max})"
    )


def test_cursor_ohlcv_row_count_matches_ohlcv_height() -> None:
    """cursor.ohlcv.row_count must equal len(ohlcv) when no filters applied."""
    cursor = httpx.get(f"{_api_base()}/snapshot-cursor", timeout=10).json()
    r = httpx.get(f"{_api_base()}/ohlcv", timeout=180)
    df = pl.read_parquet(BytesIO(r.content))

    assert df.height == cursor["ohlcv"]["row_count"], (
        f"cursor.ohlcv.row_count ({cursor['ohlcv']['row_count']}) != "
        f"len(/ohlcv) ({df.height})"
    )
