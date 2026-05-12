"""Local smoke tests — import the app and exercise each route via TestClient.

Run with: cd quant_api && uv run pytest tests/ -v
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from euieinvest_api.app import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "euieInvest"
    assert body["api_version"] == "1"


def test_snapshot_cursor() -> None:
    r = client.get("/api/v1/snapshot-cursor")
    assert r.status_code == 200
    body = r.json()
    assert {"ohlcv", "anomaly_flags", "peer_groups"} <= set(body.keys())


def test_peer_groups() -> None:
    r = client.get("/api/v1/peer-groups")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)


def test_ohlcv_parquet() -> None:
    r = client.get("/api/v1/ohlcv?symbols=AAPL")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/vnd.apache.parquet"


def test_ohlcv_includes_open_and_close_adj() -> None:
    import polars as pl
    from io import BytesIO

    r = client.get("/api/v1/ohlcv?symbols=AAPL&since=2024-06-05&until=2024-06-10")
    df = pl.read_parquet(BytesIO(r.content))
    assert "open" in df.columns
    assert "close_adj" in df.columns
    assert df.schema["open"] == pl.Float64
    assert df.schema["close_adj"] == pl.Float64


def test_ohlcv_too_many_symbols() -> None:
    long_list = ",".join(f"X{i}" for i in range(5001))
    r = client.get(f"/api/v1/ohlcv?symbols={long_list}")
    assert r.status_code == 400
    assert "problem+json" in r.headers["content-type"]


def test_anomaly_flags() -> None:
    r = client.get("/api/v1/anomaly-flags")
    assert r.status_code == 200


def test_symbols_returns_per_ticker_metadata() -> None:
    """Per consumer's fixture in contract §5.6."""
    from datetime import date as _date

    r = client.get("/api/v1/symbols")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")

    body = r.json()
    assert isinstance(body, dict)
    assert len(body) > 0

    for sym, meta in body.items():
        assert isinstance(sym, str)
        assert set(meta.keys()) >= {
            "status", "last_seen", "shares_outstanding", "sector", "listing_date"
        }
        assert meta["status"] in {"active", "delisted"}
        assert _date.fromisoformat(meta["last_seen"])
        if meta["shares_outstanding"] is not None:
            assert isinstance(meta["shares_outstanding"], int)
            assert meta["shares_outstanding"] > 0
