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


def test_ohlcv_too_many_symbols() -> None:
    long_list = ",".join(f"X{i}" for i in range(5001))
    r = client.get(f"/api/v1/ohlcv?symbols={long_list}")
    assert r.status_code == 400
    assert "problem+json" in r.headers["content-type"]


def test_anomaly_flags() -> None:
    r = client.get("/api/v1/anomaly-flags")
    assert r.status_code == 200
