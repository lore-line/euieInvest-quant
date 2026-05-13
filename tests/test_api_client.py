"""Tests for quant.data.api_client using respx for HTTP mocking.

Coverage:
- happy paths for each endpoint
- query parameter serialization
- parquet round-trip via in-memory bytes
- error mapping (4xx/5xx -> ApiError)
- env var resolution
- input validation (symbols list cap)
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from io import BytesIO

import httpx
import polars as pl
import pytest
import respx

from quant.data import api_client
from quant.data.api_client import ApiError


@pytest.fixture(autouse=True)
def _set_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EUIEINVEST_API_BASE_URL", "http://test.invalid:8080")


def _parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


@respx.mock
def test_fetch_health_happy_path() -> None:
    respx.get("http://test.invalid:8080/api/v1/health").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ok",
                "service": "euieInvest",
                "service_version": "2026.05.12",
                "api_version": "1",
                "as_of": "2026-05-12T01:23:45Z",
            },
        )
    )
    result = api_client.fetch_health()
    assert result["status"] == "ok"
    assert result["api_version"] == "1"


@respx.mock
def test_fetch_snapshot_cursor() -> None:
    respx.get("http://test.invalid:8080/api/v1/snapshot-cursor").mock(
        return_value=httpx.Response(
            200,
            json={
                "ohlcv": {
                    "max_date": "2026-05-11",
                    "row_count": 2_431_188,
                    "symbol_count": 2_046,
                },
                "anomaly_flags": {
                    "max_id": 1,
                    "max_flag_date": "2024-01-02",
                    "row_count": 1,
                },
                "peer_groups": {
                    "hash": "sha256:abc",
                    "group_count": 6,
                    "entry_count": 33,
                },
            },
        )
    )
    cursor = api_client.fetch_snapshot_cursor()
    assert cursor["ohlcv"]["row_count"] == 2_431_188
    assert cursor["peer_groups"]["group_count"] == 6


@respx.mock
def test_fetch_ohlcv_returns_parquet_dataframe() -> None:
    sent = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "date": [date(2024, 1, 2), date(2024, 1, 2)],
            "close": [10.0, 20.0],
            "high": [10.5, 20.5],
            "low": [9.9, 19.9],
            "volume": [1000, 2000],
        }
    )
    respx.get("http://test.invalid:8080/api/v1/ohlcv").mock(
        return_value=httpx.Response(
            200,
            content=_parquet_bytes(sent),
            headers={"content-type": "application/vnd.apache.parquet"},
        )
    )
    df = api_client.fetch_ohlcv()
    assert df.height == 2
    assert df.columns == ["symbol", "date", "close", "high", "low", "volume"]
    assert df["close"].to_list() == [10.0, 20.0]
    assert df.schema["date"] == pl.Date


@respx.mock
def test_fetch_ohlcv_with_filters_serializes_query_params() -> None:
    sent = pl.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [date(2024, 1, 2)],
            "close": [10.0],
            "high": [10.5],
            "low": [9.9],
            "volume": [1000],
        }
    )
    route = respx.get("http://test.invalid:8080/api/v1/ohlcv").mock(
        return_value=httpx.Response(
            200,
            content=_parquet_bytes(sent),
            headers={"content-type": "application/vnd.apache.parquet"},
        )
    )
    api_client.fetch_ohlcv(
        since=date(2024, 1, 1),
        until=date(2024, 12, 31),
        symbols=["AAA", "BBB"],
    )
    req_params = route.calls.last.request.url.params
    assert req_params["since"] == "2024-01-01"
    assert req_params["until"] == "2024-12-31"
    assert req_params["symbols"] == "AAA,BBB"
    assert req_params["format"] == "parquet"


def test_fetch_ohlcv_rejects_too_many_symbols() -> None:
    with pytest.raises(ValueError, match="exceeds API max"):
        api_client.fetch_ohlcv(symbols=["X"] * 5001)


@respx.mock
def test_fetch_peer_groups() -> None:
    respx.get("http://test.invalid:8080/api/v1/peer-groups").mock(
        return_value=httpx.Response(
            200, json={"tech": ["AAA", "BBB"], "energy": ["CCC"]}
        )
    )
    groups = api_client.fetch_peer_groups()
    assert groups == {"tech": ["AAA", "BBB"], "energy": ["CCC"]}


@respx.mock
def test_fetch_symbols() -> None:
    payload = {
        "AAPL": {
            "status": "active",
            "last_seen": "2026-05-12",
            "shares_outstanding": 15204000000,
            "sector": "Technology",
            "listing_date": "1980-12-12",
        },
        "SPY": {
            "status": "active",
            "last_seen": "2026-05-12",
            "shares_outstanding": None,
            "sector": None,
            "listing_date": "1993-01-29",
        },
    }
    respx.get("http://test.invalid:8080/api/v1/symbols").mock(
        return_value=httpx.Response(200, json=payload)
    )
    symbols = api_client.fetch_symbols()
    assert symbols == payload
    # The Track 11 contract: per-symbol .get("sector") is what we need.
    assert symbols["AAPL"]["sector"] == "Technology"
    assert symbols["SPY"]["sector"] is None


@respx.mock
def test_fetch_anomaly_flags_serializes_status_filter() -> None:
    sent = pl.DataFrame(
        {
            "id": [1],
            "symbol": ["AAA"],
            "flag_date": [date(2024, 1, 2)],
            "status": ["open"],
        }
    )
    route = respx.get("http://test.invalid:8080/api/v1/anomaly-flags").mock(
        return_value=httpx.Response(
            200,
            content=_parquet_bytes(sent),
            headers={"content-type": "application/vnd.apache.parquet"},
        )
    )
    api_client.fetch_anomaly_flags(
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=["open", "exited"],
    )
    req_params = route.calls.last.request.url.params
    assert req_params["status"] == "open,exited"
    assert req_params["since"].startswith("2024-01-01")


@respx.mock
def test_api_error_on_5xx() -> None:
    respx.get("http://test.invalid:8080/api/v1/health").mock(
        return_value=httpx.Response(503, text="db down")
    )
    with pytest.raises(ApiError) as exc:
        api_client.fetch_health()
    assert exc.value.status == 503
    assert "db down" in exc.value.body


@respx.mock
def test_api_error_on_4xx() -> None:
    respx.get("http://test.invalid:8080/api/v1/ohlcv").mock(
        return_value=httpx.Response(
            400,
            json={"type": "/errors/bad-date", "title": "Invalid date", "status": 400},
        )
    )
    with pytest.raises(ApiError) as exc:
        api_client.fetch_ohlcv()
    assert exc.value.status == 400


def test_missing_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EUIEINVEST_API_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="EUIEINVEST_API_BASE_URL"):
        api_client.fetch_health()


def test_base_url_trailing_slash_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing slashes are stripped so /api/v1/health doesn't become //api/v1/..."""
    monkeypatch.setenv("EUIEINVEST_API_BASE_URL", "http://test.invalid:8080/")
    with respx.mock:
        route = respx.get("http://test.invalid:8080/api/v1/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        api_client.fetch_health()
        assert route.called


# -- with_retry helper --------------------------------------------------------
#
# These tests exercise the retry wrapper used by scripts/pull-via-api.py.
# Connection-class errors should retry once; HTTP 4xx/5xx (ApiError) and
# other exceptions should NOT retry.


def test_with_retry_succeeds_on_first_attempt() -> None:
    """No failure → no retry, no message."""
    calls = []

    def fetcher() -> str:
        calls.append(1)
        return "ok"

    result = api_client.with_retry(fetcher, name="t", delay_s=0.0)
    assert result == "ok"
    assert len(calls) == 1


def test_with_retry_recovers_from_one_connect_error() -> None:
    """Single transient failure → retry succeeds → returns result."""
    calls = []

    def fetcher() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return "ok"

    result = api_client.with_retry(fetcher, name="t", delay_s=0.0)
    assert result == "ok"
    assert len(calls) == 2


def test_with_retry_raises_after_repeated_failures() -> None:
    """Two consecutive failures → bubble up the second exception."""
    calls = []

    def fetcher() -> str:
        calls.append(1)
        raise httpx.ConnectError(f"boom {len(calls)}")

    with pytest.raises(httpx.ConnectError, match="boom 2"):
        api_client.with_retry(fetcher, name="t", delay_s=0.0)
    assert len(calls) == 2


def test_with_retry_does_not_retry_on_api_error() -> None:
    """HTTP 4xx/5xx wrapped as ApiError is NOT a transient — no retry."""
    calls = []

    def fetcher() -> str:
        calls.append(1)
        raise ApiError(503, "db down", "http://test.invalid/api/v1/health")

    with pytest.raises(ApiError):
        api_client.with_retry(fetcher, name="t", delay_s=0.0)
    assert len(calls) == 1, "ApiError must NOT trigger retry"


def test_with_retry_does_not_retry_on_unexpected_exception() -> None:
    """Unknown exception types bubble up immediately; we only retry
    the narrow transient set."""
    calls = []

    def fetcher() -> str:
        calls.append(1)
        raise ValueError("programming error")

    with pytest.raises(ValueError):
        api_client.with_retry(fetcher, name="t", delay_s=0.0)
    assert len(calls) == 1


def test_with_retry_retries_zero_when_disabled() -> None:
    """retries=0 → single attempt, no recovery."""
    calls = []

    def fetcher() -> str:
        calls.append(1)
        raise httpx.ReadError("interrupted")

    with pytest.raises(httpx.ReadError):
        api_client.with_retry(fetcher, name="t", retries=0, delay_s=0.0)
    assert len(calls) == 1


def test_with_retry_handles_read_error_and_protocol_error() -> None:
    """The transient set includes ReadError and RemoteProtocolError, not
    just ConnectError — same retry behavior."""
    for exc_cls in (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout):
        calls = []

        def fetcher() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise exc_cls(f"transient {exc_cls.__name__}")
            return "ok"

        result = api_client.with_retry(fetcher, name="t", delay_s=0.0)
        assert result == "ok", f"{exc_cls.__name__} should be retried"
        assert len(calls) == 2


def test_api_client_class_smoke() -> None:
    """ApiClient() exposes the same surface as the module-level fetchers."""
    c = api_client.ApiClient()
    assert callable(c.health)
    assert callable(c.snapshot_cursor)
    assert callable(c.ohlcv)
    assert callable(c.peer_groups)
    assert callable(c.anomaly_flags)
