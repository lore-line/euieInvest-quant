"""Typed HTTP client for the euieInvest data API (v1).

See `docs/api-contract.md` for the canonical spec. This module is the
only place in the repo that speaks HTTP; the loader reads parquet/JSON
off disk and never reaches the network. The pull script
(`scripts/pull-via-api.py`) uses this client to refresh the local cache.

Configuration
-------------

``EUIEINVEST_API_BASE_URL``
    Base URL of the API, e.g. ``http://100.68.86.56:8443``. **No trailing
    slash, no /api/v1 suffix** — the client appends ``/api/v1`` to every
    request. Required.

Typical usage
-------------

>>> import os
>>> os.environ["EUIEINVEST_API_BASE_URL"] = "http://100.68.86.56:8443"
>>> from quant.data import api_client
>>> api_client.fetch_health()
{'status': 'ok', 'service': 'euieInvest', ...}
>>> df = api_client.fetch_ohlcv(since=date(2024, 1, 1), symbols=["AAPL"])
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime
from io import BytesIO
from typing import Any, Callable, TypeVar

import httpx
import polars as pl

__all__ = [
    "ApiClient",
    "ApiError",
    "fetch_anomaly_flags",
    "fetch_health",
    "fetch_intraday",
    "fetch_ohlcv",
    "fetch_peer_groups",
    "fetch_snapshot_cursor",
    "fetch_symbols",
    "with_retry",
]

_DEFAULT_TIMEOUT_S = 120.0
_MAX_SYMBOLS_PER_REQUEST = 5_000

# Default retry policy used by `with_retry`. The motivating reason is
# claudehost's API runs as a user-level systemd unit; without
# `loginctl enable-linger euie` it can drop briefly after a host reboot.
# One retry covers the typical post-reboot gap. Repeated failures should
# raise — they're not transients we want to silently mask.
_DEFAULT_RETRIES = 1
_DEFAULT_RETRY_DELAY_S = 5.0

_T = TypeVar("_T")

# Connection-class errors we consider transient enough to retry once.
# 4xx/5xx HTTP responses (wrapped as ApiError) are NOT retried — those
# are server-side decisions and retrying changes nothing.
_TRANSIENT_HTTP_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
)


class ApiError(RuntimeError):
    """Raised when the API returns a non-2xx response.

    Carries the HTTP status, the response body (truncated for the
    message but available in full on ``body``), and the URL hit.
    """

    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:200]}")
        self.status = status
        self.body = body
        self.url = url


def _base_url() -> str:
    base = os.environ.get("EUIEINVEST_API_BASE_URL")
    if not base:
        raise RuntimeError(
            "EUIEINVEST_API_BASE_URL is not set. Set it to the URL of the "
            "euieInvest data API, e.g. http://100.68.86.56:8443"
        )
    return base.rstrip("/") + "/api/v1"


def _get(
    path: str,
    *,
    params: dict[str, str] | None = None,
    accept: str = "application/json",
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> httpx.Response:
    url = f"{_base_url()}{path}"
    headers = {"Accept": accept}
    r = httpx.get(url, params=params, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise ApiError(r.status_code, r.text, str(r.url))
    return r


def fetch_health() -> dict[str, Any]:
    """``GET /api/v1/health`` — liveness probe.

    Returns the parsed JSON body. See `docs/api-contract.md` §5.1.
    """
    return _get("/health").json()


def fetch_snapshot_cursor() -> dict[str, Any]:
    """``GET /api/v1/snapshot-cursor`` — table high-water marks.

    Clients call this before deciding whether to refetch. See
    `docs/api-contract.md` §5.2.
    """
    return _get("/snapshot-cursor").json()


def fetch_ohlcv(
    *,
    since: date | None = None,
    until: date | None = None,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """``GET /api/v1/ohlcv`` as parquet → polars DataFrame.

    Parameters
    ----------
    since:
        If provided, server returns rows with ``date >= since``.
    until:
        If provided, server returns rows with ``date <= until``.
    symbols:
        If provided, server returns only listed tickers. Max
        ``_MAX_SYMBOLS_PER_REQUEST``; raises ``ValueError`` if exceeded.

    See `docs/api-contract.md` §5.3.
    """
    params: dict[str, str] = {"format": "parquet"}
    if since is not None:
        params["since"] = since.isoformat()
    if until is not None:
        params["until"] = until.isoformat()
    if symbols:
        if len(symbols) > _MAX_SYMBOLS_PER_REQUEST:
            raise ValueError(
                f"symbols list exceeds API max of {_MAX_SYMBOLS_PER_REQUEST} "
                f"(got {len(symbols)})"
            )
        params["symbols"] = ",".join(symbols)
    r = _get("/ohlcv", params=params, accept="application/vnd.apache.parquet")
    return pl.read_parquet(BytesIO(r.content))


def fetch_intraday(
    *,
    interval_min: int,
    since: date | None = None,
    until: date | None = None,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """``GET /api/v1/intraday`` as parquet → polars DataFrame.

    Returns rows with columns ``symbol | timestamp | interval_min |
    open | high | low | close | volume``. Timestamp is ISO 8601 with
    'T' separator and milliseconds (e.g. ``2022-09-15T00:00:00.000Z``).

    Parameters
    ----------
    interval_min:
        Bar size in minutes. Required by the server (5, 15, 60, etc.).
    since, until:
        Date-bounded filter applied to the timestamp column. ``until``
        is inclusive (server uses next-day-boundary semantics).
    symbols:
        Optional ticker filter. Same MAX as /ohlcv.

    Consumed by ``quant.backtest.dca_grid`` via the puller
    ``scripts/pull_intraday.py``.
    """
    params: dict[str, str] = {"format": "parquet", "interval_min": str(interval_min)}
    if since is not None:
        params["since"] = since.isoformat()
    if until is not None:
        params["until"] = until.isoformat()
    if symbols:
        if len(symbols) > _MAX_SYMBOLS_PER_REQUEST:
            raise ValueError(
                f"symbols list exceeds API max of {_MAX_SYMBOLS_PER_REQUEST} "
                f"(got {len(symbols)})"
            )
        params["symbols"] = ",".join(symbols)
    r = _get("/intraday", params=params, accept="application/vnd.apache.parquet")
    return pl.read_parquet(BytesIO(r.content))


def fetch_peer_groups() -> dict[str, list[str]]:
    """``GET /api/v1/peer-groups`` → ``{group_name: [symbol, ...]}``.

    See `docs/api-contract.md` §5.4.
    """
    return _get("/peer-groups").json()


def fetch_symbols() -> dict[str, dict[str, Any]]:
    """``GET /api/v1/symbols`` → per-symbol static-ish metadata.

    Returns a dict keyed by ticker; each value is a dict with
    ``status`` / ``last_seen`` / ``shares_outstanding`` / ``sector`` /
    ``listing_date``. Used by Track 11 (multi-task fine-tune) for the
    sector-relative-rank label.

    See `docs/api-contract.md` §5.6.
    """
    return _get("/symbols").json()


def fetch_anomaly_flags(
    *,
    since: datetime | None = None,
    status: list[str] | None = None,
) -> pl.DataFrame:
    """``GET /api/v1/anomaly-flags`` as parquet → polars DataFrame.

    Parameters
    ----------
    since:
        If provided, server returns rows with ``flag_date >= since``.
    status:
        If provided, filter to listed status values
        (e.g. ``["open", "exited"]``).

    See `docs/api-contract.md` §5.5.
    """
    params: dict[str, str] = {"format": "parquet"}
    if since is not None:
        params["since"] = since.isoformat()
    if status:
        params["status"] = ",".join(status)
    r = _get(
        "/anomaly-flags", params=params, accept="application/vnd.apache.parquet"
    )
    return pl.read_parquet(BytesIO(r.content))


def with_retry(
    fn: Callable[[], _T],
    *,
    name: str = "fetch",
    retries: int = _DEFAULT_RETRIES,
    delay_s: float = _DEFAULT_RETRY_DELAY_S,
) -> _T:
    """Call ``fn()`` and retry on transient connection failures.

    Parameters
    ----------
    fn:
        Zero-argument callable that performs the HTTP fetch. Wrap with
        ``functools.partial`` or a ``lambda`` if your fetcher takes
        arguments (e.g. ``with_retry(lambda: fetch_ohlcv(since=d), name="ohlcv")``).
    name:
        Short label used in the "retrying..." log line. Set to the
        endpoint name for readable output.
    retries:
        Number of additional attempts after the first failure. Defaults
        to 1 (so the call is attempted at most twice).
    delay_s:
        Seconds to sleep between attempts.

    Only retries the transient connection-class errors in
    ``_TRANSIENT_HTTP_ERRORS``. HTTP 4xx / 5xx responses (wrapped as
    ``ApiError``) are NOT retried — those are server-side decisions and
    retrying changes nothing. Any other exception type bubbles up
    immediately.
    """
    attempts = retries + 1
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except _TRANSIENT_HTTP_ERRORS as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            print(
                f"  {name}: connection failed ({type(exc).__name__}), "
                f"retrying in {delay_s}s (attempt {attempt}/{attempts})"
            )
            time.sleep(delay_s)
    # Unreachable: the loop either returns on success or re-raises.
    assert last_exc is not None
    raise last_exc  # pragma: no cover


class ApiClient:
    """Object wrapper over the module-level ``fetch_*`` functions.

    Useful when callers want a single object to pass around (or to mock
    in larger integration tests). Has no state — every call goes through
    the module-level fetchers, which re-resolve the base URL each time.
    """

    def health(self) -> dict[str, Any]:
        return fetch_health()

    def snapshot_cursor(self) -> dict[str, Any]:
        return fetch_snapshot_cursor()

    def ohlcv(self, **kwargs: Any) -> pl.DataFrame:
        return fetch_ohlcv(**kwargs)

    def peer_groups(self) -> dict[str, list[str]]:
        return fetch_peer_groups()

    def symbols(self) -> dict[str, dict[str, Any]]:
        return fetch_symbols()

    def anomaly_flags(self, **kwargs: Any) -> pl.DataFrame:
        return fetch_anomaly_flags(**kwargs)
