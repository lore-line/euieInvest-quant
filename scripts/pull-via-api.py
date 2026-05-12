"""Refresh the local snapshot cache from the euieInvest data API.

This is the preferred replacement for ``scripts/pull-snapshot.{sh,ps1}``
once the server side ships its ``/api/v1`` endpoints. While the API is
not yet live, the rsync scripts remain the operational path.

Configuration
-------------

``EUIEINVEST_API_BASE_URL``
    Base URL of the data API (no trailing slash, no ``/api/v1``
    suffix). E.g. ``http://100.68.86.56:8443``. Required.

Usage
-----

Inside the container::

    docker compose run --rm dev python scripts/pull-via-api.py
    docker compose run --rm dev python scripts/pull-via-api.py --since 2025-01-01

Output
------

Writes the following to ``data/snapshots/`` (or
``--snapshot-dir``):

- ``ohlcv.parquet``        — full or incremental price history
- ``peer_groups.json``     — sector membership dict
- ``anomaly_flags.parquet`` — doctrine flag log
- ``cursor.json``          — server-reported cursor + fetch timestamp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from quant.data import api_client

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SNAPSHOT_DIR = _REPO_ROOT / "data" / "snapshots"

# Server is currently a user-level systemd unit on claudehost; without
# `loginctl enable-linger euie` it can drop briefly after a host
# reboot. Retry ONCE on a clean connection failure before bailing, so
# routine cron runs survive the gap. This is intentionally a single
# retry — repeated failures should bubble up and page someone, not
# loop silently.
_CONNECT_RETRIES = 1
_CONNECT_RETRY_DELAY_S = 5.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull snapshot tables from the euieInvest data API"
    )
    p.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="Incremental fetch: only rows with date >= YYYY-MM-DD",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=_DEFAULT_SNAPSHOT_DIR,
        help=f"Output directory (default: {_DEFAULT_SNAPSHOT_DIR})",
    )
    return p.parse_args(argv)


def _health_with_retry() -> dict:
    """Probe /health, retrying once on a clean connection failure.

    See _CONNECT_RETRIES rationale above. We use /health as the canary
    because it's the cheapest endpoint and any infra-level problem
    will surface here before we ask for parquet bytes.
    """
    attempts = _CONNECT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return api_client.fetch_health()
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            if attempt >= attempts:
                raise
            print(
                f"  health: connection failed ({type(exc).__name__}), "
                f"retrying in {_CONNECT_RETRY_DELAY_S}s "
                f"(attempt {attempt}/{attempts})",
            )
            time.sleep(_CONNECT_RETRY_DELAY_S)
    raise RuntimeError("unreachable")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = args.snapshot_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Pulling from {api_client._base_url()}")

    health = _health_with_retry()
    print(
        f"  health: status={health['status']} "
        f"service={health['service']} v{health['service_version']}"
    )

    cursor = api_client.fetch_snapshot_cursor()
    print(
        f"  cursor: ohlcv max_date={cursor['ohlcv']['max_date']} "
        f"rows={cursor['ohlcv']['row_count']:,} | "
        f"flags max_id={cursor['anomaly_flags']['max_id']}"
    )

    ohlcv = api_client.fetch_ohlcv(since=args.since)
    ohlcv_path = out / "ohlcv.parquet"
    ohlcv.write_parquet(ohlcv_path)
    print(f"  ohlcv: wrote {ohlcv.height:,} rows -> {ohlcv_path.relative_to(_REPO_ROOT)}")

    peer_groups = api_client.fetch_peer_groups()
    pg_path = out / "peer_groups.json"
    pg_path.write_text(json.dumps(peer_groups, indent=2, sort_keys=True))
    print(
        f"  peer_groups: wrote {len(peer_groups)} groups "
        f"({sum(len(v) for v in peer_groups.values())} entries) "
        f"-> {pg_path.relative_to(_REPO_ROOT)}"
    )

    flags = api_client.fetch_anomaly_flags()
    flags_path = out / "anomaly_flags.parquet"
    flags.write_parquet(flags_path)
    print(
        f"  anomaly_flags: wrote {flags.height} rows "
        f"-> {flags_path.relative_to(_REPO_ROOT)}"
    )

    cursor_doc = {
        **cursor,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "since_filter": args.since.isoformat() if args.since else None,
    }
    (out / "cursor.json").write_text(
        json.dumps(cursor_doc, indent=2, sort_keys=True)
    )

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
