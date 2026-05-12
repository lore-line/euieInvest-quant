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

``cursor.json`` is written LAST so a crash mid-pull leaves the
previous cursor intact; the next run sees the old cursor and
re-fetches from scratch.

Each fetch is wrapped in ``api_client.with_retry`` so a transient
connection failure (typically: a brief server drop after a claudehost
reboot, while the systemd unit's linger is not yet enabled) retries
once before bailing.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path

from quant.data import api_client

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SNAPSHOT_DIR = _REPO_ROOT / "data" / "snapshots"


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = args.snapshot_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Pulling from {api_client._base_url()}")

    health = api_client.with_retry(api_client.fetch_health, name="health")
    print(
        f"  health: status={health['status']} "
        f"service={health['service']} v{health['service_version']}"
    )

    cursor = api_client.with_retry(
        api_client.fetch_snapshot_cursor, name="snapshot-cursor"
    )
    print(
        f"  cursor: ohlcv max_date={cursor['ohlcv']['max_date']} "
        f"rows={cursor['ohlcv']['row_count']:,} | "
        f"flags max_id={cursor['anomaly_flags']['max_id']}"
    )

    ohlcv = api_client.with_retry(
        partial(api_client.fetch_ohlcv, since=args.since), name="ohlcv"
    )
    ohlcv_path = out / "ohlcv.parquet"
    ohlcv.write_parquet(ohlcv_path)
    print(f"  ohlcv: wrote {ohlcv.height:,} rows -> {ohlcv_path.relative_to(_REPO_ROOT)}")

    peer_groups = api_client.with_retry(
        api_client.fetch_peer_groups, name="peer-groups"
    )
    pg_path = out / "peer_groups.json"
    pg_path.write_text(json.dumps(peer_groups, indent=2, sort_keys=True))
    print(
        f"  peer_groups: wrote {len(peer_groups)} groups "
        f"({sum(len(v) for v in peer_groups.values())} entries) "
        f"-> {pg_path.relative_to(_REPO_ROOT)}"
    )

    flags = api_client.with_retry(
        api_client.fetch_anomaly_flags, name="anomaly-flags"
    )
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
