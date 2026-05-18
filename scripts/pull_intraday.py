"""Refresh the local intraday snapshot from euieInvest's /api/v1/intraday.

Writes per-interval parquets to ``data/snapshots/intraday_{N}m.parquet`` for
the universe + window the cliff/topology sweeps need. Idempotent — overwrites
existing snapshots in place. Default behavior pulls 5m + 60m (the v1
simulator's native + 1h confirmer TFs); pass ``--intervals 5,15,60`` to
include the v2 15m TF for multi-version concurrent sweeps later.

Configuration
-------------
``EUIEINVEST_API_BASE_URL``
    Same as the other pullers, e.g. ``http://100.68.86.56:8443``.

Usage
-----
    uv run scripts/pull_intraday.py --intervals 5,60 --since 2022-09-15
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Repo-root resolution: this script lives at <repo>/scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.data import api_client  # noqa: E402

DEFAULT_SNAPSHOT_DIR = ROOT / "data" / "snapshots"
DEFAULT_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
    "RUNE-USD", "FET-USD", "DOGE-USD", "XRP-USD",
    "INJ-USD", "GRT-USD", "AAVE-USD", "UNI-USD",
    "NEAR-USD", "SUSHI-USD", "APT-USD", "TIA-USD",
]


def _parse_intervals(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_date(s: str | None) -> date | None:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intervals", default="5,60",
                   help="Comma-separated bar sizes in minutes (default 5,60).")
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols. Default = 20-symbol Stream 2 universe.")
    p.add_argument("--since", default="2022-09-15", help="ISO date (YYYY-MM-DD).")
    p.add_argument("--until", default=None,
                   help="ISO date (YYYY-MM-DD); default = today (UTC).")
    p.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR),
                   help="Target dir (default <repo>/data/snapshots/).")
    args = p.parse_args()

    intervals = _parse_intervals(args.intervals)
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_UNIVERSE
    since = _parse_date(args.since)
    until = _parse_date(args.until) or datetime.now(timezone.utc).date()
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pull-intraday] universe={len(symbols)} intervals={intervals} "
          f"window={since}→{until}")
    print(f"[pull-intraday] snapshot_dir={snapshot_dir}")

    for interval_min in intervals:
        out = snapshot_dir / f"intraday_{interval_min}m.parquet"
        print(f"[pull-intraday] {interval_min}m → {out.name} ... ", end="", flush=True)
        df = api_client.with_retry(
            lambda: api_client.fetch_intraday(
                interval_min=interval_min,
                since=since,
                until=until,
                symbols=symbols,
            )
        )
        df.write_parquet(out)
        rows = df.height
        bytes_ = out.stat().st_size
        print(f"{rows:,} rows / {bytes_/1024/1024:.1f} MiB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
