"""Batched intraday puller that concats sub-pulls into one parquet.

Use when the full universe × wide window blows API memory. Pulls in
N-symbol chunks, concatenates, writes one final parquet per interval.

Usage:
    uv run scripts/pull_intraday_batched.py --intervals 5,15,60 \\
        --batch-size 7 --symbols A,B,C,...
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant.data import api_client  # noqa: E402


def _parse_date(s: str | None) -> date | None:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--intervals", default="5,15,60")
    p.add_argument("--symbols", required=True)
    p.add_argument("--since", default="2022-09-15")
    p.add_argument("--until", default=None)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--snapshot-dir",
                   default=str(ROOT / "data" / "snapshots"))
    args = p.parse_args()

    intervals = [int(x) for x in args.intervals.split(",")]
    symbols = [s.strip() for s in args.symbols.split(",")]
    since = _parse_date(args.since)
    until = _parse_date(args.until) or datetime.now(timezone.utc).date()
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pull-batched] {len(symbols)} symbols × {intervals} intervals "
          f"in chunks of {args.batch_size}")
    print(f"[pull-batched] window={since}→{until}")

    for interval_min in intervals:
        out = snapshot_dir / f"intraday_{interval_min}m.parquet"
        chunks = []
        for i in range(0, len(symbols), args.batch_size):
            batch = symbols[i:i+args.batch_size]
            print(f"[pull-batched] {interval_min}m batch {i//args.batch_size + 1}: "
                  f"{batch[0]}..{batch[-1]} ({len(batch)} syms) ... ",
                  end="", flush=True)
            df = api_client.with_retry(
                lambda: api_client.fetch_intraday(
                    interval_min=interval_min,
                    since=since, until=until,
                    symbols=batch,
                ),
                name=f"intraday/{interval_min}m/batch{i//args.batch_size + 1}",
            )
            chunks.append(df)
            print(f"{df.height:,} rows")
        merged = pl.concat(chunks)
        merged.write_parquet(out)
        print(f"[pull-batched] {interval_min}m TOTAL: {merged.height:,} rows / "
              f"{out.stat().st_size/1024/1024:.1f} MiB → {out.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
