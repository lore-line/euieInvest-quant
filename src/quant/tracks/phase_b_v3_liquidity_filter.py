"""Phase B v3 liquidity filter — Stream 2 friction-trap escape attempt.

Per PR #1 issuecomment-4472873962 server-team direction. The walkforward
v3 corpus fails ATR-scaled friction (-1.2% to -1.8% annualized) because
it's dominated by sub-$5 small-cap stop hits where ATR-driven slippage
dwarfs the asymmetric R:R.

This module applies a post-hoc liquidity filter to an existing
signals.parquet (from `paper_sleeve_simulate`): drop trades where
entered_price < $min_price OR rolling-30d avg volume at entry < $min_volume.

Same rules, more liquid trading universe. Tells us whether the friction
problem is rule-quality (still fails after liquidity filter) vs universe-
quality (passes after filter, in which case full re-extract is justified).

The choice of post-hoc filter over full re-extract trades modeling
fidelity for compute speed — ~10 min vs ~1 hr. If results show net
Sharpe ≥ 0.8 under ATR-scaled, the next step IS the full re-extract
(rules trained on liquid cohort would be even cleaner).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "phase_b_v3_liquidity_filter_v1"
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_AVG_VOLUME_30D = 1_000_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--signals", type=Path, required=True,
        help="Path to signals.parquet from paper_sleeve_simulate.",
    )
    p.add_argument(
        "--features", type=Path, default=Path("data/features/features.parquet"),
        help="Source of price + volume for the liquidity join.",
    )
    p.add_argument(
        "--min-price", type=float, default=DEFAULT_MIN_PRICE,
        help=f"Drop trades where entered_price < this. Default ${DEFAULT_MIN_PRICE}.",
    )
    p.add_argument(
        "--min-avg-volume-30d", type=int, default=DEFAULT_MIN_AVG_VOLUME_30D,
        help=f"Drop trades where rolling-30d avg vol at entry < this. Default {DEFAULT_MIN_AVG_VOLUME_30D:,}.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output filtered signals path. Default: signals dir / filtered_signals.parquet",
    )
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    signals_path = _resolve(args.signals)
    features_path = _resolve(args.features)
    if not signals_path.exists():
        print(f"ERROR: {signals_path} not found")
        return 1
    out_path = _resolve(args.out) if args.out else (signals_path.parent / "filtered_signals.parquet")

    print(f"phase_b_v3_liquidity_filter v1")
    print(f"  signals:               {signals_path}")
    print(f"  features:              {features_path}")
    print(f"  min_price:             ${args.min_price}")
    print(f"  min_avg_volume_30d:    {args.min_avg_volume_30d:,}")

    sig = pl.read_parquet(signals_path)
    n_before = sig.height
    print(f"  loaded {n_before:,} trades")

    # Build rolling-30d avg volume per (symbol, date) from features.parquet
    # + capture close_adj + atr_pct_14 for downstream friction profile use.
    feats = pl.read_parquet(features_path).select(
        ["symbol", "date", "close_adj", "volume", "atr_pct_14"]
    )
    feats = feats.sort(["symbol", "date"]).with_columns(
        avg_volume_30d=pl.col("volume").rolling_mean(window_size=30).over("symbol"),
    )
    print(f"  computed rolling-30d avg volume on {feats.height:,} feature rows")

    # Parse signals.entered_at (string ISO datetime) → date for join
    if sig["entered_at"].dtype == pl.String:
        sig = sig.with_columns(
            entered_date=pl.col("entered_at").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%:z", time_zone="UTC", strict=False,
            ).dt.date()
        )
    else:
        sig = sig.with_columns(entered_date=pl.col("entered_at").dt.date())

    # Join: each trade gets the avg_volume_30d + close_adj + atr_pct_14
    # at its entry date for that symbol
    enriched = sig.join(
        feats.rename({"date": "entered_date"}).select(
            ["symbol", "entered_date", "close_adj", "avg_volume_30d", "atr_pct_14"]
        ).rename({
            "close_adj": "entry_close_adj",
            "atr_pct_14": "entry_atr_pct_14",
        }),
        on=["symbol", "entered_date"], how="left",
    )
    n_with_features = enriched.filter(
        pl.col("avg_volume_30d").is_not_null()
    ).height
    print(f"  joined to features: {n_with_features:,} trades have feature data "
          f"({n_before - n_with_features:,} dropped — symbol×date not in features)")

    # Apply liquidity filter
    filtered = enriched.filter(
        (pl.col("entered_price") >= args.min_price)
        & (pl.col("avg_volume_30d") >= args.min_avg_volume_30d)
    )
    n_after = filtered.height
    print(f"  after liquidity filter: {n_after:,} trades "
          f"(dropped {n_before - n_after:,}, kept {100*n_after/max(1,n_before):.1f}%)")

    # Distribution of what got dropped
    dropped = enriched.filter(
        ~((pl.col("entered_price") >= args.min_price)
          & (pl.col("avg_volume_30d") >= args.min_avg_volume_30d))
    )
    n_price_fail = enriched.filter(pl.col("entered_price") < args.min_price).height
    n_vol_fail = enriched.filter(
        pl.col("avg_volume_30d") < args.min_avg_volume_30d
    ).height
    print(f"    dropped by price <${args.min_price}: {n_price_fail:,}")
    print(f"    dropped by avg_vol <{args.min_avg_volume_30d:,}: {n_vol_fail:,}")

    # Re-aggregate net pnl on the filtered set (gross numbers — let friction
    # extension compute net)
    if n_after > 0:
        kept_gross_pnl = float(filtered["realized_pnl_usd"].sum())
        kept_win_rate = float(
            (filtered["realized_pnl_usd"] > 0).cast(pl.Int8).mean()
        )
        print(f"  filtered gross_pnl: ${kept_gross_pnl:,.0f}  win_rate: {100*kept_win_rate:.1f}%")
    else:
        print(f"  WARNING: zero trades after filter")

    # Drop the helper join column before output
    filtered.drop("entered_date").write_parquet(out_path)
    print(f"  wrote {out_path}")

    # Summary JSON for downstream consumers
    summary = {
        "pipeline_step": PIPELINE_STEP,
        "signals_in": str(signals_path),
        "signals_out": str(out_path),
        "filter": {
            "min_price_usd": args.min_price,
            "min_avg_volume_30d": args.min_avg_volume_30d,
        },
        "counts": {
            "n_input_trades": int(n_before),
            "n_with_feature_data": int(n_with_features),
            "n_dropped_price": int(n_price_fail),
            "n_dropped_volume": int(n_vol_fail),
            "n_kept": int(n_after),
            "kept_pct": round(100.0 * n_after / max(1, n_before), 2),
        },
        "filtered_gross_summary": (
            {
                "total_gross_pnl_usd": kept_gross_pnl,
                "win_rate": kept_win_rate,
            }
            if n_after > 0 else None
        ),
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    (out_path.parent / "liquidity_filter_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
