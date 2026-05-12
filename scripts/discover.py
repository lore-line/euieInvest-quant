"""Discovery pipeline entrypoint — 5-step orchestrator.

Step 1 (feature engineering) is implemented. Steps 2-5 (supervised
discovery, clustering, counterfactuals, tier-3 comparison) remain
scaffolded — see CLAUDE.md §5 for the full methodology.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from quant.data.loader import load_anomaly_flags, load_ohlcv, load_peer_groups
from quant.features import (
    behavioral,
    gaps,
    momentum,
    price,
    relative,
    volatility,
    volume,
)
from quant.labels import compute_forward_winner_labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Winner-fingerprint discovery pipeline (CLAUDE.md §5)"
    )
    p.add_argument("--train-end", type=date.fromisoformat, default=date(2023, 12, 31))
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument(
        "--features-out",
        type=Path,
        default=Path("data/features/features.parquet"),
        help="Where to write the joined features + labels parquet",
    )
    return p.parse_args(argv)


def _build_features(
    ohlcv: pl.DataFrame,
    spy: pl.DataFrame,
    peer_groups_dict: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply every implemented feature transform to ``ohlcv`` and return
    a single DataFrame sorted by (symbol, date) with all feature columns
    appended.

    Features that require data missing from the current snapshot
    (`open` for gap_pct/body_range_ratio; market-cap for cap_bucket)
    are skipped here — they remain importable but unused until upstream
    catches up.
    """
    out = ohlcv
    # price.py — close/SMA, SMA slope, Bollinger band position, N-day extremes
    out = price.sma_distance(out)
    out = price.sma_slope(out, window=20, lookback=5)
    out = price.sma_slope(out, window=50, lookback=5)
    out = price.band_position(out, window=20)
    out = price.n_day_high_low(out)

    # volume.py — volume multiples, OBV slope, A/D line
    out = volume.vol_mult(out)
    out = volume.obv_slope(out, lookback=20)
    out = volume.accumulation_distribution(out)

    # volatility.py — ATR%, BB squeeze, NR4/7, HV ratio
    out = volatility.atr_pct(out, window=14)
    out = volatility.bb_squeeze(out, window=20)
    out = volatility.nr4_nr7(out)
    out = volatility.hv_ratio(out, short_window=10, long_window=60)

    # momentum.py — RSI{2,5,14}, MACD, ROC, consecutive runs
    out = momentum.rsi(out)
    out = momentum.macd(out)
    out = momentum.roc(out)
    out = momentum.consecutive_run(out)

    # gaps.py — only range_expansion and inside_bar are implementable
    # (gap_pct + body_range_ratio need `open` from upstream)
    out = gaps.range_expansion(out, lookback=5)
    out = gaps.inside_bar(out)

    # relative.py — vs SPY (full df), vs sector (peer groups), peer z-scores
    out = relative.rel_strength_spy(out, spy, lookback=20)
    out = relative.rel_strength_sector(out, peer_groups_dict, lookback=20)
    out = relative.peer_zscore(out, peer_groups_dict, column="close_over_sma_20")

    # behavioral.py — days_since_last_20pct + SPY-derived market regime
    out = behavioral.days_since_last_20pct(out)
    regime = behavioral.market_regime(spy)
    out = out.join(regime, on="date", how="left")

    return out


def step1_build_features(args: argparse.Namespace) -> pl.DataFrame:
    """Build features + labels, write to parquet, return the DataFrame."""
    print("step 1: building features ...")
    ohlcv = load_ohlcv()
    print(f"  loaded ohlcv: {ohlcv.height:,} rows, {ohlcv['symbol'].n_unique()} symbols")
    spy = load_ohlcv("SPY")
    print(f"  loaded SPY: {spy.height} rows")
    peer_groups_dict = load_peer_groups()
    print(f"  loaded peer_groups: {len(peer_groups_dict)} groups")

    features = _build_features(ohlcv, spy, peer_groups_dict)
    print(f"  built features: {features.height:,} rows × {features.width} cols")

    labeled = compute_forward_winner_labels(features, lookahead=30, threshold=0.20)
    print(f"  labeled: {labeled['is_winner'].sum()} winners "
          f"({100.0 * labeled['is_winner'].sum() / labeled['is_winner'].drop_nulls().len():.2f}% "
          f"of non-null rows)")

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    labeled.write_parquet(args.features_out)
    print(f"  wrote features+labels parquet -> {args.features_out}")
    return labeled


def step2_supervised_discovery(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step2_supervised_discovery — train XGBDiscovery "
        "with scale_pos_weight from train-set imbalance, emit SHAP summary. "
        "See CLAUDE.md §5 step 2 and §9."
    )


def step3_cluster_winners(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step3_cluster_winners — KMeans on winner-only "
        "rows for k in (3,5,8); select by silhouette. See CLAUDE.md §5 step 3."
    )


def step4_counterfactuals(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step4_counterfactuals — closest non-winners "
        "per winner cluster; report feature deltas. See CLAUDE.md §5 step 4."
    )


def step5_tier3_comparison(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step5_tier3_comparison — overlap, recall, and "
        "missed-winners vs the anomaly_flags baseline. See CLAUDE.md §5 step 5."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Sanity-touch anomaly_flags so step 5 has its hands on the baseline
    # cohort; the actual comparison is deferred to step5_tier3_comparison.
    _ = load_anomaly_flags()
    step1_build_features(args)
    step2_supervised_discovery(args)
    step3_cluster_winners(args)
    step4_counterfactuals(args)
    step5_tier3_comparison(args)


if __name__ == "__main__":
    main()
