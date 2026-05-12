"""Relative-strength features: vs SPY, vs sector, vs peer cluster.

These functions need additional inputs beyond the per-symbol OHLCV:

- ``rel_strength_spy`` needs SPY's OHLCV (load via
  ``load_ohlcv("SPY")``).
- ``rel_strength_sector`` and ``peer_zscore`` need the peer_groups dict
  from ``load_peer_groups()``.

Spec: CLAUDE.md §7 (relative.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["rel_strength_spy", "rel_strength_sector", "peer_zscore"]


def rel_strength_spy(
    df: pl.DataFrame, spy: pl.DataFrame, lookback: int = 20
) -> pl.DataFrame:
    """Add ``rs_spy_{lookback}d`` per symbol.

    ``rs_spy = symbol_return_over_lookback / spy_return_over_lookback``
    where return = ``close[t] / close[t-lookback] - 1``. Values > 1 mean
    the symbol outperformed SPY over the lookback; < 1 means
    underperformed.
    """
    out = df.sort(["symbol", "date"])
    # Compute SPY return per date (broadcast back to every symbol)
    spy_sorted = spy.sort("date").with_columns(
        (pl.col("close") / pl.col("close").shift(lookback) - 1.0).alias(
            f"_spy_ret_{lookback}d"
        )
    )
    out = out.join(
        spy_sorted.select(["date", f"_spy_ret_{lookback}d"]),
        on="date",
        how="left",
    )
    # Per-symbol return
    symbol_ret = (
        pl.col("close") / pl.col("close").shift(lookback) - 1.0
    ).over("symbol")
    out = out.with_columns(symbol_ret.alias(f"_sym_ret_{lookback}d"))
    return out.with_columns(
        (
            pl.col(f"_sym_ret_{lookback}d") / pl.col(f"_spy_ret_{lookback}d")
        ).alias(f"rs_spy_{lookback}d")
    ).drop([f"_sym_ret_{lookback}d", f"_spy_ret_{lookback}d"])


def rel_strength_sector(
    df: pl.DataFrame,
    peer_groups: dict[str, list[str]],
    lookback: int = 20,
) -> pl.DataFrame:
    """Add ``rs_sector_{lookback}d`` per symbol.

    Computes each peer-group's mean lookback return per date, then
    ``rs_sector = symbol_ret / sector_mean_ret``. Symbols not in any
    peer group get null.
    """
    out = df.sort(["symbol", "date"])
    # Per-symbol return
    out = out.with_columns(
        (pl.col("close") / pl.col("close").shift(lookback) - 1.0)
        .over("symbol")
        .alias(f"_sym_ret_{lookback}d")
    )
    # Map symbol -> group
    sym_to_group = {sym: g for g, syms in peer_groups.items() for sym in syms}
    out = out.with_columns(
        pl.col("symbol")
        .map_elements(
            lambda s: sym_to_group.get(s, None), return_dtype=pl.Utf8
        )
        .alias("_peer_group")
    )
    # Per (date, group) mean of _sym_ret_{lookback}d
    sector_mean = (
        out.filter(pl.col("_peer_group").is_not_null())
        .group_by(["date", "_peer_group"])
        .agg(
            pl.col(f"_sym_ret_{lookback}d")
            .mean()
            .alias(f"_sector_ret_{lookback}d")
        )
    )
    out = out.join(
        sector_mean, on=["date", "_peer_group"], how="left"
    )
    return out.with_columns(
        (
            pl.col(f"_sym_ret_{lookback}d")
            / pl.col(f"_sector_ret_{lookback}d")
        ).alias(f"rs_sector_{lookback}d")
    ).drop(
        [
            f"_sym_ret_{lookback}d",
            f"_sector_ret_{lookback}d",
            "_peer_group",
        ]
    )


def peer_zscore(
    df: pl.DataFrame,
    peer_groups: dict[str, list[str]],
    column: str,
) -> pl.DataFrame:
    """Add ``{column}_peer_z`` per symbol.

    z-score of ``column`` for each symbol relative to its peer group on
    the same date: ``(value - peer_mean) / peer_std``. Symbols outside
    any peer group, dates where the peer group has < 2 members, or
    rows where std == 0 produce null.
    """
    out = df.sort(["symbol", "date"])
    sym_to_group = {sym: g for g, syms in peer_groups.items() for sym in syms}
    out = out.with_columns(
        pl.col("symbol")
        .map_elements(
            lambda s: sym_to_group.get(s, None), return_dtype=pl.Utf8
        )
        .alias("_peer_group")
    )
    stats = (
        out.filter(pl.col("_peer_group").is_not_null())
        .group_by(["date", "_peer_group"])
        .agg(
            pl.col(column).mean().alias("_peer_mean"),
            pl.col(column).std().alias("_peer_std"),
            pl.len().alias("_peer_n"),
        )
    )
    out = out.join(stats, on=["date", "_peer_group"], how="left")
    z = pl.when((pl.col("_peer_std") > 0) & (pl.col("_peer_n") >= 2)).then(
        (pl.col(column) - pl.col("_peer_mean")) / pl.col("_peer_std")
    ).otherwise(None)
    return out.with_columns(z.alias(f"{column}_peer_z")).drop(
        ["_peer_mean", "_peer_std", "_peer_n", "_peer_group"]
    )
