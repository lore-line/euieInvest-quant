"""Volatility features: ATR%, Bollinger squeeze, NR4/NR7, HV ratios.

All functions take a polars DataFrame with at least ``symbol``,
``date``, ``close``, ``high``, ``low`` and return it sorted by
``(symbol, date)`` with feature columns appended.

Spec: CLAUDE.md §7 (volatility.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["atr_pct", "bb_squeeze", "nr4_nr7", "hv_ratio"]


def _true_range_expr() -> pl.Expr:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    Computed per symbol (caller is responsible for sorting + grouping
    via ``.over("symbol")``).
    """
    prev_close = pl.col("close").shift(1)
    hl = pl.col("high") - pl.col("low")
    hpc = (pl.col("high") - prev_close).abs()
    lpc = (pl.col("low") - prev_close).abs()
    return pl.max_horizontal([hl, hpc, lpc])


def atr_pct(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    """Add ``atr_pct_{window}`` = ATR(window) / close per symbol.

    ATR uses simple-mean smoothing on True Range over ``window`` days.
    Expressed as a percentage of close so it's scale-invariant across
    symbols.
    """
    out = df.sort(["symbol", "date"])
    # True Range requires shifted close, which needs per-symbol grouping
    out = out.with_columns(
        _true_range_expr().over("symbol").alias("_tr")
    )
    atr = (
        pl.col("_tr")
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
    )
    return out.with_columns(
        (atr / pl.col("close")).alias(f"atr_pct_{window}")
    ).drop("_tr")


def bb_squeeze(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add ``bb_squeeze_{window}`` per symbol.

    Bollinger band width as a fraction of the SMA:
    ``bb_squeeze = (4 * STD_w) / SMA_w``. A small value means the bands
    are tight around the mean (low recent volatility) — historically a
    setup for breakouts.
    """
    out = df.sort(["symbol", "date"])
    sma = (
        pl.col("close")
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
    )
    std = (
        pl.col("close")
        .rolling_std(window_size=window, min_samples=window)
        .over("symbol")
    )
    return out.with_columns(
        ((4.0 * std) / sma).alias(f"bb_squeeze_{window}")
    )


def nr4_nr7(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_nr4`` and ``is_nr7`` per symbol.

    - ``is_nr4`` = today's true range is the **smallest** of the last
      4 trading days (today inclusive).
    - ``is_nr7`` = same idea over the last 7 days.

    Both are boolean; null when the window can't fill.
    """
    out = df.sort(["symbol", "date"])
    out = out.with_columns(
        _true_range_expr().over("symbol").alias("_tr")
    )
    cols = []
    for n, name in ((4, "is_nr4"), (7, "is_nr7")):
        min_tr = (
            pl.col("_tr")
            .rolling_min(window_size=n, min_samples=n)
            .over("symbol")
        )
        cols.append((pl.col("_tr") == min_tr).alias(name))
    return out.with_columns(cols).drop("_tr")


def hv_ratio(
    df: pl.DataFrame, short_window: int = 10, long_window: int = 60
) -> pl.DataFrame:
    """Add ``hv_ratio_{short}_{long}`` per symbol.

    Historical-volatility ratio = ``HV(short) / HV(long)``, where HV is
    the rolling stdev of log returns. Values > 1 mean short-term vol is
    elevated relative to longer-term vol — a possible coil-then-expand
    signal.
    """
    out = df.sort(["symbol", "date"])
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log().over("symbol")
    out = out.with_columns(log_ret.alias("_log_ret"))
    hv_short = (
        pl.col("_log_ret")
        .rolling_std(window_size=short_window, min_samples=short_window)
        .over("symbol")
    )
    hv_long = (
        pl.col("_log_ret")
        .rolling_std(window_size=long_window, min_samples=long_window)
        .over("symbol")
    )
    return out.with_columns(
        (hv_short / hv_long).alias(f"hv_ratio_{short_window}_{long_window}")
    ).drop("_log_ret")
