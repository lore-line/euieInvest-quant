"""Momentum features: RSI, MACD, ROC, consecutive runs.

All functions take a polars DataFrame with at least ``symbol``, ``date``,
``close`` and return it sorted by ``(symbol, date)`` with feature
columns appended.

Spec: CLAUDE.md §7 (momentum.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["rsi", "macd", "roc", "consecutive_run"]


def rsi(df: pl.DataFrame, windows: tuple[int, ...] = (2, 5, 14)) -> pl.DataFrame:
    """Add ``rsi_{N}`` for each N per symbol.

    Wilder's RSI:
    1. ``delta = close.diff()`` per symbol
    2. ``gain = max(delta, 0)``; ``loss = max(-delta, 0)``
    3. Simple-mean smoothed gain/loss over N (Wilder's true smoothing
       uses ``alpha=1/N`` EMA; we use the simple-mean variant as an
       acceptable approximation for feature-engineering purposes).
    4. ``rs = avg_gain / avg_loss``
    5. ``rsi = 100 - 100/(1+rs)``
    """
    out = df.sort(["symbol", "date"])
    delta = pl.col("close").diff().over("symbol")
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    out = out.with_columns([gain.alias("_gain"), loss.alias("_loss")])
    cols = []
    for n in windows:
        avg_gain = (
            pl.col("_gain")
            .rolling_mean(window_size=n, min_samples=n)
            .over("symbol")
        )
        avg_loss = (
            pl.col("_loss")
            .rolling_mean(window_size=n, min_samples=n)
            .over("symbol")
        )
        # When avg_loss == 0, rs is infinite → rsi = 100. Guard with when().
        rs = avg_gain / avg_loss
        cols.append(
            pl.when(avg_loss == 0)
            .then(100.0)
            .otherwise(100.0 - 100.0 / (1.0 + rs))
            .alias(f"rsi_{n}")
        )
    return out.with_columns(cols).drop(["_gain", "_loss"])


def macd(
    df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pl.DataFrame:
    """Add ``macd_line``, ``macd_signal``, ``macd_hist`` per symbol.

    - ``macd_line = EMA_fast(close) - EMA_slow(close)``
    - ``macd_signal = EMA_signal(macd_line)``
    - ``macd_hist = macd_line - macd_signal``
    """
    out = df.sort(["symbol", "date"])
    ema_fast = (
        pl.col("close")
        .ewm_mean(span=fast, adjust=False, min_samples=fast)
        .over("symbol")
    )
    ema_slow = (
        pl.col("close")
        .ewm_mean(span=slow, adjust=False, min_samples=slow)
        .over("symbol")
    )
    out = out.with_columns((ema_fast - ema_slow).alias("macd_line"))
    macd_signal = (
        pl.col("macd_line")
        .ewm_mean(span=signal, adjust=False, min_samples=signal)
        .over("symbol")
    )
    out = out.with_columns(macd_signal.alias("macd_signal"))
    return out.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
    )


def roc(
    df: pl.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 60)
) -> pl.DataFrame:
    """Add ``roc_{N}`` = ``(close[t] / close[t-N]) - 1`` per symbol.

    Rate of change as a fraction (not %). Trailing window; null while
    insufficient history.
    """
    out = df.sort(["symbol", "date"])
    cols = [
        (
            pl.col("close") / pl.col("close").shift(n).over("symbol") - 1.0
        ).alias(f"roc_{n}")
        for n in windows
    ]
    return out.with_columns(cols)


def consecutive_run(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``consec_up`` and ``consec_down`` per symbol.

    Run-length of consecutive up days (close > prev_close) and down days
    (close < prev_close). Counter resets to 0 when direction changes or
    stays flat. First row per symbol is 0 / 0 (no prior close to compare).
    """
    out = df.sort(["symbol", "date"])
    # diff().fill_null(0.0) treats the first row as "no movement"
    delta = pl.col("close").diff().over("symbol").fill_null(0.0)
    up_day = (delta > 0).cast(pl.Int64)
    down_day = (delta < 0).cast(pl.Int64)
    out = out.with_columns([up_day.alias("_up"), down_day.alias("_down")])

    # Run-length idiom: build a "break" group id that increments on
    # every non-up day, then cum_sum within (symbol, break) for the run
    # length. Same for down days.
    out = out.with_columns(
        [
            ((pl.col("_up") == 0).cast(pl.Int64).cum_sum().over("symbol"))
            .alias("_up_break"),
            ((pl.col("_down") == 0).cast(pl.Int64).cum_sum().over("symbol"))
            .alias("_down_break"),
        ]
    )
    out = out.with_columns(
        [
            pl.col("_up")
            .cum_sum()
            .over(["symbol", "_up_break"])
            .alias("consec_up"),
            pl.col("_down")
            .cum_sum()
            .over(["symbol", "_down_break"])
            .alias("consec_down"),
        ]
    )
    return out.drop(["_up", "_down", "_up_break", "_down_break"])
