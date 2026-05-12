"""Behavioral / regime features: recency of prior winner-grade runs,
market regime classification, market-cap bucketing.

Spec: CLAUDE.md §7 (behavioral.py).
"""
from __future__ import annotations

import polars as pl

__all__ = ["days_since_last_20pct", "market_regime", "cap_bucket"]


def days_since_last_20pct(
    df: pl.DataFrame, lookahead: int = 30, threshold: float = 0.20
) -> pl.DataFrame:
    """Add ``days_since_last_20pct`` per symbol.

    Counts trading days since the most recent **event**, where an event
    is the *first* day a winner-grade backward move is observed. A
    backward winner at row t means ``close[t] / min(close[t-lookahead..t-1])
    >= 1 + threshold`` — i.e., today's close is +20% above the lowest
    close in the prior 30 days. Consecutive elevated days do NOT count
    as separate events.

    Null while there's no event yet (including the warm-up window where
    the rolling min isn't filled). BACKWARD-LOOKING and safe to use as
    a predictor at time t.
    """
    out = df.sort(["symbol", "date"])
    # Backward winner: today's close vs min over the prior `lookahead` days
    min_back = (
        pl.col("close")
        .rolling_min(window_size=lookahead, min_samples=lookahead)
        .shift(1)
        .over("symbol")
    )
    was_winner = pl.col("close") / min_back >= 1.0 + threshold
    # "Event" = first winner day after a non-winner day (or after warm-up)
    prior_not_winner = (
        was_winner.shift(1).over("symbol").fill_null(False).not_()
    )
    event = was_winner.fill_null(False) & prior_not_winner
    out = out.with_columns(
        [
            pl.int_range(0, pl.len()).over("symbol").alias("_idx"),
            event.alias("_event"),
        ]
    )
    # Forward-fill the index of the most recent event per symbol; null
    # while no event has been seen yet
    out = out.with_columns(
        pl.when(pl.col("_event"))
        .then(pl.col("_idx"))
        .otherwise(None)
        .forward_fill()
        .over("symbol")
        .alias("_last_event_idx")
    )
    return out.with_columns(
        (pl.col("_idx") - pl.col("_last_event_idx")).alias(
            "days_since_last_20pct"
        )
    ).drop(["_idx", "_event", "_last_event_idx"])


def market_regime(spy: pl.DataFrame) -> pl.DataFrame:
    """Classify SPY regime per date.

    Returns a DataFrame with columns ``date`` and ``market_regime``,
    where ``market_regime`` is one of:

    - ``"uptrend"`` — SPY close > SMA50 > SMA200
    - ``"downtrend"`` — SPY close < SMA50 < SMA200
    - ``"chop"`` — anything else

    Join this onto the per-symbol features on ``date`` to broadcast.
    """
    out = spy.sort("date").with_columns(
        [
            pl.col("close")
            .rolling_mean(window_size=50, min_samples=50)
            .alias("_sma50"),
            pl.col("close")
            .rolling_mean(window_size=200, min_samples=200)
            .alias("_sma200"),
        ]
    )
    regime = (
        pl.when(
            (pl.col("close") > pl.col("_sma50"))
            & (pl.col("_sma50") > pl.col("_sma200"))
        )
        .then(pl.lit("uptrend"))
        .when(
            (pl.col("close") < pl.col("_sma50"))
            & (pl.col("_sma50") < pl.col("_sma200"))
        )
        .then(pl.lit("downtrend"))
        .otherwise(pl.lit("chop"))
    )
    return out.with_columns(regime.alias("market_regime")).select(
        ["date", "market_regime"]
    )


def cap_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``cap_bucket`` per symbol (mega/large/mid/small/micro).

    BLOCKED: market-cap is not in the current snapshot. Options:
    (a) request the server team add a ``market_cap`` column to
        ``/api/v1/ohlcv`` (additive, non-breaking) or a separate
        ``/api/v1/symbols`` endpoint with static metadata, or
    (b) join from an external static reference table at feature-build
        time.
    """
    raise NotImplementedError(
        "src/quant/features/behavioral.py: cap_bucket — requires market-cap "
        "data not present in the current snapshot. See module docstring "
        "for the two upstream paths."
    )
