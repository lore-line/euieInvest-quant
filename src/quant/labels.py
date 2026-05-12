"""Forward-looking winner labels.

A 'winner' is a row where ``max(price[t+1..t+lookahead]) / price[t] >= 1+threshold``
(default +20% within the next 30 trading days). Strictly forward-looking;
never references the current or past price beyond ``price[t]``.

The canonical price column is ``close_adj`` (split + dividend-adjusted)
per CLAUDE.md §6 — pass ``price_col="close_adj"``. The default ``"close"``
exists for back-compat with older callers/tests.
"""
from __future__ import annotations

import polars as pl

__all__ = ["compute_forward_winner_labels"]


def compute_forward_winner_labels(
    df: pl.DataFrame,
    lookahead: int = 30,
    threshold: float = 0.20,
    price_col: str = "close",
) -> pl.DataFrame:
    """Append an ``is_winner`` boolean column to ``df``.

    Parameters
    ----------
    df:
        Polars DataFrame containing at least ``symbol``, ``date``, and
        the column named by ``price_col``.
    lookahead:
        Number of forward rows (per symbol) to scan. Default 30.
    threshold:
        Minimum forward gain to count as a winner (e.g. 0.20 = +20%).
    price_col:
        Column to compute the forward gain against. Pass ``"close_adj"``
        for the canonical CLAUDE.md §6 total-return label.

    Returns
    -------
    pl.DataFrame
        Same rows as ``df`` (sorted by symbol, date) plus an
        ``is_winner: pl.Boolean`` column. The final ``lookahead`` rows
        per symbol have null ``is_winner`` (insufficient forward data).
    """
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if price_col not in df.columns:
        raise KeyError(f"price_col {price_col!r} not in df.columns={df.columns}")

    # Forward max over price[t+1..t+lookahead], per symbol.
    # rolling_max(N) at row s reads price[s-N+1..s]; shifting that column
    # back by N rows maps row t to rolling_max[t+N] = max(price[t+1..t+N]).
    # min_samples=N + the shift jointly guarantee null for the last N rows
    # per symbol (insufficient forward data).
    out = df.sort(["symbol", "date"]).with_columns(
        _forward_max=pl.col(price_col)
        .rolling_max(window_size=lookahead, min_samples=lookahead)
        .shift(-lookahead)
        .over("symbol")
    )
    out = out.with_columns(
        is_winner=(pl.col("_forward_max") / pl.col(price_col)) >= (1.0 + threshold)
    )
    return out.drop("_forward_max")
