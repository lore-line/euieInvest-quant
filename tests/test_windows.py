"""Tests for ``quant.data.windows.build_window_index``."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quant.data.windows import CHANNELS, WINDOW, build_window_index


def _fake_labeled(n_rows_per_symbol: list[int], start: date = date(2024, 1, 1)) -> pl.DataFrame:
    """Build a minimal labeled OHLCV frame with synthetic ramps.

    Each symbol gets ``n_rows_per_symbol[i]`` consecutive trading days.
    `is_winner` is True on every odd-indexed row, False on every even-indexed
    row (excluding the last 30 per symbol, which are null — mimicking the
    forward-lookahead truncation).
    """
    frames = []
    for sym_i, n_rows in enumerate(n_rows_per_symbol):
        symbol = f"SYM{sym_i:02d}"
        dates = [start + timedelta(days=i) for i in range(n_rows)]
        base = 10.0 + sym_i
        prices = np.linspace(base, base * 1.5, n_rows)
        is_winner: list[bool | None] = [
            (i % 2 == 1) if i < n_rows - 30 else None for i in range(n_rows)
        ]
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [symbol] * n_rows,
                    "date": dates,
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                    "close_adj": prices,
                    "volume": np.linspace(1000, 2000, n_rows).astype(np.int64),
                    "is_winner": pl.Series("is_winner", is_winner, dtype=pl.Boolean),
                }
            )
        )
    return pl.concat(frames)


def test_channels_order_matches_contract() -> None:
    assert CHANNELS == ("open", "high", "low", "close", "close_adj", "volume")
    assert WINDOW == 60


def test_endpoints_skip_warmup_and_null_label() -> None:
    # 200 rows per symbol → 200 - WINDOW + 1 = 141 candidate endpoints,
    # but the last 30 have null is_winner → 141 - 30 = 111 valid endpoints
    df = _fake_labeled([200])
    idx = build_window_index(df)
    assert idx.n_windows == 200 - WINDOW + 1 - 30
    # All endpoints' local indices in [WINDOW-1, n_rows-31]
    locals_ = idx.endpoints[:, 1]
    assert locals_.min() == WINDOW - 1
    assert locals_.max() == 200 - 31


def test_symbol_isolation() -> None:
    df = _fake_labeled([120, 120])
    idx = build_window_index(df)
    # Both symbols contribute equally.
    n_per_sym = idx.n_windows // 2
    assert (idx.endpoints[:, 0] == 0).sum() == n_per_sym
    assert (idx.endpoints[:, 0] == 1).sum() == n_per_sym


def test_short_symbol_contributes_nothing() -> None:
    # Symbol with < WINDOW rows should produce zero windows
    df = _fake_labeled([WINDOW - 1])
    idx = build_window_index(df)
    assert idx.n_windows == 0


def test_channels_buffer_aligns_with_endpoints() -> None:
    df = _fake_labeled([100])
    idx = build_window_index(df)
    # Pick the first endpoint and check the slice matches the source frame
    sym_id, local_end = idx.endpoints[0]
    global_start = idx.symbol_starts[sym_id] + local_end - WINDOW + 1
    global_end = idx.symbol_starts[sym_id] + local_end
    window = idx.channels[global_start : global_end + 1]
    assert window.shape == (WINDOW, len(CHANNELS))
    # The 'close' column is index 3 — should match the source frame's `close`
    source = df.sort(["symbol", "date"])["close"].to_numpy()
    np.testing.assert_allclose(window[:, 3], source[global_start : global_end + 1])


def test_missing_channel_raises() -> None:
    df = _fake_labeled([100]).drop("close_adj")
    with pytest.raises(KeyError, match="close_adj"):
        build_window_index(df)
