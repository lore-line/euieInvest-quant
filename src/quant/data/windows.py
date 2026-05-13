"""60-day rolling windows over raw OHLCV channels — input shape for the
Step 2b sequence models (CNN / LSTM / Transformer).

CLAUDE.md §12 (post-2026-05-12) authorizes DL as a parallel research
track. All sequence models in Step 2b share the same input contract:

  shape: (N, channels=6, window=60)
  channels (in this order): open, high, low, close, close_adj, volume
  normalization: per-window z-score using ONLY that window's mean/std
  prediction date: t = last index of the window
  label: is_winner[t] (max(close_adj[t+1..t+30]) / close_adj[t] >= 1.20)

Per-window z-norm is intentional. It (a) prevents future-statistic
leakage (the model can't see the population mean of close that exists
in 2026 when predicting at 2024), and (b) normalizes across the wildly
different price scales in the universe (a $2 stock and a $2000 stock
get comparable representations).

The dataset materializes windows lazily — keeping all 2.4M × 60 × 6
floats in memory would be ~3.5 GB. Per-symbol contiguous buffers cost
~60 MB total; per-row slicing in __getitem__ is cheap and lets a
DataLoader stream batches to the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

__all__ = ["CHANNELS", "WINDOW", "WindowIndex", "build_window_index"]


CHANNELS: tuple[str, ...] = ("open", "high", "low", "close", "close_adj", "volume")
WINDOW: int = 60


@dataclass
class WindowIndex:
    """Lazy index over a labeled OHLCV frame.

    Holds per-symbol contiguous float32 buffers ``(n_rows, 6)`` plus a
    flat list of (symbol_id, row_index_within_symbol) pointing at the
    LAST day of each valid 60-day window. Used by ``WindowDataset`` to
    materialize one (window, label) example per ``__getitem__``.

    Construction is O(N) over the input frame; per-window slicing in
    ``__getitem__`` is O(WINDOW * channels) = 360 floats per example.
    """

    # Per-symbol stacked OHLCV float32 arrays. Shape: (total_rows, 6).
    channels: np.ndarray
    # Per-symbol starts: row[symbol_id] = first index in `channels` belonging
    # to that symbol. Length: n_symbols + 1 (last entry = total_rows).
    symbol_starts: np.ndarray
    # Flat list of valid window endpoints: (symbol_id, row_within_symbol).
    # row_within_symbol must be >= WINDOW - 1 (enough trailing data) and have
    # a non-null is_winner label.
    endpoints: np.ndarray  # shape (n_windows, 2), int32
    # Aligned to endpoints — label and date for each window's prediction day.
    labels: np.ndarray  # shape (n_windows,), int8
    dates: np.ndarray  # shape (n_windows,), datetime64[D]
    symbols: list[str]  # symbol-id → ticker mapping

    @property
    def n_windows(self) -> int:
        return self.endpoints.shape[0]


def build_window_index(labeled: pl.DataFrame) -> WindowIndex:
    """Build a ``WindowIndex`` from a labeled OHLCV polars frame.

    Parameters
    ----------
    labeled:
        Polars frame containing at minimum the columns in :data:`CHANNELS`
        plus ``symbol``, ``date``, ``is_winner``. Rows with null
        ``is_winner`` are dropped before indexing (they're the last 30
        per symbol; no forward window to evaluate).

    Returns
    -------
    WindowIndex
        Wraps per-symbol contiguous buffers + a flat endpoint list.
        Only endpoints with WINDOW-1 trailing rows of valid data are
        included.
    """
    needed = [*CHANNELS, "symbol", "date", "is_winner"]
    missing = set(needed) - set(labeled.columns)
    if missing:
        raise KeyError(f"labeled frame is missing required columns: {sorted(missing)}")

    df = labeled.sort(["symbol", "date"]).select(needed)

    symbols = sorted(df["symbol"].unique().to_list())
    sym_to_id = {s: i for i, s in enumerate(symbols)}

    # Materialize the (N, 6) channel buffer. Cast volume to float32 so
    # everything's homogeneous.
    channels_buf = np.column_stack(
        [df[c].cast(pl.Float32).to_numpy() for c in CHANNELS]
    ).astype(np.float32, copy=False)
    sym_id_per_row = np.array([sym_to_id[s] for s in df["symbol"].to_list()], dtype=np.int32)
    is_winner_per_row = df["is_winner"]
    date_per_row = df["date"].to_numpy()

    # Symbol boundaries — leverage the sorted invariant.
    symbol_starts = np.empty(len(symbols) + 1, dtype=np.int64)
    symbol_starts[0] = 0
    boundaries = np.where(np.diff(sym_id_per_row) != 0)[0] + 1
    symbol_starts[1:-1] = boundaries
    symbol_starts[-1] = len(sym_id_per_row)

    # Endpoints: per symbol, rows with index_within_symbol >= WINDOW-1
    # AND non-null is_winner. Vectorized symbol-by-symbol.
    endpoints_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    dates_chunks: list[np.ndarray] = []

    is_winner_np = is_winner_per_row.cast(pl.Int8, strict=False).to_numpy()
    is_winner_null = is_winner_per_row.is_null().to_numpy()

    for sym_id in range(len(symbols)):
        lo = symbol_starts[sym_id]
        hi = symbol_starts[sym_id + 1]
        n_rows_sym = hi - lo
        if n_rows_sym < WINDOW:
            continue
        # Local indices (within this symbol) that have a valid window+label.
        local_idx = np.arange(WINDOW - 1, n_rows_sym, dtype=np.int32)
        # Filter out rows with null is_winner.
        valid_mask = ~is_winner_null[lo + local_idx]
        local_idx = local_idx[valid_mask]
        if local_idx.size == 0:
            continue
        sym_id_col = np.full(local_idx.shape, sym_id, dtype=np.int32)
        endpoints_chunks.append(np.stack([sym_id_col, local_idx], axis=1))
        labels_chunks.append(is_winner_np[lo + local_idx])
        dates_chunks.append(date_per_row[lo + local_idx])

    endpoints = (
        np.concatenate(endpoints_chunks, axis=0)
        if endpoints_chunks
        else np.empty((0, 2), dtype=np.int32)
    )
    labels = (
        np.concatenate(labels_chunks).astype(np.int8, copy=False)
        if labels_chunks
        else np.empty(0, dtype=np.int8)
    )
    dates = (
        np.concatenate(dates_chunks)
        if dates_chunks
        else np.empty(0, dtype="datetime64[D]")
    )

    return WindowIndex(
        channels=channels_buf,
        symbol_starts=symbol_starts,
        endpoints=endpoints,
        labels=labels,
        dates=dates,
        symbols=symbols,
    )
