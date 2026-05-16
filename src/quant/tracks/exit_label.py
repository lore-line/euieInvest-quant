"""Exit-signal label design for Stage 2 of the quant signal contract.

Per server-team direction (PR #1 issuecomment-4467135126 in the
2026-05-16 thread), we evaluate four label-shape variants empirically
via joint-validation against Stage 1 entry signals:

| variant | gain_threshold | forward_window | giveback_threshold |
|---------|---------------|----------------|--------------------|
|  A      | 10%           | 10d            | 50%                |
|  B      | 10%           | 20d            | 50%                |
|  C      | 10%           | 10d            | 66%                |
|  D      | 15%           | 20d            | 66%                |

The label is **binary**:

  1 if (forward_window_return / 100) ≤ -giveback_threshold * (gain / 100)
  0 otherwise

Restricted to rows where the symbol is "currently up by ≥ gain_threshold"
relative to its close `--gain-lookback-days` ago (default 30, aligned
with the entry-side `expected_horizon_days = 30`).

This module is pure label computation. Training (XGB) and joint
validation live in sibling modules so each piece can be unit-tested
independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class ExitVariant:
    """Spec for a single top-prediction exit label variant."""

    name: str  # "A", "B", "C", "D"
    gain_threshold_pct: float  # filter to (symbol, date) where current gain ≥ this
    forward_window_days: int  # how far forward to measure give-back
    giveback_threshold_pct: float  # label=1 if forward return ≤ -giveback * gain
    gain_lookback_days: int = 30  # define "currently up" as gain vs N trading days ago

    def label_id(self) -> str:
        """Stable identifier embedded in output paths + manifests."""
        return (
            f"exit_{self.name}_"
            f"g{int(self.gain_threshold_pct)}_"
            f"f{self.forward_window_days}_"
            f"b{int(self.giveback_threshold_pct)}"
        )


# Convenience: the four variants from the spec
VARIANTS: dict[str, ExitVariant] = {
    "A": ExitVariant("A", 10.0, 10, 50.0),
    "B": ExitVariant("B", 10.0, 20, 50.0),
    "C": ExitVariant("C", 10.0, 10, 66.0),
    "D": ExitVariant("D", 15.0, 20, 66.0),
}


def compute_position_gain_pct(
    features: pl.DataFrame, lookback_days: int
) -> pl.DataFrame:
    """Add a `position_gain_pct` column: percent gain in close_adj over
    `lookback_days` trading days, computed per-symbol.

    The column is null for the first `lookback_days` rows per symbol
    (no lookback available). All downstream filters must handle null.

    Uses close_adj (split-and-dividend-adjusted) so corporate actions
    don't pollute the gain measurement.
    """
    if "close_adj" not in features.columns:
        raise ValueError("features.parquet must contain 'close_adj' column")
    return (
        features.sort(["symbol", "date"]).with_columns(
            position_gain_pct=(
                (pl.col("close_adj") / pl.col("close_adj").shift(lookback_days).over("symbol") - 1.0)
                * 100.0
            )
        )
    )


def compute_forward_return_pct(
    features: pl.DataFrame, forward_days: int
) -> pl.DataFrame:
    """Add a `forward_return_pct` column: percent return in close_adj from
    day t to day t+forward_days, computed per-symbol.

    Null for the last `forward_days` rows per symbol (no forward
    horizon available — these rows can't be labeled).
    """
    if "close_adj" not in features.columns:
        raise ValueError("features.parquet must contain 'close_adj' column")
    return (
        features.sort(["symbol", "date"]).with_columns(
            forward_return_pct=(
                (pl.col("close_adj").shift(-forward_days).over("symbol") / pl.col("close_adj") - 1.0)
                * 100.0
            )
        )
    )


def compute_exit_label(
    df: pl.DataFrame, variant: ExitVariant
) -> pl.DataFrame:
    """Filter df to qualified positions (current gain ≥ variant.gain_threshold_pct)
    and add a binary `exit_label` column.

    The df must already have `position_gain_pct` and `forward_return_pct`
    columns (use `prepare_label_set` for the full pipeline).

    Label semantics: 1 if forward_return_pct ≤ -giveback * position_gain_pct,
    else 0. The threshold scales with the actual gain: a position up 30%
    needs to fall more than a position up 10% to label as 1.
    """
    if "position_gain_pct" not in df.columns or "forward_return_pct" not in df.columns:
        raise ValueError("call prepare_label_set first to add gain + forward columns")
    qualified = df.filter(
        pl.col("position_gain_pct").is_not_null()
        & pl.col("forward_return_pct").is_not_null()
        & (pl.col("position_gain_pct") >= variant.gain_threshold_pct)
    )
    return qualified.with_columns(
        exit_label=(
            pl.col("forward_return_pct")
            <= -(variant.giveback_threshold_pct / 100.0) * pl.col("position_gain_pct")
        ).cast(pl.Int8)
    )


def prepare_label_set(
    features: pl.DataFrame, variant: ExitVariant
) -> pl.DataFrame:
    """End-to-end label preparation for a single variant.

    Pipeline:
      1. Compute position_gain_pct using variant.gain_lookback_days
      2. Compute forward_return_pct using variant.forward_window_days
      3. Filter to qualified positions (gain ≥ variant.gain_threshold_pct)
      4. Attach binary exit_label

    Returns a DataFrame with all original feature columns PLUS
    position_gain_pct, forward_return_pct, exit_label. Restricted to
    the labeled subset (qualified positions only).
    """
    with_gain = compute_position_gain_pct(features, variant.gain_lookback_days)
    with_forward = compute_forward_return_pct(with_gain, variant.forward_window_days)
    return compute_exit_label(with_forward, variant)


def label_statistics(
    labeled: pl.DataFrame, variant: ExitVariant
) -> dict:
    """Summary stats for a labeled set, useful for sanity-checking the
    label balance before XGB training."""
    if labeled.height == 0:
        return {
            "variant": variant.label_id(),
            "n_qualified_positions": 0,
            "n_positive_labels": 0,
            "positive_rate": 0.0,
            "mean_position_gain_pct": None,
            "mean_forward_return_pct": None,
        }
    n_pos = int(labeled.filter(pl.col("exit_label") == 1).height)
    return {
        "variant": variant.label_id(),
        "n_qualified_positions": int(labeled.height),
        "n_positive_labels": n_pos,
        "positive_rate": n_pos / labeled.height,
        "mean_position_gain_pct": float(labeled["position_gain_pct"].mean()),
        "median_position_gain_pct": float(labeled["position_gain_pct"].median()),
        "mean_forward_return_pct": float(labeled["forward_return_pct"].mean()),
        "median_forward_return_pct": float(labeled["forward_return_pct"].median()),
        "exit_trigger_at_pct": -(variant.giveback_threshold_pct / 100.0) * variant.gain_threshold_pct,
    }
