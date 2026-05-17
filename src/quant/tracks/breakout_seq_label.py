"""breakout_seq_v1 cohort label — Workstream D (DL Angle 2 commissioned spec).

Per PR #1 issuecomment-4469607665 (server team, 2026-05-17 06:29Z):

  For each (symbol, entry_date) where close >= $1.00 at entry:
    winner_label = TRUE iff max(close[entry+1 .. entry+60]) >= entry * 1.20

NO sustained constraint. The label is purely "peak ≥+20% within 60 trading
days post-entry" — letting the 1D CNN find pre-breakout shape patterns
whose timing-to-peak varies across the 3-month window.

Key differences from `sustained_winner_label`:
  - Horizon: 60 trading days (vs 20)
  - No endpoint constraint (only the peak matters)
  - Cohort base rate expected ~3x larger (longer window catches more winners)

Exit mechanic for joint validation (defined here for label-realized-return
computation; the trading platform sees only the SIGNAL, not the exit logic):
  - Buy on signal at next-day open
  - Exit at FIRST of: (a) +20% target hit (next-day close >= entry * 1.20)
                     (b) day 60 close (hard cap)
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


PIPELINE_STEP = "breakout_seq_v1"
PATTERN_PREFIX = "bsq60"  # "breakout sequence 60-day"; the g{NN} part comes from spec.name


@dataclass(frozen=True)
class BreakoutSeqSpec:
    """Spec for the breakout_seq_v1 cohort label.

    Defaults match the server-team commissioned spec exactly. The Pareto
    sweep parameter `g` (gain threshold) is exposed for future expansion
    but v1 ships at g=20.
    """

    name: str = "g20"
    touch_threshold_pct: float = 20.0
    horizon_days: int = 60  # trading days
    min_entry_price_usd: float = 1.00

    def label_column(self) -> str:
        return f"is_breakout_seq_{self.name}"

    def pattern(self, rule_id: int | str) -> str:
        return f"{PATTERN_PREFIX}_{self.name}_rule_{rule_id}"


SPEC_DEFAULT = BreakoutSeqSpec()


def compute_breakout_seq_label(
    features: pl.DataFrame, spec: BreakoutSeqSpec = SPEC_DEFAULT,
) -> pl.DataFrame:
    """Add the breakout_seq label column to features.

    The features frame must have `symbol`, `date`, and `close_adj` columns
    (sortable per-symbol by date).

    Returns features with two new columns:
      - `forward_max_pct_60td` — max forward-60-day return (percent)
      - `<spec.label_column()>` — boolean (or null if forward window
        unavailable / fails the $1 price floor)

    Caller MUST handle null on the label column when training.
    """
    if "close_adj" not in features.columns:
        raise ValueError("features must contain close_adj")
    n = spec.horizon_days
    sorted_features = features.sort(["symbol", "date"])

    # Same shift-then-max idiom as sustained_winner_label, with 60-day window
    shift_cols = [
        pl.col("close_adj").shift(-i).over("symbol").alias(f"_fwd_close_{i}")
        for i in range(1, n + 1)
    ]
    out = sorted_features.with_columns(shift_cols)

    # Forward window is "complete" only when the LAST day (day N) is non-null;
    # if it's null, the (symbol, date) row is too close to the end of the data.
    forward_close_max = pl.max_horizontal(
        [pl.col(f"_fwd_close_{i}") for i in range(1, n + 1)]
    )
    window_complete = pl.col(f"_fwd_close_{n}").is_not_null()
    out = out.with_columns(
        forward_max_pct_60td=pl.when(window_complete)
        .then((forward_close_max / pl.col("close_adj") - 1.0) * 100.0)
        .otherwise(None),
    )
    # Drop bookkeeping columns
    out = out.drop([f"_fwd_close_{i}" for i in range(1, n + 1)])

    # Compute label: True if entry price >= floor AND peak >= +g%
    label_col = spec.label_column()
    out = out.with_columns(
        **{
            label_col: pl.when(
                (pl.col("close_adj") >= spec.min_entry_price_usd)
                & (pl.col("forward_max_pct_60td").is_not_null())
            )
            .then(pl.col("forward_max_pct_60td") >= spec.touch_threshold_pct)
            .otherwise(None)
        }
    )
    return out


def compute_realized_returns_60td(
    features: pl.DataFrame, spec: BreakoutSeqSpec = SPEC_DEFAULT,
) -> pl.DataFrame:
    """Compute REALIZED returns assuming the hard-cap-60d exit mechanic:

    For each (symbol, entry_date):
      - Buy at close[entry] (proxy for next-day open; close-based is conservative)
      - Exit at first of:
        (a) close[t] >= entry * 1.20 for some t in [entry+1, entry+60]
        (b) close[entry+60] (hard cap)
      - realized_gain_pct = (exit_price / entry_price - 1.0) * 100
      - hold_days = number of trading days until exit (1..60)

    Returns features with new columns:
      - `bsq_realized_gain_pct`
      - `bsq_hold_trading_days`
      - `bsq_exit_reason` ("target_hit" or "day60_cap")

    Rows where the 60-day forward window is unavailable get null on all three.
    """
    if "close_adj" not in features.columns:
        raise ValueError("features must contain close_adj")
    n = spec.horizon_days
    target_mult = 1.0 + spec.touch_threshold_pct / 100.0

    sorted_features = features.sort(["symbol", "date"])

    # Build the 60 forward closes per row
    shift_cols = [
        pl.col("close_adj").shift(-i).over("symbol").alias(f"_fwd_close_{i}")
        for i in range(1, n + 1)
    ]
    out = sorted_features.with_columns(shift_cols)

    # For each forward day, compute (fwd_close / entry_close - 1) and check
    # whether it crosses the target. The exit is the FIRST such day; if none
    # crosses, exit is day 60.
    # Polars-idiomatic: iterate columns; for each day i, compute "is target
    # reached at day i" mask = (fwd_close_i / entry >= target_mult).
    # The "first hit" day = argmin over i where the mask is True.
    #
    # For correctness + simplicity I do this with an explicit polars expr that
    # materializes per-day target-hit flags then folds them.

    # First-hit day expr: returns 1..60 if any hit, else null
    # Implementation: cumulative-OR doesn't exist in polars; use horizontal
    # min over (i if hit else 9999), clamp.
    hit_day_exprs = [
        pl.when(
            (pl.col(f"_fwd_close_{i}") / pl.col("close_adj")) >= target_mult
        ).then(pl.lit(i)).otherwise(pl.lit(10_000)).alias(f"_hit_day_{i}")
        for i in range(1, n + 1)
    ]
    out = out.with_columns(hit_day_exprs)
    first_hit_day = pl.min_horizontal([pl.col(f"_hit_day_{i}") for i in range(1, n + 1)])
    out = out.with_columns(_first_hit_day=first_hit_day)
    out = out.drop([f"_hit_day_{i}" for i in range(1, n + 1)])

    # Exit price = close at first_hit_day if hit (target_mult * entry), else close at day 60
    # When target is hit at day h: realized_gain = (close[h]/entry - 1)*100
    #   ≥ target_threshold by construction
    # When no hit: realized_gain = (close[60]/entry - 1)*100
    #   could be positive, zero, or negative

    # For the realized-gain calculation we need the actual close at first_hit_day.
    # The simplest approach: build a list-of-floats column with all 60 forward closes,
    # then index into it by first_hit_day. Polars list-indexing supports this.
    forward_list = pl.concat_list(
        [pl.col(f"_fwd_close_{i}") for i in range(1, n + 1)]
    ).alias("_fwd_list")
    out = out.with_columns(forward_list)

    # Exit-day close: if first_hit_day <= 60, use forward_list[first_hit_day-1],
    # else use forward_list[59] (day 60 = index 59)
    out = out.with_columns(
        _exit_idx=pl.when(pl.col("_first_hit_day") <= n)
        .then(pl.col("_first_hit_day") - 1)
        .otherwise(n - 1),
    )

    # Pull exit price from the forward list using the computed index
    out = out.with_columns(
        _exit_close=pl.col("_fwd_list").list.get(pl.col("_exit_idx").cast(pl.Int64))
    )

    # Compute outputs
    out = out.with_columns(
        bsq_realized_gain_pct=pl.when(pl.col("_exit_close").is_not_null())
        .then((pl.col("_exit_close") / pl.col("close_adj") - 1.0) * 100.0)
        .otherwise(None),
        bsq_hold_trading_days=pl.when(pl.col("_first_hit_day") <= n)
        .then(pl.col("_first_hit_day"))
        .otherwise(n),
        bsq_exit_reason=pl.when(pl.col("_first_hit_day") <= n)
        .then(pl.lit("target_hit"))
        .otherwise(pl.lit("day60_cap")),
    )

    # Clean up bookkeeping columns
    out = out.drop([
        f"_fwd_close_{i}" for i in range(1, n + 1)
    ] + ["_first_hit_day", "_fwd_list", "_exit_idx", "_exit_close"])

    # Mask out rows where the 60-day window doesn't exist (last 60 per symbol)
    out = out.with_columns(
        bsq_realized_gain_pct=pl.when(
            (pl.col("close_adj") >= spec.min_entry_price_usd)
        ).then(pl.col("bsq_realized_gain_pct")).otherwise(None),
        bsq_hold_trading_days=pl.when(
            (pl.col("close_adj") >= spec.min_entry_price_usd)
            & pl.col("bsq_realized_gain_pct").is_not_null()
        ).then(pl.col("bsq_hold_trading_days")).otherwise(None),
    )

    return out


def label_statistics(
    labeled: pl.DataFrame, spec: BreakoutSeqSpec = SPEC_DEFAULT,
) -> dict:
    """Diagnostic stats for sanity-checking the cohort before training."""
    label_col = spec.label_column()
    if label_col not in labeled.columns:
        raise ValueError(
            f"call compute_breakout_seq_label first; missing {label_col}"
        )
    non_null = labeled.filter(pl.col(label_col).is_not_null())
    if non_null.height == 0:
        return {
            "spec": label_col,
            "n_labelable_rows": 0,
            "n_winners": 0,
            "winner_rate": 0.0,
        }
    winners = non_null.filter(pl.col(label_col) == True)
    return {
        "spec": label_col,
        "touch_threshold_pct": spec.touch_threshold_pct,
        "horizon_days": spec.horizon_days,
        "min_entry_price_usd": spec.min_entry_price_usd,
        "n_labelable_rows": int(non_null.height),
        "n_winners": int(winners.height),
        "winner_rate": float(winners.height / non_null.height),
        "mean_forward_max_pct_all": float(non_null["forward_max_pct_60td"].mean()),
        "median_forward_max_pct_all": float(non_null["forward_max_pct_60td"].median()),
        "mean_forward_max_pct_winners": float(
            winners["forward_max_pct_60td"].mean() if winners.height else 0.0
        ),
        "median_forward_max_pct_winners": float(
            winners["forward_max_pct_60td"].median() if winners.height else 0.0
        ),
    }
