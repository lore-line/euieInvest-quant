"""Sustained-winner cohort label — Workstream C of the quant signal contract.

User-defined cohort (PR #1 issuecomment-4467XXX, server-team coordination
response 2026-05-17 03:08 UTC):

  "Find every symbol that grew ≥20% over a 30d period that is over $1
   share price, and a common signal between them."

Translated into a precise label:

  For each (symbol, entry_date), label as `is_sustained_winner=True` if BOTH:
  - close_adj[entry_date] >= 1.00   (penny-stock exclusion)
  - max(close_adj[entry_date+1 .. entry_date+30]) >= entry_price * 1.20
  - close_adj[entry_date+30] >= entry_price * 1.10  ← the load-bearing
    constraint that distinguishes sustained winners from flash-and-fade

This REPLACES Phase B v3's transient-touch winner label, which was
contaminated by names that touched +20% intra-window then reverted by
day 30. The reversion cohort is what poisoned median realized gain in
the Stage 2 EXIT signal joint validation — see [exit_variant_sweep
README §The binding-constraint finding](../../../euieInvest-reports/runs/2026-05-16-exit_variant_sweep/README.md).

A stricter variant (`is_sustained_winner_strict=True`) requires the
day-30 endpoint to also clear +20% (full sustained 20% at day 30, not
just 10%). Reported alongside the standard variant for comparison.

This module is pure label computation. The downstream supervised
discovery (XGB rule extraction + walk-forward validation) lives in
sibling modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SustainedWinnerSpec:
    """Spec for the cohort label.

    Two variants intended for side-by-side comparison:

      standard (the user's spec):
        touch_threshold_pct = 20%, endpoint_threshold_pct = 10%
      strict:
        touch_threshold_pct = 20%, endpoint_threshold_pct = 20%
    """

    name: str  # "standard" or "strict"
    touch_threshold_pct: float  # max-forward-return must exceed this
    endpoint_threshold_pct: float  # day-30 close must exceed this above entry
    horizon_days: int = 30
    min_entry_price_usd: float = 1.00

    def label_column(self) -> str:
        return f"is_sustained_winner_{self.name}"


# The two canonical specs from the original direction (kept for back-compat
# in tests; the production discovery sweeps via `sweep_specs` below)
SPECS: dict[str, SustainedWinnerSpec] = {
    "standard": SustainedWinnerSpec("standard", 20.0, 10.0),
    "strict": SustainedWinnerSpec("strict", 20.0, 20.0),
}


def sweep_specs(
    g_max_pct: float = 20.0,
    g_min_pct: float = 1.0,
    step_pct: float = 1.0,
    endpoint_ratio: float = 0.5,
    horizon_days: int = 30,
    min_entry_price_usd: float = 1.00,
) -> list[SustainedWinnerSpec]:
    """Generate a list of specs for the Pareto-frontier gain sweep
    (PR #1 issuecomment 2026-05-17 03:18 + 03:20 server-team refinements).

    For each g% in [g_min_pct, g_max_pct] (stepped by step_pct, DESCENDING
    so the discovery can stop at the highest g that clears gates):

      spec = SustainedWinnerSpec(
        name = f"g{g_pct:02d}",
        touch_threshold_pct = g_pct,
        endpoint_threshold_pct = g_pct * endpoint_ratio,  # default g/2
      )

    Default range 20% → 1% with 1% step → 20 specs. Hard floor at 1%
    matches server-team's safety bound (1% over 30d ≈ +9% annualized,
    basically index-equivalent — sweep stops here at the latest, but
    typically EV-positivity or label-coverage stops it sooner).
    """
    if endpoint_ratio < 0 or endpoint_ratio > 1:
        raise ValueError(f"endpoint_ratio must be in [0, 1], got {endpoint_ratio}")
    specs: list[SustainedWinnerSpec] = []
    g_pct = g_max_pct
    while g_pct >= g_min_pct - 1e-9:
        # Format the name with the gain percent as a 2-digit integer
        # e.g. g_pct=20.0 → "g20", g_pct=18.0 → "g18", g_pct=1.0 → "g01"
        # This matches the platform-side pattern naming sw1_g{NN}_{rule_id}.
        name = f"g{int(round(g_pct)):02d}"
        specs.append(SustainedWinnerSpec(
            name=name,
            touch_threshold_pct=g_pct,
            endpoint_threshold_pct=g_pct * endpoint_ratio,
            horizon_days=horizon_days,
            min_entry_price_usd=min_entry_price_usd,
        ))
        g_pct -= step_pct
    return specs


def compute_sustained_winner_label(
    features: pl.DataFrame, spec: SustainedWinnerSpec
) -> pl.DataFrame:
    """Add the sustained-winner label column to features per the given spec.

    The features frame must have `symbol`, `date`, and `close_adj` columns
    and must be sorted/sortable per-symbol by date.

    Returns features with two new columns:
      - `forward_max_pct` — max forward-N-day return (percent)
      - `forward_endpoint_pct` — return at exactly day N (percent)
      - `<spec.label_column()>` — boolean (or null if forward window
        unavailable / fails price floor)

    Rows where the forward window extends past the data are left null on
    the label column. Caller MUST handle null (filter or impute) when
    training.
    """
    if "close_adj" not in features.columns:
        raise ValueError("features must contain close_adj")
    n = spec.horizon_days
    # Pre-sort once per call (callers can pre-sort to avoid this if needed)
    sorted_features = features.sort(["symbol", "date"])
    # Build a rolling max over the forward N rows per symbol via shift + reverse.
    # The simplest correct approach: shift the close_adj backward by 1..N
    # rows per symbol, take row-wise max, then compare to entry price.
    #
    # Why row-wise: polars rolling_max with a window CENTERED forward is
    # awkward; the shift-then-max idiom is clearer and runs in O(N) per
    # column.
    out = sorted_features
    shift_cols: list[pl.Expr] = []
    for i in range(1, n + 1):
        shift_cols.append(
            pl.col("close_adj").shift(-i).over("symbol").alias(f"_fwd_close_{i}")
        )
    out = out.with_columns(shift_cols)
    # Row-wise max across the forward window
    forward_close_max = pl.max_horizontal(
        [pl.col(f"_fwd_close_{i}") for i in range(1, n + 1)]
    )
    # Forward-N-day endpoint
    forward_close_endpoint = pl.col(f"_fwd_close_{n}")
    out = out.with_columns(
        forward_max_pct=((forward_close_max / pl.col("close_adj") - 1.0) * 100.0),
        forward_endpoint_pct=(
            (forward_close_endpoint / pl.col("close_adj") - 1.0) * 100.0
        ),
    )
    # Drop bookkeeping columns
    out = out.drop([f"_fwd_close_{i}" for i in range(1, n + 1)])
    # Compute label
    label_col = spec.label_column()
    out = out.with_columns(
        **{
            label_col: pl.when(
                (pl.col("close_adj") >= spec.min_entry_price_usd)
                & (pl.col("forward_max_pct").is_not_null())
                & (pl.col("forward_endpoint_pct").is_not_null())
            )
            .then(
                (pl.col("forward_max_pct") >= spec.touch_threshold_pct)
                & (pl.col("forward_endpoint_pct") >= spec.endpoint_threshold_pct)
            )
            .otherwise(None)
        }
    )
    return out


def label_statistics(
    labeled: pl.DataFrame, spec: SustainedWinnerSpec
) -> dict:
    """Diagnostic stats for sanity-checking the cohort before training."""
    label_col = spec.label_column()
    if label_col not in labeled.columns:
        raise ValueError(f"call compute_sustained_winner_label first; missing {label_col}")
    non_null = labeled.filter(pl.col(label_col).is_not_null())
    if non_null.height == 0:
        return {
            "spec": spec.label_column(),
            "n_labelable_rows": 0,
            "n_winners": 0,
            "winner_rate": 0.0,
        }
    winners = non_null.filter(pl.col(label_col) == True)
    return {
        "spec": spec.label_column(),
        "touch_threshold_pct": spec.touch_threshold_pct,
        "endpoint_threshold_pct": spec.endpoint_threshold_pct,
        "horizon_days": spec.horizon_days,
        "min_entry_price_usd": spec.min_entry_price_usd,
        "n_labelable_rows": int(non_null.height),
        "n_winners": int(winners.height),
        "winner_rate": float(winners.height / non_null.height),
        "mean_forward_max_pct_all": float(non_null["forward_max_pct"].mean()),
        "median_forward_max_pct_all": float(non_null["forward_max_pct"].median()),
        "mean_forward_endpoint_pct_all": float(non_null["forward_endpoint_pct"].mean()),
        "median_forward_endpoint_pct_all": float(non_null["forward_endpoint_pct"].median()),
        "mean_forward_max_pct_winners": float(
            winners["forward_max_pct"].mean() if winners.height else 0.0
        ),
        "median_forward_max_pct_winners": float(
            winners["forward_max_pct"].median() if winners.height else 0.0
        ),
        "mean_forward_endpoint_pct_winners": float(
            winners["forward_endpoint_pct"].mean() if winners.height else 0.0
        ),
        "median_forward_endpoint_pct_winners": float(
            winners["forward_endpoint_pct"].median() if winners.height else 0.0
        ),
    }
