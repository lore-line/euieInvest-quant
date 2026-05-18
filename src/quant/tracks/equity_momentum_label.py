"""Equity momentum cohort label — Stream 2a/2b v1.

Per PR #1 issuecomment-4473357174 + 4473364135 (Stream 2 trifecta scope).
Defines two parallel momentum variants for walkforward validation:

  fast (target Stream 2a, 30-60d hold):
    - Entry: 60d Donchian high breakout (close > max(close[-60:-1]))
    - Look-ahead label: max(close[entry+1..entry+60]) >= entry × (1+target)
    - Target: +15% (calibrated to 60d realistic momentum continuation)
    - Min price: $10, min ADV-dollar: $10M

  slow (target Stream 2b, 90-180d hold):
    - Entry: 252d Donchian high breakout (close > max(close[-252:-1]))
    - Look-ahead label: max(close[entry+1..entry+180]) >= entry × (1+target)
    - Target: +25% (calibrated to 180d realistic momentum continuation
      and "TFSA-defensible no-frequent-trading" hold)
    - Min price: $10, min ADV-dollar: $10M

Both variants are universe-filtered at LABEL TIME (lesson learned from
Phase B v3 — post-hoc filter ≠ training-cohort filter; rules trained on
liquid universe outperform rules trained on broad universe + filtered).

Russell-1000-ish universe filter:
  - min_price ≥ $10 at entry
  - min_avg_volume_30d_dollar ≥ $10M
  - (TODO: add Russell-1000-membership filter when membership data lands;
    for v1 use the liquidity proxy)

Output schema (added to features dataframe):
  - `forward_max_pct_{horizon}td` (e.g. forward_max_pct_60td, _180td)
  - `is_equity_momentum_{name}` (e.g. is_equity_momentum_fast_g15, _slow_g25)
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


PIPELINE_STEP = "equity_momentum_v1"
PATTERN_PREFIX_FAST = "em_fast"   # equity_momentum_fast → Stream 2a
PATTERN_PREFIX_SLOW = "em_slow"   # equity_momentum_slow → Stream 2b


@dataclass(frozen=True)
class EquityMomentumSpec:
    """Spec for one equity-momentum variant.

    Defaults align with PR #1 issuecomment-4473364135:
      fast: 60d entry breakout, +15% target over 60d hold
      slow: 252d entry breakout, +25% target over 180d hold

    Per PR #1 issuecomment-4473629237 (server-team edge finding), the
    `vol_confirm_mult` field adds a volume-confirmation filter at entry
    day: entry volume must be ≥ vol_confirm_mult × trailing-30d-avg.
    Set to 0 (default) to disable for backwards-compat baseline runs.
    """

    name: str                       # "fast_g15" or "slow_g25"
    entry_breakout_days: int        # Donchian-high lookback for entry signal
    horizon_trading_days: int       # forward window for label evaluation
    target_pct: float               # gain threshold for is_winner label (decimal: 0.15 = 15%)
    min_entry_price_usd: float = 10.0
    min_avg_volume_30d_dollar: float = 10_000_000.0  # $10M ADV
    vol_confirm_mult: float = 0.0    # 0 = disabled; 1.5 = entry volume ≥ 1.5× trailing-30d-avg

    def label_column(self) -> str:
        return f"is_equity_momentum_{self.name}"

    def forward_max_column(self) -> str:
        return f"forward_max_pct_{self.horizon_trading_days}td"

    def pattern_prefix(self) -> str:
        return PATTERN_PREFIX_FAST if "fast" in self.name else PATTERN_PREFIX_SLOW

    def pattern(self, rule_id: int | str) -> str:
        return f"{self.pattern_prefix()}_{self.name}_rule_{rule_id}"


# Canonical specs for the two variants
SPEC_FAST = EquityMomentumSpec(
    name="fast_g15", entry_breakout_days=60, horizon_trading_days=60, target_pct=0.15,
)
SPEC_SLOW = EquityMomentumSpec(
    name="slow_g25", entry_breakout_days=252, horizon_trading_days=180, target_pct=0.25,
)
SPECS: dict[str, EquityMomentumSpec] = {"fast": SPEC_FAST, "slow": SPEC_SLOW}


def compute_equity_momentum_label(
    features: pl.DataFrame, spec: EquityMomentumSpec,
) -> pl.DataFrame:
    """Add the equity-momentum cohort label + forward-max column to features.

    Requires `symbol`, `date`, `close_adj`, `volume` columns. Computes
    rolling-30d avg dollar volume internally for the liquidity filter.

    Label is TRUE iff:
      - close_adj[entry] >= spec.min_entry_price_usd
      - avg_volume_30d_dollar[entry] >= spec.min_avg_volume_30d_dollar
      - close_adj[entry] > max(close_adj[entry-N..entry-1]) for N = entry_breakout_days
        (Donchian high breakout — entry day's close is the new N-day high)
      - max(close_adj[entry+1..entry+H]) >= close_adj[entry] × (1 + target_pct)
        for H = horizon_trading_days

    Rows where the entry filter or forward window can't be evaluated get
    null on the label column.
    """
    if "close_adj" not in features.columns or "volume" not in features.columns:
        raise ValueError("features must contain close_adj and volume")

    N = spec.entry_breakout_days
    H = spec.horizon_trading_days
    target_mult = 1.0 + spec.target_pct

    df = features.sort(["symbol", "date"])

    # Step 1: rolling-30d avg dollar volume per symbol (avg over 30 trading days
    # of volume × close_adj). $10M ADV is a typical liquid-mid-cap floor.
    df = df.with_columns(
        avg_dollar_volume_30d=(
            (pl.col("volume") * pl.col("close_adj"))
            .rolling_mean(window_size=30)
            .over("symbol")
        ),
    )

    # Step 2: Donchian high lookback — max(close_adj over the prior N days,
    # NOT including today). Use shift(1) to exclude today, then rolling_max
    # over N days.
    df = df.with_columns(
        prior_n_day_high=(
            pl.col("close_adj")
            .shift(1)
            .rolling_max(window_size=N)
            .over("symbol")
        ),
    )

    # Step 3: forward-max over H days (max of close_adj[t+1..t+H]).
    # Use the same shift-then-max idiom as sustained_winner_label.
    shift_cols = [
        pl.col("close_adj").shift(-i).over("symbol").alias(f"_fwd_close_{i}")
        for i in range(1, H + 1)
    ]
    df = df.with_columns(shift_cols)

    forward_max = pl.max_horizontal(
        [pl.col(f"_fwd_close_{i}") for i in range(1, H + 1)]
    )
    # Window-complete check: last day of forward window must be non-null
    window_complete = pl.col(f"_fwd_close_{H}").is_not_null()

    fwd_col = spec.forward_max_column()
    # Also capture exit-at-horizon close BEFORE we drop the bookkeeping shift cols
    # (shift in post-filter dataframe doesn't work — it operates on the sparse
    # per-symbol filtered series, jumping over many calendar days).
    df = df.with_columns(
        **{
            fwd_col: pl.when(window_complete)
            .then((forward_max / pl.col("close_adj") - 1.0) * 100.0)
            .otherwise(None),
            f"exit_close_adj_{H}td": pl.col(f"_fwd_close_{H}"),
            f"exit_date_{H}td": pl.col("date").shift(-H).over("symbol"),
        }
    )
    df = df.drop([f"_fwd_close_{i}" for i in range(1, H + 1)])

    # Step 4: composite label
    # Volume confirmation: per server-team finding (vol-confirm = the unlock).
    # Compute trailing-30d avg volume; require entry volume ≥ vol_confirm_mult ×
    # that average. When vol_confirm_mult=0 the filter is disabled.
    df = df.with_columns(
        avg_volume_30d_shares=(
            pl.col("volume").rolling_mean(window_size=30).over("symbol")
        ),
    )
    vol_confirm_condition = (
        pl.lit(True) if spec.vol_confirm_mult <= 0.0 else
        (pl.col("volume") >= spec.vol_confirm_mult * pl.col("avg_volume_30d_shares"))
    )

    label_col = spec.label_column()
    df = df.with_columns(
        **{
            label_col: pl.when(
                (pl.col("close_adj") >= spec.min_entry_price_usd)
                & (pl.col("avg_dollar_volume_30d") >= spec.min_avg_volume_30d_dollar)
                & (pl.col("close_adj") > pl.col("prior_n_day_high"))
                & vol_confirm_condition
                & (pl.col(fwd_col).is_not_null())
            )
            .then(pl.col(fwd_col) >= spec.target_pct * 100.0)
            .otherwise(None)
        }
    )

    return df


def label_statistics(
    labeled: pl.DataFrame, spec: EquityMomentumSpec,
) -> dict:
    """Diagnostic stats for cohort sanity-check."""
    label_col = spec.label_column()
    fwd_col = spec.forward_max_column()
    if label_col not in labeled.columns:
        raise ValueError(f"call compute_equity_momentum_label first; missing {label_col}")

    non_null = labeled.filter(pl.col(label_col).is_not_null())
    if non_null.height == 0:
        return {
            "spec": label_col,
            "n_labelable_rows": 0,
            "n_winners": 0,
            "winner_rate": 0.0,
        }

    winners = non_null.filter(pl.col(label_col) == True)
    n_entry_signals = labeled.filter(
        (pl.col("close_adj") >= spec.min_entry_price_usd)
        & (pl.col("avg_dollar_volume_30d") >= spec.min_avg_volume_30d_dollar)
        & (pl.col("close_adj") > pl.col("prior_n_day_high"))
    ).height

    return {
        "spec": label_col,
        "entry_breakout_days": spec.entry_breakout_days,
        "horizon_trading_days": spec.horizon_trading_days,
        "target_pct": spec.target_pct,
        "min_entry_price_usd": spec.min_entry_price_usd,
        "min_avg_volume_30d_dollar": spec.min_avg_volume_30d_dollar,
        "n_entry_signals_universe": int(n_entry_signals),
        "n_labelable_rows": int(non_null.height),
        "n_winners": int(winners.height),
        "winner_rate": float(winners.height / non_null.height),
        "mean_forward_max_pct_all": float(non_null[fwd_col].mean()),
        "median_forward_max_pct_all": float(non_null[fwd_col].median()),
        "mean_forward_max_pct_winners": float(
            winners[fwd_col].mean() if winners.height else 0.0
        ),
        "median_forward_max_pct_winners": float(
            winners[fwd_col].median() if winners.height else 0.0
        ),
    }
