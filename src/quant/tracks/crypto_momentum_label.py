"""Crypto momentum cohort label — Stream 2c v1.

Per PR #1 issuecomment-4473364135 + 4473464239. Donchian-breakout label
on a fixed 8-symbol crypto universe (per consumer-side recommendation):

    BTC-USD, ETH-USD, SOL-USD, ADA-USD, AVAX-USD, DOT-USD, LINK-USD, ATOM-USD

Differs from equity_momentum_label in three ways:
  1. Fixed universe (no liquidity filter — all 8 are top-cap, deeply liquid
     on Kraken Pro with clean 5y price history)
  2. Shorter breakout window (14d default — crypto regime shifts faster)
  3. Tighter horizon (60d max — crypto moves are faster + capital turnover
     is more important when trading mechanics are cleaner)

No survivorship haircut (single-asset class, no universe selection bias).

Output schema:
  - `forward_max_pct_60td`
  - `is_crypto_momentum_g30` — entry breakout AND price ≥ +30% within 60d
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


PIPELINE_STEP = "crypto_momentum_v1"
PATTERN_PREFIX = "cm_v1"  # crypto_momentum_v1

# Consumer-recommended universe (per issuecomment-4473364135).
# 8 top-cap symbols with clean Kraken Pro USD pairs + multi-year continuous
# price history. Excludes XRP (regulatory), BNB (Kraken thin liquidity),
# MATIC/POL (rebrand continuity broken).
CRYPTO_UNIVERSE: list[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD",
]


@dataclass(frozen=True)
class CryptoMomentumSpec:
    """Spec for crypto momentum cohort label.

    Defaults align with PR #1 issuecomment-4473464239:
      14d Donchian high entry breakout, 60d horizon, +30% target.
    """

    name: str = "g30"
    entry_breakout_days: int = 14
    horizon_trading_days: int = 60
    target_pct: float = 0.30   # higher than equity (15-25%) — crypto's hit-rate-at-bigger-target

    def label_column(self) -> str:
        return f"is_crypto_momentum_{self.name}"

    def forward_max_column(self) -> str:
        return f"forward_max_pct_{self.horizon_trading_days}td"

    def pattern(self, rule_id: int | str) -> str:
        return f"{PATTERN_PREFIX}_{self.name}_rule_{rule_id}"


SPEC_DEFAULT = CryptoMomentumSpec()


def compute_crypto_momentum_label(
    ohlcv: pl.DataFrame, spec: CryptoMomentumSpec = SPEC_DEFAULT,
) -> pl.DataFrame:
    """Add the crypto-momentum cohort label + forward-max column.

    Input ohlcv: per-(symbol, date) OHLCV. Must have `symbol`, `date`,
    `close` (close_adj also OK since crypto doesn't split). The function
    uses `close` (or `close_adj` if present) for breakout + horizon calc.
    """
    # Use close_adj if available AND populated, else close. Crypto symbols
    # from the sidecar OHLCV endpoint have a close_adj column for schema
    # compatibility but it's NULL (no splits/dividends).
    if "close_adj" in ohlcv.columns and ohlcv["close_adj"].drop_nulls().len() > 0:
        price_col = "close_adj"
    elif "close" in ohlcv.columns:
        price_col = "close"
    else:
        raise ValueError("ohlcv must contain close or close_adj")

    N = spec.entry_breakout_days
    H = spec.horizon_trading_days

    df = ohlcv.sort(["symbol", "date"])

    # Donchian high — max(close over prior N days, excluding today)
    df = df.with_columns(
        prior_n_day_high=(
            pl.col(price_col).shift(1).rolling_max(window_size=N).over("symbol")
        ),
    )

    # Forward shifts for horizon return
    shift_cols = [
        pl.col(price_col).shift(-i).over("symbol").alias(f"_fwd_close_{i}")
        for i in range(1, H + 1)
    ]
    df = df.with_columns(shift_cols)

    forward_max = pl.max_horizontal(
        [pl.col(f"_fwd_close_{i}") for i in range(1, H + 1)]
    )
    window_complete = pl.col(f"_fwd_close_{H}").is_not_null()

    fwd_col = spec.forward_max_column()
    df = df.with_columns(
        **{
            fwd_col: pl.when(window_complete)
            .then((forward_max / pl.col(price_col) - 1.0) * 100.0)
            .otherwise(None),
            f"exit_close_{H}td": pl.col(f"_fwd_close_{H}"),
            f"exit_date_{H}td": pl.col("date").shift(-H).over("symbol"),
        }
    )
    df = df.drop([f"_fwd_close_{i}" for i in range(1, H + 1)])

    label_col = spec.label_column()
    df = df.with_columns(
        **{
            label_col: pl.when(
                (pl.col(price_col) > pl.col("prior_n_day_high"))
                & pl.col(fwd_col).is_not_null()
            )
            .then(pl.col(fwd_col) >= spec.target_pct * 100.0)
            .otherwise(None)
        }
    )
    return df


def label_statistics(
    labeled: pl.DataFrame, spec: CryptoMomentumSpec = SPEC_DEFAULT,
) -> dict:
    label_col = spec.label_column()
    fwd_col = spec.forward_max_column()
    non_null = labeled.filter(pl.col(label_col).is_not_null())
    if non_null.height == 0:
        return {"spec": label_col, "n_labelable_rows": 0, "n_winners": 0, "winner_rate": 0.0}
    winners = non_null.filter(pl.col(label_col) == True)
    return {
        "spec": label_col,
        "entry_breakout_days": spec.entry_breakout_days,
        "horizon_trading_days": spec.horizon_trading_days,
        "target_pct": spec.target_pct,
        "n_labelable_rows": int(non_null.height),
        "n_winners": int(winners.height),
        "winner_rate": float(winners.height / non_null.height),
        "mean_forward_max_pct_all": float(non_null[fwd_col].mean()),
        "median_forward_max_pct_all": float(non_null[fwd_col].median()),
        "per_symbol_winner_counts": {
            row["symbol"]: int(row["n"])
            for row in winners.group_by("symbol").len().rename({"len": "n"}).iter_rows(named=True)
        },
    }
