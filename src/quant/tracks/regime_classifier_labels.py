"""Regime classifier — rule-based labels for supervised training.

Per PR #1 issuecomment-4475073599 platform-team spec. Defines 8 market
regimes via deterministic rules on the 14 macro features. These labels
serve as the *initial training labels* for the XGBoost classifier;
human review of confidently-misclassified days refines the rules
iteratively.

Regimes:
  1. steady-bull              — uptrend + low vol + risk-on
  2. choppy-recovery          — uptrend resumed after drawdown, elevated vol
  3. crypto-decoupled-bull    — BTC ripping while equities flat/down
  4. low-vol-grind            — quiet uptrend (or sideways), low ATR
  5. bear-trend               — equity index breakdown, sustained underperf
  6. crash-shock              — VIX > 35 + credit spreads blown
  7. high-correlation-risk-off — everything down, correlations spike
  8. sideways-range           — SMA-stack converged, no trend, mid-vol

Each row gets a single label (priority order in REGIME_PRIORITY) OR
'unlabeled' if no rule matches. Unlabeled rows can be left out of
training or hand-labeled later.

Status: Day 1 scaffold per PR #1 issuecomment-4475073599. Rules are
deliberately conservative to produce high-confidence training labels;
expected coverage is 50-70% of trading days, with the residual passed
through to manual review or unsupervised clustering.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


# Priority order — first matching rule wins per row. This ordering
# encodes which regime "dominates" when multiple rules would match.
# Crash-shock and high-correlation-risk-off are highest priority because
# they are the most economically distinctive states; ordinary up/down
# trends are lower priority.
REGIME_PRIORITY = [
    "crash_shock",                  # most extreme — checked first
    "high_correlation_risk_off",
    "bear_trend",
    "crypto_decoupled_bull",
    "choppy_recovery",
    "steady_bull",
    "low_vol_grind",
    "sideways_range",
    # fallthrough: "unlabeled"
]


@dataclass(frozen=True)
class LabelThresholds:
    """Tunable thresholds for rule-based labels.

    Defaults align with PR #1 issuecomment-4475073599 spec. Adjust if
    label coverage falls outside 50-70% target band or if XGBoost
    confusion matrix shows specific regimes systematically over/under
    represented.
    """
    # crash-shock
    crash_vix_floor: float = 35.0
    crash_credit_spread_sigma: float = 2.0  # ≥2σ above trailing 90d

    # high-correlation risk-off
    risk_off_corr_floor: float = 0.70
    # "all-asset returns negative" — proxied by BTC < 0 AND SPX < 0 daily

    # bear-trend
    bear_spx_6mo_return_ceiling: float = -0.10
    bear_vix_floor: float = 22.0

    # crypto-decoupled-bull
    decoupled_btc_30d_return_floor: float = 0.30
    decoupled_btc_equity_corr_ceiling: float = 0.40

    # choppy-recovery
    choppy_drawdown_threshold: float = -0.20
    choppy_btc_atr_floor: float = 2.5

    # steady-bull
    steady_vix_ceiling: float = 18.0
    steady_btc_atr_ceiling: float = 2.0

    # low-vol-grind
    lowvol_btc_atr_ceiling: float = 1.5
    lowvol_vix_ceiling: float = 14.0
    lowvol_abs_btc_return_ceiling: float = 0.10

    # sideways-range
    sideways_sma_spread_ceiling: float = 0.05   # SMAs within 5%
    sideways_vix_floor: float = 15.0
    sideways_vix_ceiling: float = 22.0


THRESHOLDS_DEFAULT = LabelThresholds()


def assign_regime_labels(
    features: pl.DataFrame, thresholds: LabelThresholds = THRESHOLDS_DEFAULT,
) -> pl.DataFrame:
    """Apply rule-based regime labels to the daily feature matrix.

    Input: features dataframe from compute_regime_features() with the
    14 feature columns. Must include `date`.
    Output: same dataframe with added column `regime_label_rule`
    (categorical) and `regime_label_rule_was_unlabeled` (Boolean).
    Missing-data rows propagate as `unlabeled`.
    """
    t = thresholds

    # Pre-compute helper indicators
    df = features.with_columns([
        # SMA50/200 above (uptrend) per asset
        (pl.col("btc_sma50_200_position") > 0).alias("_btc_uptrend"),
        (pl.col("spx_sma50_200_position") > 0).alias("_spx_uptrend"),
        # Credit spread 2σ-above-trailing-90d (z-score-style)
        (
            (pl.col("spx_credit_spread_proxy") -
             pl.col("spx_credit_spread_proxy").rolling_mean(window_size=90))
            / pl.col("spx_credit_spread_proxy").rolling_std(window_size=90)
        ).alias("_credit_spread_z90"),
    ])

    # Build per-rule masks. Order matters — apply in REGIME_PRIORITY.
    df = df.with_columns([
        # 1. crash_shock
        (
            (pl.col("spx_vix_level") > t.crash_vix_floor) &
            (pl.col("_credit_spread_z90") > t.crash_credit_spread_sigma)
        ).alias("_m_crash_shock"),

        # 2. high_correlation_risk_off
        (
            (pl.col("crypto_equity_30d_corr") > t.risk_off_corr_floor) &
            (pl.col("btc_30d_return") < 0) &
            (pl.col("spx_6mo_return") < 0)
        ).alias("_m_high_correlation_risk_off"),

        # 3. bear_trend
        (
            ~pl.col("_spx_uptrend") &
            (pl.col("spx_6mo_return") < t.bear_spx_6mo_return_ceiling) &
            (pl.col("spx_vix_level") > t.bear_vix_floor)
        ).alias("_m_bear_trend"),

        # 4. crypto_decoupled_bull
        (
            (pl.col("btc_30d_return") > t.decoupled_btc_30d_return_floor) &
            (pl.col("crypto_equity_30d_corr") < t.decoupled_btc_equity_corr_ceiling)
        ).alias("_m_crypto_decoupled_bull"),

        # 5. choppy_recovery
        (
            pl.col("_btc_uptrend") &
            (pl.col("btc_drawdown_from_200d_high") < t.choppy_drawdown_threshold) &
            (pl.col("btc_atr_pct_daily") > t.choppy_btc_atr_floor)
        ).alias("_m_choppy_recovery"),

        # 6. steady_bull
        (
            pl.col("_btc_uptrend") & pl.col("_spx_uptrend") &
            (pl.col("spx_vix_level") < t.steady_vix_ceiling) &
            (pl.col("btc_atr_pct_daily") < t.steady_btc_atr_ceiling)
        ).alias("_m_steady_bull"),

        # 7. low_vol_grind
        (
            (pl.col("btc_atr_pct_daily") < t.lowvol_btc_atr_ceiling) &
            (pl.col("spx_vix_level") < t.lowvol_vix_ceiling) &
            (pl.col("btc_30d_return").abs() < t.lowvol_abs_btc_return_ceiling)
        ).alias("_m_low_vol_grind"),

        # 8. sideways_range
        (
            (pl.col("btc_sma20_50_slope").abs() < 0.001) &  # near-flat 20/50 spread
            (pl.col("spx_vix_level") > t.sideways_vix_floor) &
            (pl.col("spx_vix_level") < t.sideways_vix_ceiling)
        ).alias("_m_sideways_range"),
    ])

    # Apply priority order: first matching rule wins.
    label_expr = pl.lit("unlabeled")
    for regime in reversed(REGIME_PRIORITY):  # apply in reverse so first-priority overrides
        mask_col = f"_m_{regime}"
        label_expr = pl.when(pl.col(mask_col).fill_null(False)).then(pl.lit(regime)).otherwise(label_expr)

    df = df.with_columns([
        label_expr.alias("regime_label_rule"),
    ])
    df = df.with_columns([
        (pl.col("regime_label_rule") == "unlabeled").alias("regime_label_rule_was_unlabeled"),
    ])

    # Drop helper columns to keep output clean
    helper_cols = [c for c in df.columns if c.startswith("_m_") or c.startswith("_")]
    return df.drop(helper_cols)


def label_statistics(labeled: pl.DataFrame) -> dict:
    """Diagnostic stats — coverage rate per regime, total unlabeled, date span."""
    counts = (
        labeled.group_by("regime_label_rule")
        .len()
        .sort("len", descending=True)
    )
    total = labeled.height
    coverage = 1.0 - (labeled.filter(pl.col("regime_label_rule") == "unlabeled").height / total)

    return {
        "n_total_days": total,
        "n_labeled_days": int(coverage * total),
        "coverage_rate": float(coverage),
        "per_regime_counts": {
            row["regime_label_rule"]: int(row["len"])
            for row in counts.iter_rows(named=True)
        },
        "date_range": (
            str(labeled["date"].min()) if "date" in labeled.columns else None,
            str(labeled["date"].max()) if "date" in labeled.columns else None,
        ),
    }
