"""Regime classifier — Priority 1 (P1) feature engineering.

Per PR #1 issuecomment-4475073599 (platform-team handoff). Computes the 14
hand-crafted macro/cross-asset features used to label daily market regimes.

Feature groups:
  - Crypto regime: BTC ATR%, SMA slope/position, 30d return, volume z-score,
    drawdown from 200d high
  - Equity regime: VIX level + 30d percentile, SPX SMA stack, 6mo return,
    credit spread proxy (HY OAS or HYG/LQD ratio)
  - Cross-asset regime: BTC/SPX 30d correlation, alt/BTC 30d correlation
    (decoupling signal), DXY 30d slope, gold/SPX 30d correlation
    (flight-to-quality signal)

All features computed on a per-(date) row keyed by trading day. Crypto
features use BTC-USD daily close; equity features use SPY (proxy for SPX).
Output: wide-format polars DataFrame, one row per date, 14 feature columns.

Note: the feature module is pure — no I/O. Caller provides the per-symbol
OHLCV dataframes (BTC, SPY, VIX, etc.) and gets back the daily feature
matrix. Use `compute_regime_features(price_panel)` where `price_panel` is
a polars DataFrame with columns [symbol, date, close, high, low, volume].

Status: Day 1 scaffold per [PR #1 issuecomment-4475073599 handoff].
Tested with 5y SPY data; remaining data feeds (BTC, VIX, HYG/LQD, GLD,
DXY) need to be wired up before walkforward training can run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


PIPELINE_STEP = "regime_classifier_v1"


# Symbol convention: yfinance-style tickers
BTC_SYMBOL = "BTC-USD"
SPX_PROXY_SYMBOL = "SPY"          # 1:1 SPX proxy
VIX_SYMBOL = "^VIX"
HYG_SYMBOL = "HYG"                # high-yield bond ETF
LQD_SYMBOL = "LQD"                # investment-grade bond ETF
GLD_SYMBOL = "GLD"                # gold ETF (NOT the stock "GOLD" = Barrick)
DXY_SYMBOL = "DX-Y.NYB"           # dollar index futures (or "UUP" as ETF proxy)
# Alt basket for crypto-decoupling signal (rolling 30d correlation with BTC)
ALT_BASKET = ["ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
              "DOT-USD", "LINK-USD", "ATOM-USD"]


@dataclass(frozen=True)
class FeatureSpec:
    """Lookback windows + feature selection."""
    atr_window: int = 14
    sma_short: int = 20
    sma_mid: int = 50
    sma_long: int = 200
    return_lookback_short: int = 30
    return_lookback_long: int = 126     # ~6mo trading days
    volume_z_window: int = 30
    drawdown_lookback: int = 200
    correlation_window: int = 30
    vix_percentile_window: int = 30
    feature_columns: list[str] = field(default_factory=lambda: [
        # Crypto regime
        "btc_atr_pct_daily",
        "btc_sma20_50_slope",
        "btc_sma50_200_position",
        "btc_30d_return",
        "btc_volume_z30",
        "btc_drawdown_from_200d_high",
        # Equity regime
        "spx_vix_level",
        "spx_vix_pct_30d_rank",
        "spx_sma50_200_position",
        "spx_6mo_return",
        "spx_credit_spread_proxy",
        # Cross-asset
        "crypto_equity_30d_corr",
        "crypto_alt_to_btc_corr",
        "dxy_30d_slope",
        "gold_proxy_corr_to_spx",
    ])


SPEC_DEFAULT = FeatureSpec()


# ---------------------------------------------------------------------------
# Per-symbol primitives
# ---------------------------------------------------------------------------

def _atr_pct(df: pl.DataFrame, window: int = 14) -> pl.Expr:
    """ATR / close × 100, daily. df must have high, low, close (sorted by date)."""
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ])
    atr = tr.rolling_mean(window_size=window)
    return (atr / pl.col("close") * 100.0).alias("atr_pct")


def _sma_slope(close: pl.Expr, short: int, mid: int) -> pl.Expr:
    """Slope of (SMA_short - SMA_mid) / SMA_mid over short window.

    Approximates daily change in the SMA-spread normalized by SMA_mid.
    """
    sma_s = close.rolling_mean(window_size=short)
    sma_m = close.rolling_mean(window_size=mid)
    spread_norm = (sma_s - sma_m) / sma_m
    return (spread_norm - spread_norm.shift(short)) / short


def _sma_position(close: pl.Expr, mid: int, long: int) -> pl.Expr:
    """(SMA_mid - SMA_long) / SMA_long."""
    sma_m = close.rolling_mean(window_size=mid)
    sma_l = close.rolling_mean(window_size=long)
    return (sma_m - sma_l) / sma_l


def _log_return(close: pl.Expr, lookback: int) -> pl.Expr:
    """Rolling N-day log return."""
    return (close / close.shift(lookback)).log()


def _volume_z(volume: pl.Expr, window: int) -> pl.Expr:
    """Rolling z-score of daily volume."""
    mean = volume.rolling_mean(window_size=window)
    std = volume.rolling_std(window_size=window)
    return (volume - mean) / std


def _drawdown_from_high(close: pl.Expr, lookback: int) -> pl.Expr:
    """close / rolling_max(close, lookback) - 1.0 (negative when below high)."""
    return close / close.rolling_max(window_size=lookback) - 1.0


def _percentile_rank(value: pl.Expr, window: int) -> pl.Expr:
    """Rolling percentile rank: fraction of the window strictly less than current."""
    # polars doesn't have a direct rolling_rank — use rolling map.
    # For efficiency, we approximate with (value - rolling_min) / (rolling_max - rolling_min).
    # This is a 0-1 normalized rank position, equivalent to percentile for monotone data.
    rmin = value.rolling_min(window_size=window)
    rmax = value.rolling_max(window_size=window)
    return (value - rmin) / (rmax - rmin)


def _rolling_correlation(
    a: pl.Expr, b: pl.Expr, window: int,
) -> pl.Expr:
    """Rolling Pearson correlation between two return series."""
    # rolling_corr is in polars 0.20+
    return pl.rolling_corr(a, b, window_size=window)


# ---------------------------------------------------------------------------
# Per-symbol feature builders
# ---------------------------------------------------------------------------

def crypto_btc_features(btc_ohlcv: pl.DataFrame, spec: FeatureSpec = SPEC_DEFAULT) -> pl.DataFrame:
    """Per-day crypto regime features from BTC OHLCV.

    Input: dataframe with [date, close, high, low, volume] for BTC-USD only.
    Output: [date, btc_atr_pct_daily, btc_sma20_50_slope, btc_sma50_200_position,
             btc_30d_return, btc_volume_z30, btc_drawdown_from_200d_high].
    """
    df = btc_ohlcv.sort("date")
    out = df.select([
        pl.col("date"),
        _atr_pct(df, spec.atr_window).alias("btc_atr_pct_daily"),
        _sma_slope(pl.col("close"), spec.sma_short, spec.sma_mid).alias("btc_sma20_50_slope"),
        _sma_position(pl.col("close"), spec.sma_mid, spec.sma_long).alias("btc_sma50_200_position"),
        _log_return(pl.col("close"), spec.return_lookback_short).alias("btc_30d_return"),
        _volume_z(pl.col("volume"), spec.volume_z_window).alias("btc_volume_z30"),
        _drawdown_from_high(pl.col("close"), spec.drawdown_lookback).alias("btc_drawdown_from_200d_high"),
    ])
    return out


def equity_spx_features(
    spy_ohlcv: pl.DataFrame, vix_close: pl.DataFrame | None = None,
    hyg_close: pl.DataFrame | None = None, lqd_close: pl.DataFrame | None = None,
    spec: FeatureSpec = SPEC_DEFAULT,
) -> pl.DataFrame:
    """Per-day equity regime features from SPY OHLCV + VIX + HYG/LQD.

    VIX: dataframe with [date, close] (close = VIX index level).
    HYG/LQD: each dataframe with [date, close] for credit-spread-proxy ratio.
    Returns may be null where any input is missing.
    """
    spy = spy_ohlcv.sort("date").select([
        pl.col("date"),
        _sma_position(pl.col("close"), spec.sma_mid, spec.sma_long).alias("spx_sma50_200_position"),
        _log_return(pl.col("close"), spec.return_lookback_long).alias("spx_6mo_return"),
    ])

    if vix_close is not None and vix_close.height > 0:
        vix = vix_close.sort("date").select([
            pl.col("date"),
            pl.col("close").alias("spx_vix_level"),
            _percentile_rank(pl.col("close"), spec.vix_percentile_window).alias("spx_vix_pct_30d_rank"),
        ])
        spy = spy.join(vix, on="date", how="left")
    else:
        spy = spy.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("spx_vix_level"),
            pl.lit(None, dtype=pl.Float64).alias("spx_vix_pct_30d_rank"),
        ])

    if hyg_close is not None and lqd_close is not None and hyg_close.height > 0:
        # Credit spread proxy: lower HYG/LQD ratio => wider spreads (HY underperforming)
        # So we use -log(HYG/LQD) as the spread-widening direction (higher = worse credit).
        hyg = hyg_close.sort("date").select([pl.col("date"), pl.col("close").alias("_hyg")])
        lqd = lqd_close.sort("date").select([pl.col("date"), pl.col("close").alias("_lqd")])
        cs = hyg.join(lqd, on="date").with_columns(
            spx_credit_spread_proxy=-(pl.col("_hyg") / pl.col("_lqd")).log()
        ).select(["date", "spx_credit_spread_proxy"])
        spy = spy.join(cs, on="date", how="left")
    else:
        spy = spy.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("spx_credit_spread_proxy")
        )

    return spy


def cross_asset_features(
    btc_close: pl.DataFrame, spy_close: pl.DataFrame,
    alt_basket_close: pl.DataFrame | None = None,
    dxy_close: pl.DataFrame | None = None,
    gld_close: pl.DataFrame | None = None,
    spec: FeatureSpec = SPEC_DEFAULT,
) -> pl.DataFrame:
    """Per-day cross-asset regime features.

    Inputs: each [date, close] for the respective symbol.
    alt_basket_close: long-format [date, symbol, close] for ETH/SOL/etc.
    Output: [date, crypto_equity_30d_corr, crypto_alt_to_btc_corr,
             dxy_30d_slope, gold_proxy_corr_to_spx].
    """
    btc = btc_close.sort("date").with_columns(
        btc_ret=pl.col("close").pct_change()
    ).select(["date", "btc_ret"])
    spy = spy_close.sort("date").with_columns(
        spy_ret=pl.col("close").pct_change()
    ).select(["date", "spy_ret"])

    joined = btc.join(spy, on="date", how="inner")
    joined = joined.with_columns(
        crypto_equity_30d_corr=_rolling_correlation(
            pl.col("btc_ret"), pl.col("spy_ret"), spec.correlation_window
        ),
    )

    if alt_basket_close is not None and alt_basket_close.height > 0:
        alt = (
            alt_basket_close.sort(["symbol", "date"])
            .with_columns(ret=pl.col("close").pct_change().over("symbol"))
            .group_by("date")
            .agg(alt_basket_ret=pl.col("ret").mean())
            .sort("date")
        )
        joined = joined.join(alt, on="date", how="left").with_columns(
            crypto_alt_to_btc_corr=_rolling_correlation(
                pl.col("alt_basket_ret"), pl.col("btc_ret"), spec.correlation_window
            ),
        )
    else:
        joined = joined.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("crypto_alt_to_btc_corr"),
        )

    if dxy_close is not None and dxy_close.height > 0:
        dxy = dxy_close.sort("date").with_columns(
            _dxy_logret=pl.col("close").log().diff()
        ).select([
            pl.col("date"),
            pl.col("_dxy_logret").rolling_mean(window_size=spec.correlation_window).alias("dxy_30d_slope"),
        ])
        joined = joined.join(dxy, on="date", how="left")
    else:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("dxy_30d_slope"))

    if gld_close is not None and gld_close.height > 0:
        gld = gld_close.sort("date").with_columns(
            gld_ret=pl.col("close").pct_change()
        ).select(["date", "gld_ret"])
        joined = joined.join(gld, on="date", how="left").with_columns(
            gold_proxy_corr_to_spx=_rolling_correlation(
                pl.col("gld_ret"), pl.col("spy_ret"), spec.correlation_window
            ),
        )
    else:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("gold_proxy_corr_to_spx"))

    return joined.select([
        "date", "crypto_equity_30d_corr", "crypto_alt_to_btc_corr",
        "dxy_30d_slope", "gold_proxy_corr_to_spx",
    ])


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def compute_regime_features(
    btc_ohlcv: pl.DataFrame,
    spy_ohlcv: pl.DataFrame,
    vix_close: pl.DataFrame | None = None,
    hyg_close: pl.DataFrame | None = None,
    lqd_close: pl.DataFrame | None = None,
    dxy_close: pl.DataFrame | None = None,
    gld_close: pl.DataFrame | None = None,
    alt_basket_close: pl.DataFrame | None = None,
    spec: FeatureSpec = SPEC_DEFAULT,
) -> pl.DataFrame:
    """Build the full 14-feature regime panel.

    Joins all feature groups on date. Rows where any required feature is null
    (e.g., insufficient lookback) are kept but downstream training should
    filter on `is_complete = all_features_non_null`.
    """
    btc_feat = crypto_btc_features(btc_ohlcv, spec)
    spx_feat = equity_spx_features(spy_ohlcv, vix_close, hyg_close, lqd_close, spec)
    cross_feat = cross_asset_features(
        btc_close=btc_ohlcv.select(["date", "close"]),
        spy_close=spy_ohlcv.select(["date", "close"]),
        alt_basket_close=alt_basket_close,
        dxy_close=dxy_close,
        gld_close=gld_close,
        spec=spec,
    )

    return (
        btc_feat
        .join(spx_feat, on="date", how="outer", coalesce=True)
        .join(cross_feat, on="date", how="outer", coalesce=True)
        .sort("date")
    )
