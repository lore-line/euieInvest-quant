"""Bivariate winner-vs-loser feature scan on the cleaned price_history.

Independent baseline for cross-checking the XGBoost Step 2 SHAP rankings.

For each candidate feature, compute:
  - mean(feature | winner=1) vs mean(feature | winner=0)
  - Cohen's d (pooled-std-normalized mean difference)
  - n_winners, n_losers
  - rank by |Cohen's d|

The win condition for "signal exists" is at least a handful of features
with |d| >= 0.2 (small effect) — confirms the 18.94% base rate isn't
pure noise and the model has something to learn from.

Cross-reference: XGB Step 2 top SHAP features were
  atr_pct_14 (+), pct_of_252d_high (-), market_regime_chop (-),
  pct_of_252d_low (+), days_since_last_20pct (-), close_over_sma_200 (+).

If the bivariate scan finds the same features near the top, the signal
is robust. If bivariate finds nothing and XGB claims AUC 0.73, we get
suspicious about overfit.

Run:
    cd quant_api
    .venv/bin/python analysis/bivariate_winner_scan.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

DB_PATH = Path("/home/euie/nextcloud/CODE/euieInvest/data/euieinvest.db")

# Holdout window matches XGB Step 2 manifest.json so the comparison is
# apples-to-apples.
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-03-30"


def load_prices() -> pl.DataFrame:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT symbol, date, close, close_adj, high, low, volume, open "
        "FROM price_history "
        "WHERE close_adj IS NOT NULL AND close_adj > 0"
    ).fetchall()
    conn.close()
    df = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "date": pl.Utf8,
            "close": pl.Float64,
            "close_adj": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "volume": pl.Float64,
            "open": pl.Float64,
        },
        orient="row",
    ).with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))
    return df.sort(["symbol", "date"])


def add_features_and_label(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-symbol features + the +20%/30d forward-max label.

    All features at time t use only data from t and earlier (no leakage).
    Label uses close_adj[t+1..t+30] strictly forward.
    """
    return df.with_columns(
        # Daily returns
        ret_1d=pl.col("close_adj").pct_change().over("symbol"),
    ).with_columns(
        # Rolling means / max / min over close_adj
        sma_20=pl.col("close_adj").rolling_mean(window_size=20).over("symbol"),
        sma_50=pl.col("close_adj").rolling_mean(window_size=50).over("symbol"),
        sma_200=pl.col("close_adj").rolling_mean(window_size=200).over("symbol"),
        max_252=pl.col("close_adj").rolling_max(window_size=252).over("symbol"),
        min_252=pl.col("close_adj").rolling_min(window_size=252).over("symbol"),
        max_30=pl.col("close_adj").rolling_max(window_size=30).over("symbol"),
        min_30=pl.col("close_adj").rolling_min(window_size=30).over("symbol"),

        # True range (Wilder ATR ingredient): max of (high-low, |high-prev_close|, |low-prev_close|)
        # Use plain (high-low)/close as a cheaper proxy for ATR%.
        hl_pct=((pl.col("high") - pl.col("low")) / pl.col("close")),

        # Volume avg for vol-ratio
        avg_vol_20=pl.col("volume").rolling_mean(window_size=20).over("symbol"),

        # Realized vol of daily returns (20d)
        rvol_20=pl.col("close_adj").pct_change().rolling_std(window_size=20).over("symbol"),

        # Returns over lookback windows
        ret_5d=(pl.col("close_adj") / pl.col("close_adj").shift(5).over("symbol") - 1),
        ret_20d=(pl.col("close_adj") / pl.col("close_adj").shift(20).over("symbol") - 1),
        ret_60d=(pl.col("close_adj") / pl.col("close_adj").shift(60).over("symbol") - 1),

        # 30-day forward MAX of close_adj — the label numerator. Strictly forward.
        # shift(-30) + rolling_max(30) gives max(close[t+1..t+30]) at index t:
        # shift(-30)[t] = close[t+30]; rolling_max(30) at t = max over shifted[t-29..t]
        # = max of close[t-29+30 .. t+30] = max of close[t+1 .. t+30]. ✓
        # The naive shift(-1).rolling_max(30) is WRONG — that yields max(close[t-28..t+1]),
        # which is mostly BACKWARD-looking. Verified by unit test.
        fwd_max_30=pl.col("close_adj").shift(-30).rolling_max(window_size=30).over("symbol"),
    ).with_columns(
        # Derived features
        atr_pct_14=pl.col("hl_pct").rolling_mean(window_size=14).over("symbol"),
        vol_ratio_20=pl.col("volume") / pl.col("avg_vol_20"),
        pct_of_252d_high=pl.col("close_adj") / pl.col("max_252"),
        pct_of_252d_low=pl.col("close_adj") / pl.col("min_252"),
        close_over_sma_20=pl.col("close_adj") / pl.col("sma_20"),
        close_over_sma_50=pl.col("close_adj") / pl.col("sma_50"),
        close_over_sma_200=pl.col("close_adj") / pl.col("sma_200"),
        distance_from_max_30=pl.col("close_adj") / pl.col("max_30"),
        distance_from_min_30=pl.col("close_adj") / pl.col("min_30"),

        # Label: max forward close in next 30 days / current close >= 1.20
        winner=(pl.col("fwd_max_30") / pl.col("close_adj") >= 1.20).cast(pl.Int8),
    )


def cohens_d(s: pl.DataFrame, feature: str) -> dict:
    """Compute Cohen's d for a feature between winner and loser distributions."""
    w = s.filter(pl.col("winner") == 1)[feature].drop_nans().drop_nulls()
    l = s.filter(pl.col("winner") == 0)[feature].drop_nans().drop_nulls()
    n_w = w.len()
    n_l = l.len()
    if n_w < 100 or n_l < 100:
        return {
            "feature": feature, "n_winners": n_w, "n_losers": n_l,
            "mean_winner": None, "mean_loser": None, "cohens_d": None,
        }
    mu_w = w.mean()
    mu_l = l.mean()
    var_w = w.var()
    var_l = l.var()
    pooled = (((n_w - 1) * var_w + (n_l - 1) * var_l) / (n_w + n_l - 2)) ** 0.5
    d = (mu_w - mu_l) / pooled if pooled > 0 else None
    return {
        "feature": feature, "n_winners": n_w, "n_losers": n_l,
        "mean_winner": mu_w, "mean_loser": mu_l, "cohens_d": d,
    }


def main() -> None:
    print("loading price_history...")
    df = load_prices()
    print(f"  {df.height:,} rows, {df['symbol'].n_unique():,} symbols")

    print("computing features + label...")
    df = add_features_and_label(df)

    # Filter to the holdout window (same as XGB Step 2 manifest)
    holdout = df.filter(
        (pl.col("date") >= pl.lit(HOLDOUT_START).str.strptime(pl.Date))
        & (pl.col("date") <= pl.lit(HOLDOUT_END).str.strptime(pl.Date))
        & (pl.col("winner").is_not_null())
        & (pl.col("fwd_max_30").is_not_null())  # need 30 forward days for label
    )
    print(f"  holdout: {holdout.height:,} rows ({HOLDOUT_START} → {HOLDOUT_END})")

    pos_rate = holdout["winner"].mean()
    print(f"  positive rate (winners): {pos_rate:.4f}")

    features = [
        "atr_pct_14",
        "pct_of_252d_high",
        "pct_of_252d_low",
        "close_over_sma_20",
        "close_over_sma_50",
        "close_over_sma_200",
        "vol_ratio_20",
        "rvol_20",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "distance_from_max_30",
        "distance_from_min_30",
        "hl_pct",
    ]

    print("\nbivariate scan:")
    print(f"  {'feature':<22} {'n_w':>8} {'n_l':>8} {'mean(W)':>10} {'mean(L)':>10} {'Cohen d':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    results = []
    for feat in features:
        r = cohens_d(holdout, feat)
        results.append(r)

    results.sort(key=lambda r: abs(r["cohens_d"]) if r["cohens_d"] is not None else -1, reverse=True)
    for r in results:
        d = r["cohens_d"]
        if d is None:
            print(f"  {r['feature']:<22} {r['n_winners']:>8} {r['n_losers']:>8} {'—':>10} {'—':>10} {'—':>10}")
        else:
            mw = r["mean_winner"]
            ml = r["mean_loser"]
            print(f"  {r['feature']:<22} {r['n_winners']:>8,} {r['n_losers']:>8,} {mw:>10.4f} {ml:>10.4f} {d:>+10.4f}")

    print("\nfeatures with |Cohen's d| >= 0.20 (small effect or better):")
    significant = [r for r in results if r["cohens_d"] is not None and abs(r["cohens_d"]) >= 0.20]
    for r in significant:
        print(f"  {r['feature']:<22}  d = {r['cohens_d']:+.4f}  (W={r['mean_winner']:.4f} vs L={r['mean_loser']:.4f})")

    if not significant:
        print("  (none — bivariate signal is weak; XGB might be relying on feature interactions)")


if __name__ == "__main__":
    main()
