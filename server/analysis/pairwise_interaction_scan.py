"""Pairwise feature interaction scan on the cleaned price_history.

Extends bivariate_winner_scan.py (which is univariate, Cohen's d) to find
feature PAIRS whose joint distribution discriminates winners more strongly
than either feature alone. Cross-validates Track 1's rule extraction: if
XGB rules combine features (X, Y) at high lift, pairwise should
independently surface (X, Y) as a strong interaction.

For each pair (X, Y):
  - Discretize each into quartiles
  - Compute the 4×4 contingency table of winner rates
  - Find the cell with highest lift × coverage
  - Compute "interaction strength" = best_cell_lift - max(marginal_lift_X, marginal_lift_Y)
    A positive interaction strength means the combination is more
    discriminating than either feature on its own (synergistic).

Output:
  - Top-50 pairs by interaction strength
  - For each: best cell description (X range, Y range, lift, precision, coverage)
  - Comparison to Track 1's rules that combine these features

Run:
    cd quant_api
    .venv/bin/python analysis/pairwise_interaction_scan.py
"""

from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path

import polars as pl

DB_PATH = Path("/home/euie/nextcloud/CODE/euieInvest/data/euieinvest.db")
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
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8, "date": pl.Utf8,
            "close": pl.Float64, "close_adj": pl.Float64,
            "high": pl.Float64, "low": pl.Float64,
            "volume": pl.Float64, "open": pl.Float64,
        },
        orient="row",
    ).with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")).sort(["symbol", "date"])


def add_features_and_label(df: pl.DataFrame) -> pl.DataFrame:
    """Same feature set as bivariate_winner_scan, label uses forward-only
    shift(-30).rolling_max(30) for max(close_adj[t+1..t+30])."""
    return df.with_columns(
        ret_1d=pl.col("close_adj").pct_change().over("symbol"),
    ).with_columns(
        sma_20=pl.col("close_adj").rolling_mean(window_size=20).over("symbol"),
        sma_50=pl.col("close_adj").rolling_mean(window_size=50).over("symbol"),
        sma_200=pl.col("close_adj").rolling_mean(window_size=200).over("symbol"),
        max_252=pl.col("close_adj").rolling_max(window_size=252).over("symbol"),
        min_252=pl.col("close_adj").rolling_min(window_size=252).over("symbol"),
        max_30=pl.col("close_adj").rolling_max(window_size=30).over("symbol"),
        min_30=pl.col("close_adj").rolling_min(window_size=30).over("symbol"),
        hl_pct=((pl.col("high") - pl.col("low")) / pl.col("close")),
        avg_vol_20=pl.col("volume").rolling_mean(window_size=20).over("symbol"),
        rvol_20=pl.col("close_adj").pct_change().rolling_std(window_size=20).over("symbol"),
        ret_5d=(pl.col("close_adj") / pl.col("close_adj").shift(5).over("symbol") - 1),
        ret_20d=(pl.col("close_adj") / pl.col("close_adj").shift(20).over("symbol") - 1),
        ret_60d=(pl.col("close_adj") / pl.col("close_adj").shift(60).over("symbol") - 1),
        fwd_max_30=pl.col("close_adj").shift(-30).rolling_max(window_size=30).over("symbol"),
    ).with_columns(
        atr_pct_14=pl.col("hl_pct").rolling_mean(window_size=14).over("symbol"),
        vol_ratio_20=pl.col("volume") / pl.col("avg_vol_20"),
        pct_of_252d_high=pl.col("close_adj") / pl.col("max_252"),
        pct_of_252d_low=pl.col("close_adj") / pl.col("min_252"),
        close_over_sma_20=pl.col("close_adj") / pl.col("sma_20"),
        close_over_sma_50=pl.col("close_adj") / pl.col("sma_50"),
        close_over_sma_200=pl.col("close_adj") / pl.col("sma_200"),
        distance_from_max_30=pl.col("close_adj") / pl.col("max_30"),
        distance_from_min_30=pl.col("close_adj") / pl.col("min_30"),
        winner=(pl.col("close_adj").shift(-30).rolling_max(window_size=30).over("symbol") / pl.col("close_adj") >= 1.20).cast(pl.Int8),
    )


FEATURES = [
    "atr_pct_14", "pct_of_252d_high", "pct_of_252d_low",
    "close_over_sma_20", "close_over_sma_50", "close_over_sma_200",
    "vol_ratio_20", "rvol_20", "ret_5d", "ret_20d", "ret_60d",
    "distance_from_max_30", "distance_from_min_30", "hl_pct",
]


def quartile_bins(s: pl.Series) -> tuple[float, float, float]:
    """Return (q25, q50, q75) on non-null values."""
    s2 = s.drop_nans().drop_nulls()
    return float(s2.quantile(0.25)), float(s2.quantile(0.5)), float(s2.quantile(0.75))


def discretize(s: pl.Series, q25: float, q50: float, q75: float) -> pl.Series:
    """0=lowest quartile, 1, 2, 3=highest. Null stays null."""
    return (
        pl.when(s.is_null())
        .then(None)
        .when(s < q25)
        .then(0)
        .when(s < q50)
        .then(1)
        .when(s < q75)
        .then(2)
        .otherwise(3)
    )


def main() -> None:
    print("loading + computing features...")
    df = add_features_and_label(load_prices())

    holdout = df.filter(
        (pl.col("date") >= pl.lit(HOLDOUT_START).str.strptime(pl.Date))
        & (pl.col("date") <= pl.lit(HOLDOUT_END).str.strptime(pl.Date))
        & (pl.col("winner").is_not_null())
        & (pl.col("fwd_max_30").is_not_null())
    )
    n_total = holdout.height
    base_rate = float(holdout["winner"].mean())
    print(f"  holdout: {n_total:,} rows, base rate {base_rate:.4f}")

    # Discretize all features into quartiles
    print("\ndiscretizing features...")
    bins: dict[str, tuple[float, float, float]] = {}
    discretized = holdout.select(["winner"])
    for f in FEATURES:
        q25, q50, q75 = quartile_bins(holdout[f])
        bins[f] = (q25, q50, q75)
        discretized = discretized.with_columns(
            discretize(holdout[f], q25, q50, q75).alias(f"q_{f}")
        )

    # Univariate quartile lift (best cell per feature) — for comparison
    print("\nunivariate quartile lifts (best cell per feature):")
    univariate_best: dict[str, dict] = {}
    for f in FEATURES:
        agg = (
            discretized.group_by(f"q_{f}")
            .agg([pl.col("winner").mean().alias("rate"), pl.col("winner").count().alias("n")])
            .filter(pl.col(f"q_{f}").is_not_null())
            .with_columns(lift=pl.col("rate") / base_rate)
            .sort("lift", descending=True)
        )
        best = agg.row(0, named=True)
        q = int(best[f"q_{f}"])
        q_range = (("<", bins[f][0]), ("[q25,q50)", bins[f][0:2]), ("[q50,q75)", bins[f][1:3]), (">=", bins[f][2]))[q]
        univariate_best[f] = {
            "quartile": q,
            "rate": float(best["rate"]),
            "lift": float(best["lift"]),
            "n": int(best["n"]),
            "coverage_pct": 100.0 * int(best["n"]) / n_total,
        }
        print(f"  {f:<22} q{q}: rate={best['rate']:.4f} lift={best['lift']:.3f} cov={100.0*best['n']/n_total:.1f}%")

    # Pairwise interaction scan
    print(f"\npairwise scan: {len(FEATURES) * (len(FEATURES)-1) // 2} pairs × 16 cells each...")
    results = []
    for X, Y in combinations(FEATURES, 2):
        cell_agg = (
            discretized.group_by([f"q_{X}", f"q_{Y}"])
            .agg([pl.col("winner").mean().alias("rate"), pl.col("winner").count().alias("n")])
            .filter(pl.col(f"q_{X}").is_not_null() & pl.col(f"q_{Y}").is_not_null())
            .with_columns(lift=pl.col("rate") / base_rate)
            .filter(pl.col("n") >= n_total * 0.005)  # require coverage >= 0.5% (1k+ rows)
        )
        if cell_agg.height == 0:
            continue
        # Best cell
        best_idx = cell_agg["lift"].arg_max()
        best_cell = cell_agg.row(best_idx, named=True)
        best_cell_lift = float(best_cell["lift"])
        # Interaction strength = best joint lift - max univariate lift on either feature
        marginal_max = max(univariate_best[X]["lift"], univariate_best[Y]["lift"])
        interaction_strength = best_cell_lift - marginal_max
        results.append({
            "X": X, "Y": Y,
            "qX": int(best_cell[f"q_{X}"]),
            "qY": int(best_cell[f"q_{Y}"]),
            "rate": float(best_cell["rate"]),
            "lift": best_cell_lift,
            "n": int(best_cell["n"]),
            "coverage_pct": 100.0 * int(best_cell["n"]) / n_total,
            "marginal_lift_X": univariate_best[X]["lift"],
            "marginal_lift_Y": univariate_best[Y]["lift"],
            "interaction_strength": interaction_strength,
        })

    # Sort by lift * coverage_pct (matches Track 1's ranking convention)
    results.sort(key=lambda r: r["lift"] * r["coverage_pct"], reverse=True)

    print("\nTOP-15 pairs by lift × coverage:\n")
    q_label = ["<q25", "[q25,q50)", "[q50,q75)", ">=q75"]
    header = f"{'X':<22} qX {'Y':<22} qY  lift  prec   cov%   Δvs marg"
    print(header)
    print("-" * len(header))
    for r in results[:15]:
        marg = max(r['marginal_lift_X'], r['marginal_lift_Y'])
        delta = r['lift'] - marg
        sign = "+" if delta >= 0 else ""
        print(
            f"{r['X']:<22} {q_label[r['qX']]:<10} "
            f"{r['Y']:<22} {q_label[r['qY']]:<10} "
            f"{r['lift']:.3f}  {r['rate']*100:.1f}%  {r['coverage_pct']:.1f}%  {sign}{delta:.3f}"
        )

    # Also sort by interaction strength (synergistic only)
    print("\nTOP-10 pairs by interaction strength (synergistic):\n")
    synergistic = [r for r in results if r["interaction_strength"] > 0]
    synergistic.sort(key=lambda r: r["interaction_strength"], reverse=True)
    print(header.replace("Δvs marg", "synergy"))
    print("-" * len(header))
    for r in synergistic[:10]:
        print(
            f"{r['X']:<22} {q_label[r['qX']]:<10} "
            f"{r['Y']:<22} {q_label[r['qY']]:<10} "
            f"{r['lift']:.3f}  {r['rate']*100:.1f}%  {r['coverage_pct']:.1f}%  +{r['interaction_strength']:.3f}"
        )

    # Save full results
    out_df = pl.DataFrame(results)
    out_path = Path("/tmp/pairwise_interactions.parquet")
    out_df.write_parquet(out_path)
    print(f"\nfull results: {out_path} ({len(results)} pairs)")


if __name__ == "__main__":
    main()
