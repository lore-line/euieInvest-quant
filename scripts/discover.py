"""Discovery pipeline entrypoint — 5-step orchestrator.

Steps 1 and 2 (feature engineering, supervised discovery + SHAP) are
implemented. Steps 3-5 (clustering, counterfactuals, tier-3 comparison)
remain scaffolded — see CLAUDE.md §5 for the full methodology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant.backtest.temporal import split_by_date
from quant.data.loader import load_ohlcv, load_peer_groups
from quant.features import (
    behavioral,
    gaps,
    momentum,
    price,
    relative,
    volatility,
    volume,
)
from quant.labels import compute_forward_winner_labels
from quant.models import XGBDiscovery

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Columns excluded from the feature matrix: raw OHLCV (used to derive
# features but not as features themselves) and the label.
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"symbol", "date", "open", "high", "low", "close", "close_adj", "volume", "is_winner"}
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Winner-fingerprint discovery pipeline (CLAUDE.md §5)"
    )
    p.add_argument("--train-end", type=date.fromisoformat, default=date(2023, 12, 31))
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument(
        "--features-out",
        type=Path,
        default=Path("data/features/features.parquet"),
        help="Where to write the joined features + labels parquet",
    )
    p.add_argument(
        "--stop-after",
        choices=["step1", "step2", "step3", "step4", "step5"],
        default="step2",
        help="Run pipeline up to and including this step (default: step2). "
        "Steps 3-5 are scaffolded; ask before bumping past step2.",
    )
    p.add_argument(
        "--skip-step1",
        action="store_true",
        help="Skip feature engineering and reuse the existing features parquet.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override the run output dir. Default: "
        "runs/<today>-<pipeline_step>/. Use for retroactive overwrites "
        "of an existing dated run (e.g. runs/2026-05-12/).",
    )
    p.add_argument(
        "--top-per-day-k",
        type=int,
        default=20,
        help="Per-day K for top-per-day.parquet (default 20).",
    )
    return p.parse_args(argv)


def _build_features(
    ohlcv: pl.DataFrame,
    spy: pl.DataFrame,
    peer_groups_dict: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply every implemented feature transform to ``ohlcv`` and return
    a single DataFrame sorted by (symbol, date) with all feature columns
    appended.

    Features that require data missing from the current snapshot
    (`open` for gap_pct/body_range_ratio; market-cap for cap_bucket)
    are skipped here — they remain importable but unused until upstream
    catches up.
    """
    out = ohlcv
    # price.py — close/SMA, SMA slope, Bollinger band position, N-day extremes
    out = price.sma_distance(out)
    out = price.sma_slope(out, window=20, lookback=5)
    out = price.sma_slope(out, window=50, lookback=5)
    out = price.band_position(out, window=20)
    out = price.n_day_high_low(out)

    # volume.py — volume multiples, OBV slope, A/D line
    out = volume.vol_mult(out)
    out = volume.obv_slope(out, lookback=20)
    out = volume.accumulation_distribution(out)

    # volatility.py — ATR%, BB squeeze, NR4/7, HV ratio
    out = volatility.atr_pct(out, window=14)
    out = volatility.bb_squeeze(out, window=20)
    out = volatility.nr4_nr7(out)
    out = volatility.hv_ratio(out, short_window=10, long_window=60)

    # momentum.py — RSI{2,5,14}, MACD, ROC, consecutive runs
    out = momentum.rsi(out)
    out = momentum.macd(out)
    out = momentum.roc(out)
    out = momentum.consecutive_run(out)

    # gaps.py — all four functions now implementable (open column landed)
    out = gaps.gap_pct(out)
    out = gaps.range_expansion(out, lookback=5)
    out = gaps.body_range_ratio(out)
    out = gaps.inside_bar(out)

    # relative.py — vs SPY (full df), vs sector (peer groups), peer z-scores
    out = relative.rel_strength_spy(out, spy, lookback=20)
    out = relative.rel_strength_sector(out, peer_groups_dict, lookback=20)
    out = relative.peer_zscore(out, peer_groups_dict, column="close_over_sma_20")

    # behavioral.py — days_since_last_20pct + SPY-derived market regime
    out = behavioral.days_since_last_20pct(out)
    regime = behavioral.market_regime(spy)
    out = out.join(regime, on="date", how="left")

    return out


def step1_build_features(args: argparse.Namespace) -> pl.DataFrame:
    """Build features + labels, write to parquet, return the DataFrame."""
    print("step 1: building features ...")
    ohlcv = load_ohlcv()
    print(f"  loaded ohlcv: {ohlcv.height:,} rows, {ohlcv['symbol'].n_unique()} symbols")
    spy = load_ohlcv("SPY")
    print(f"  loaded SPY: {spy.height} rows")
    peer_groups_dict = load_peer_groups()
    print(f"  loaded peer_groups: {len(peer_groups_dict)} groups")

    features = _build_features(ohlcv, spy, peer_groups_dict)
    print(f"  built features: {features.height:,} rows × {features.width} cols")

    # Labels per CLAUDE.md §6 — total-return on close_adj, NOT split-only close.
    labeled = compute_forward_winner_labels(
        features, lookahead=30, threshold=0.20, price_col="close_adj"
    )
    print(f"  labeled: {labeled['is_winner'].sum()} winners "
          f"({100.0 * labeled['is_winner'].sum() / labeled['is_winner'].drop_nulls().len():.2f}% "
          f"of non-null rows)")

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    labeled.write_parquet(args.features_out)
    print(f"  wrote features+labels parquet -> {args.features_out}")
    return labeled


def _feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def step2_supervised_discovery(args: argparse.Namespace) -> None:
    """Train XGBClassifier on the train slice, evaluate on the holdout
    slice ONCE, write three artifacts to ``runs/<date>/``.

    Artifacts (per the reports-repo-layout contract, Step 2 fields):
    - ``manifest.json`` — run metadata + headline metrics
    - ``top-decile.parquet`` — holdout rows in the top decile by predicted proba
    - ``shap-summary.parquet`` — mean-|SHAP| per feature + direction
    """
    print("step 2: supervised discovery ...")
    labeled = pl.read_parquet(args.features_out)

    # Drop rows without a forward label (last 30 per symbol).
    labeled = labeled.filter(pl.col("is_winner").is_not_null())

    feature_cols = _feature_columns(labeled)

    # XGBoost needs numeric inputs.
    # - Booleans → int8
    # - Strings (e.g. market_regime: "uptrend"/"downtrend"/"chop") → one-hot dummies
    bool_cols = [c for c in feature_cols if labeled[c].dtype == pl.Boolean]
    if bool_cols:
        labeled = labeled.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])

    str_cols = [c for c in feature_cols if labeled[c].dtype == pl.Utf8]
    if str_cols:
        labeled = labeled.to_dummies(columns=str_cols)
        # to_dummies emits cols named "<col>_<value>" as UInt8 — keep them, drop originals.
        feature_cols = _feature_columns(labeled)

    print(f"  feature columns: {len(feature_cols)} (str→dummies: {len(str_cols)})")

    train, val, holdout = split_by_date(labeled, args.train_end, args.val_end)
    print(
        f"  splits: train={train.height:,} ({train['date'].min()}→{train['date'].max()}) | "
        f"val={val.height:,} ({val['date'].min()}→{val['date'].max()}) | "
        f"holdout={holdout.height:,} ({holdout['date'].min()}→{holdout['date'].max()})"
    )

    # scale_pos_weight from the TRAIN slice only (CLAUDE.md §9).
    n_pos_train = int(train["is_winner"].sum())
    n_neg_train = train.height - n_pos_train
    spw = n_neg_train / max(n_pos_train, 1)
    print(
        f"  train balance: {n_pos_train:,} pos / {n_neg_train:,} neg "
        f"({100.0 * n_pos_train / train.height:.2f}% positive) "
        f"→ scale_pos_weight={spw:.4f}"
    )

    X_train = train.select(feature_cols)
    y_train = train["is_winner"]
    X_val = val.select(feature_cols)
    y_val = val["is_winner"]
    X_holdout = holdout.select(feature_cols)
    y_holdout = holdout["is_winner"]

    model = XGBDiscovery(scale_pos_weight=spw)
    print("  fitting XGBClassifier ...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    print("  predicting on holdout ...")
    holdout_proba = model.predict(X_holdout)

    # Metrics on the holdout (touched ONCE — CLAUDE.md §8).
    from sklearn.metrics import roc_auc_score  # local import to keep top-level light

    proba_np = holdout_proba.to_numpy()
    y_np = y_holdout.cast(pl.Boolean).to_numpy()

    auc = float(roc_auc_score(y_np, proba_np))

    n_holdout = len(proba_np)
    k = max(1, n_holdout // 10)
    # Partial sort: indices of the k highest probas (in arbitrary order).
    top_idx = np.argpartition(-proba_np, k - 1)[:k]
    n_true_in_top = int(y_np[top_idx].sum())
    n_total_pos = int(y_np.sum())
    precision_at_topdecile = n_true_in_top / k
    recall_at_topdecile = n_true_in_top / n_total_pos if n_total_pos else 0.0
    base_rate = n_total_pos / n_holdout
    lift = precision_at_topdecile / base_rate if base_rate else float("nan")

    print()
    print(f"  HOLDOUT METRICS (n={n_holdout:,}, top decile k={k:,})")
    print(f"    AUC                        = {auc:.4f}")
    print(
        f"    precision @ top-decile     = {precision_at_topdecile:.4f}  "
        f"(base rate {base_rate:.4f}, lift {lift:.2f}x)"
    )
    print(f"    recall    @ top-decile     = {recall_at_topdecile:.4f}")
    print()

    # SHAP on the holdout — that's what tells us what the model leaned on.
    print("  computing SHAP on holdout ...")
    shap_df = model.shap_summary(X_holdout)
    print("  top-10 features by mean-|SHAP|:")
    for row in shap_df.head(10).iter_rows(named=True):
        print(f"    {row['feature_name']:<35} {row['mean_abs_shap']:>10.5f}  {row['direction']}")
    print()

    # Pipeline-step bucket (drives both manifest field + default run dir name).
    edge_threshold = 0.25  # 25% > 18.94% base; matches the brief's edge bar.
    pipeline_step = (
        "step2_supervised_discovery"
        if precision_at_topdecile >= edge_threshold
        else "step2_no_edge_found"
    )

    # ----- Artifacts -----
    # Default to the canonical suffixed dir; --run-dir overrides for
    # retroactive overwrites of historical dated runs (e.g. 2026-05-12).
    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else (_REPO_ROOT / args.run_dir)
        # When overriding the dir (retroactive overwrite), the run_id's date
        # prefix should match the dir's date prefix — not today's UTC date.
        run_date_str = run_dir.name[:10]
    else:
        run_date_str = date.today().isoformat()
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run dir: {run_dir.relative_to(_REPO_ROOT)}")

    # Save model alongside the manifest so its sha can be referenced.
    model_path = run_dir / "model.json"
    model._model.save_model(str(model_path))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # top-decile.parquet: global top 10% — cross-architecture comparison surface.
    top_mask = np.zeros(n_holdout, dtype=bool)
    top_mask[top_idx] = True
    top_df = (
        holdout.select(["symbol", "date"])
        .with_columns(holdout_proba.cast(pl.Float64))
        .filter(pl.Series(values=top_mask))
        .sort(["date", "predicted_proba"], descending=[False, True])
    )
    topdecile_path = run_dir / "top-decile.parquet"
    top_df.write_parquet(topdecile_path)
    print(f"  wrote {topdecile_path.relative_to(_REPO_ROOT)}  ({top_df.height:,} rows)")

    # top-per-day.parquet: per-day top-K — dashboard surface.
    per_day_k = max(1, int(args.top_per_day_k))
    top_per_day = (
        holdout.select(["symbol", "date"])
        .with_columns(holdout_proba.cast(pl.Float64))
        .with_columns(
            pl.col("predicted_proba")
            .rank(method="ordinal", descending=True)
            .over("date")
            .cast(pl.Int64)
            .alias("rank_within_day")
        )
        .filter(pl.col("rank_within_day") <= per_day_k)
        .sort(["date", "rank_within_day"])
    )
    top_per_day_path = run_dir / "top-per-day.parquet"
    top_per_day.write_parquet(top_per_day_path)
    print(
        f"  wrote {top_per_day_path.relative_to(_REPO_ROOT)}  "
        f"(K={per_day_k}/day, {top_per_day.height:,} rows)"
    )

    # shap-summary.parquet (column meaning: TreeSHAP for XGB; for DL the
    # same column will carry IG attribution — see reports-repo-layout.md).
    shap_path = run_dir / "shap-summary.parquet"
    shap_df.write_parquet(shap_path)
    print(f"  wrote {shap_path.relative_to(_REPO_ROOT)}  ({shap_df.height} features)")

    manifest = {
        "run_id": f"{run_date_str}-001",
        "train_end": args.train_end.isoformat(),
        "val_end": args.val_end.isoformat(),
        "holdout_end": str(holdout["date"].max()),
        "model_sha": f"sha256:{model_sha}",
        "feature_count": len(feature_cols),
        "positive_rate_train": round(n_pos_train / train.height, 6),
        "holdout_precision_at_topdecile": round(precision_at_topdecile, 6),
        "holdout_recall_at_topdecile": round(recall_at_topdecile, 6),
        "holdout_auc": round(auc, 6),
        "holdout_base_rate": round(base_rate, 6),
        "holdout_n_rows": n_holdout,
        "holdout_top_decile_k": k,
        "top_per_day_k": per_day_k,
        "universe_size": int(labeled["symbol"].n_unique()),
        "git_commit_of_quant_repo": _git_head_sha(),
        "pipeline_step": pipeline_step,
        # Self-attestation: where fit() actually ran and how long it took.
        # Sourced from the trained booster's save_config(), so a silent
        # CUDA→CPU fallback would surface here as device="cpu".
        "runtime_device": model.runtime_device,
        "train_wall_clock_s": round(model.train_wall_clock_s or 0.0, 3),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {manifest_path.relative_to(_REPO_ROOT)}")
    print()
    print(
        f"=== STEP 2 RESULT: precision@top-decile = {precision_at_topdecile*100:.2f}% "
        f"(base rate {base_rate*100:.2f}%, AUC {auc:.3f}, "
        f"pipeline_step={pipeline_step}) ==="
    )


def step3_cluster_winners(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step3_cluster_winners — KMeans on winner-only "
        "rows for k in (3,5,8); select by silhouette. See CLAUDE.md §5 step 3."
    )


def step4_counterfactuals(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step4_counterfactuals — closest non-winners "
        "per winner cluster; report feature deltas. See CLAUDE.md §5 step 4."
    )


def step5_tier3_comparison(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step5_tier3_comparison — overlap, recall, and "
        "missed-winners vs the anomaly_flags baseline. See CLAUDE.md §5 step 5."
    )


_STEP_ORDER = ["step1", "step2", "step3", "step4", "step5"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop_idx = _STEP_ORDER.index(args.stop_after)

    if args.skip_step1:
        if not args.features_out.exists():
            raise SystemExit(
                f"--skip-step1 set but {args.features_out} does not exist. "
                "Run without --skip-step1 once to build it."
            )
        print(f"step 1: skipped (reusing {args.features_out})")
    else:
        step1_build_features(args)

    if stop_idx >= _STEP_ORDER.index("step2"):
        step2_supervised_discovery(args)
    if stop_idx >= _STEP_ORDER.index("step3"):
        step3_cluster_winners(args)
    if stop_idx >= _STEP_ORDER.index("step4"):
        step4_counterfactuals(args)
    if stop_idx >= _STEP_ORDER.index("step5"):
        step5_tier3_comparison(args)


if __name__ == "__main__":
    main()
