"""Track 2 — Hand-crafted feature clustering of winners.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 2.

Cluster the WINNERS-only subset of the holdout in the 47-dim
hand-crafted feature space, across multiple algorithms and K values.
Each clustering's per-cluster signature (size, centroid feature
values, top differentiating features vs population mean, example
(symbol, date) windows) becomes candidate thesis material — a
cluster is a "shape" of winner setup.

Configs (per brief):
  - KMeans  × K ∈ {5, 8, 12, 20}
  - GMM     × K ∈ {5, 8, 12, 20}
  - HDBSCAN × min_cluster_size = 200  (K auto-determined)
  → 9 clusterings total

Outputs:
  clusters.parquet           — per-cluster signature + examples
  cluster-membership.parquet — per-window cluster assignments

Both schemas are pinned in docs/reports-repo-layout.md.

CPU only; ~2-5 min on the ~128K winners-on-holdout subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_HDBSCAN_AVAILABLE = True  # sklearn.cluster.HDBSCAN is built in to sklearn ≥ 1.3

from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.tracks import make_run_id

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NON_FEATURE_COLS = frozenset(
    {"symbol", "date", "open", "high", "low", "close", "close_adj", "volume", "is_winner"}
)
_MAX_EXAMPLES_PER_CLUSTER = 20
_TOP_SIGNATURE_FEATURES = 8


def _prepare_feature_matrix(winners: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Convert to numpy, fill NaN with column-median, standardize.

    KMeans and GMM both choke on NaN; HDBSCAN tolerates them but
    distance computations get noisy. Median-impute is the lowest-
    risk default — preserves the distribution shape vs zero-impute
    which biases toward the mean for skewed features.
    """
    X = winners.select(feature_cols).to_numpy().astype(np.float64, copy=False)
    # Column-wise NaN handling.
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        medians = np.nanmedian(np.where(nan_mask, np.nan, X), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        for c in range(X.shape[1]):
            X[nan_mask[:, c], c] = medians[c]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols


def _cluster_signature(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    feature_names: list[str],
    population_mean: np.ndarray,
    population_std: np.ndarray,
) -> dict[str, Any]:
    """Per-cluster signature: mean feature values + top features by
    |mean(cluster) - mean(population)| / population_std (z-score
    distance from population)."""
    member_mask = labels == cluster_id
    n_members = int(member_mask.sum())
    if n_members == 0:
        return {"size": 0}
    cluster_mean = X_scaled[member_mask].mean(axis=0)
    # Already standardized, so cluster_mean IS the z-score vs population.
    abs_z = np.abs(cluster_mean)
    top_indices = np.argsort(-abs_z)[:_TOP_SIGNATURE_FEATURES]
    signature = [
        {
            "feature": feature_names[i],
            "cluster_mean_z": round(float(cluster_mean[i]), 4),
            "direction": "+" if cluster_mean[i] > 0 else "-",
        }
        for i in top_indices
        if abs_z[i] > 0.1  # filter very weak signals
    ]
    return {"size": n_members, "signature_features": signature}


def _examples(symbols: np.ndarray, dates: np.ndarray, member_indices: np.ndarray) -> list[dict[str, str]]:
    """Up to N example (symbol, date) windows from a cluster, sampled
    deterministically (first N by index — the winners frame is sorted
    by (symbol, date))."""
    picks = member_indices[:_MAX_EXAMPLES_PER_CLUSTER]
    return [
        {"symbol": str(symbols[i]), "date": str(dates[i])}
        for i in picks
    ]


def _run_one_clustering(
    algorithm: str,
    k: int | None,
    X_scaled: np.ndarray,
    feature_names: list[str],
    symbols: np.ndarray,
    dates: np.ndarray,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one clustering config; return (cluster_rows, membership_rows)."""
    print(
        f"    {algorithm:<8} "
        f"{'K=' + str(k) if k is not None else 'min_size=200':<14} "
        f"fitting ...",
        end="",
        flush=True,
    )
    t0 = time.perf_counter()
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=k, random_state=random_state, covariance_type="diag"
        )
        labels = model.fit_predict(X_scaled)
    elif algorithm == "hdbscan":
        # HDBSCAN on the full 128K-winner set is O(N²) in the worst case
        # and runs > 15 min. Subsample to 30K for the fit, propagate
        # cluster labels back via 1-NN to the unsubsampled rows. Beyond
        # 30K the cluster shapes are stable; 1-NN propagation is cheap.
        n = X_scaled.shape[0]
        sample_size = min(30_000, n)
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
        sample_model = HDBSCAN(min_cluster_size=200)
        sample_labels = sample_model.fit_predict(X_sample)
        # 1-NN propagation. sklearn's HDBSCAN doesn't transform-on-new;
        # NearestNeighbors is fast on the 30K sample (~1s).
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1).fit(X_sample)
        _, neighbor_idx = nn.kneighbors(X_scaled)
        labels = sample_labels[neighbor_idx[:, 0]]
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    elapsed = time.perf_counter() - t0

    # population stats (in z-space the mean is ~0, std is ~1, but compute
    # for completeness — HDBSCAN can leave outliers as label=-1 which
    # would otherwise skew the cluster signature math).
    pop_mean = X_scaled.mean(axis=0)
    pop_std = X_scaled.std(axis=0) + 1e-9

    # Skip outlier label -1 (HDBSCAN's "not in any cluster" bucket).
    unique_clusters = sorted(set(labels.tolist()) - {-1})
    cluster_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for cid in unique_clusters:
        sig = _cluster_signature(X_scaled, labels, cid, feature_names, pop_mean, pop_std)
        if sig["size"] == 0:
            continue
        idx = np.flatnonzero(labels == cid)
        ex = _examples(symbols, dates, idx)
        cluster_rows.append(
            {
                "algorithm": algorithm,
                "k": k if k is not None else 0,
                "cluster_id": int(cid),
                "size": sig["size"],
                "signature_features_json": json.dumps(sig.get("signature_features", [])),
                "example_symbol_dates_json": json.dumps(ex),
            }
        )
        for i in idx:
            membership_rows.append(
                {
                    "symbol": str(symbols[i]),
                    "date": str(dates[i]),
                    "algorithm": algorithm,
                    "k": k if k is not None else 0,
                    "cluster_id": int(cid),
                }
            )
    n_outliers = int((labels == -1).sum())
    print(
        f" {len(unique_clusters)} clusters, "
        f"{n_outliers:,} outliers, "
        f"{elapsed:.1f}s"
    )
    return cluster_rows, membership_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 2 — winners clustering")
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument(
        "--k-values", type=str, default="5,8,12,20",
        help="Comma-separated K values for KMeans and GMM.",
    )
    p.add_argument(
        "--hdbscan-min-cluster-size", type=int, default=200,
        help="min_cluster_size for the single HDBSCAN run.",
    )
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step3b_handcrafted_clustering"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(dir=run_dir, run_id=make_run_id(run_date_str, pipeline_step), pipeline_step=pipeline_step, epoch_total=1)
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        print(f"track 2 (winners clustering) — run dir: {run_dir.relative_to(_REPO_ROOT)}")
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        if not features_path.exists():
            raise FileNotFoundError(features_path)

        labeled = pl.read_parquet(features_path).filter(pl.col("is_winner").is_not_null())
        labeled, feature_cols = _replay_feature_selection(labeled)
        winners = labeled.filter(
            (pl.col("date") > args.val_end) & (pl.col("is_winner") == True)
        ).sort(["symbol", "date"])
        print(
            f"  winners-on-holdout: {winners.height:,} rows × {len(feature_cols)} features  "
            f"({winners['date'].min()}→{winners['date'].max()})"
        )

        X_scaled, feature_names = _prepare_feature_matrix(winners, feature_cols)
        symbols = winners["symbol"].to_numpy()
        dates = winners["date"].to_numpy().astype("datetime64[D]").astype(str)

        k_values = [int(s) for s in args.k_values.split(",")]
        configs: list[tuple[str, int | None]] = []
        for k in k_values:
            configs.append(("kmeans", k))
            configs.append(("gmm", k))
        configs.append(("hdbscan", None))

        print(f"  running {len(configs)} clusterings ...")
        all_cluster_rows: list[dict[str, Any]] = []
        all_membership_rows: list[dict[str, Any]] = []
        for algorithm, k in configs:
            if stop_flag["stop"]:
                raise KeyboardInterrupt
            cluster_rows, membership_rows = _run_one_clustering(
                algorithm, k, X_scaled, feature_names, symbols, dates
            )
            all_cluster_rows.extend(cluster_rows)
            all_membership_rows.extend(membership_rows)
        print(
            f"  total: {len(all_cluster_rows)} clusters across "
            f"{len(configs)} configurations"
        )

        # Write artifacts.
        clusters_df = pl.DataFrame(
            all_cluster_rows,
            schema={
                "algorithm": pl.Utf8,
                "k": pl.Int64,
                "cluster_id": pl.Int64,
                "size": pl.Int64,
                "signature_features_json": pl.Utf8,
                "example_symbol_dates_json": pl.Utf8,
            },
        ).sort(["algorithm", "k", "cluster_id"])
        clusters_path = run_dir / "clusters.parquet"
        clusters_df.write_parquet(clusters_path)
        print(f"  wrote {clusters_path.relative_to(_REPO_ROOT)}  ({clusters_df.height} cluster rows)")

        membership_df = pl.DataFrame(
            all_membership_rows,
            schema={
                "symbol": pl.Utf8,
                "date": pl.Utf8,
                "algorithm": pl.Utf8,
                "k": pl.Int64,
                "cluster_id": pl.Int64,
            },
        ).with_columns(pl.col("date").str.to_date())
        membership_path = run_dir / "cluster-membership.parquet"
        membership_df.write_parquet(membership_path)
        print(f"  wrote {membership_path.relative_to(_REPO_ROOT)}  ({membership_df.height:,} rows)")

        # Manifest.
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "val_end": args.val_end.isoformat(),
            "holdout_start": "2025-01-02",
            "holdout_end": str(winners["date"].max()),
            "n_winners_clustered": int(winners.height),
            "feature_count": len(feature_cols),
            "n_clusterings": len(configs),
            "n_clusters_total": int(clusters_df.height),
            "algorithms": ["kmeans", "gmm"] + (["hdbscan"] if _HDBSCAN_AVAILABLE else []),
            "k_values": k_values,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "runtime_device": "cpu",
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(
            f"=== TRACK 2 RESULT: {clusters_df.height} clusters across "
            f"{len(configs)} configs ({wall_clock_s:.1f}s) ==="
        )
        status.record_checkpoint(epoch=1)
        status.update(state="done", epoch_current=1)
        return 0
    except KeyboardInterrupt:
        status.update(state="paused", epoch_current=0)
        return 130
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
