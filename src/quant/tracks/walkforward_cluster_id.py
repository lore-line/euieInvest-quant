"""Track B v3 — Walk-forward cluster identification for honest sleeve baseline.

Track 7's original cluster assignment was produced by k-means on Track F
encoder embeddings of the **same 2025-2026 holdout** the Phase B sleeve
later traded. That's universe-selection forward-look: a real-time sleeve
at date T < 2026-05-13 couldn't have known which symbols would land in
cluster 7 by 2026-05-13.

This script fixes that:

1. Train k-means k=10 on encoder embeddings of windows with date ≤ T
   (default T = 2024-12-31, the Phase A val end / holdout boundary).
2. Compute winner_fraction per cluster on the SAME ≤-T data and identify
   the high-winner "wf-cluster-7" purely from the ≤-T information.
3. Project windows with date > T forward into the trained-cluster space,
   produce a cluster-membership.parquet with the same schema as Track 7's
   original output.
4. (Sleeve step is run separately via paper_sleeve_simulate.py with
   ``--cluster-membership`` pointed at the new file.)

The output validates Phase B v2's +55% headline against a properly
out-of-sample universe definition. Server team's expectation
(PR #1 issuecomment-4452291):

- If walk-forward universe captures ≥ 80% of original (symbol, date)
  tuples → +55% number largely intact.
- If overlap < 60% → expect Sharpe degradation; different problem.

Outputs
-------

- ``cluster-membership-walkforward.parquet`` — same schema as Track 7's
  ``cluster-membership.parquet`` (symbol, date, algorithm, k, cluster_id),
  but only for the > T period and using the walk-forward-trained k-means.
- ``walkforward-cluster-summary.parquet`` — per-cluster aggregates on the
  train period (≤ T) AND the holdout period (> T), used to identify
  which cluster is the "wf-cluster-7" and to compare drift.
- ``overlap-vs-original-cluster-7.parquet`` — diagnostic: per-symbol-date
  comparison of original Track 7 cluster-7 membership vs walk-forward
  cluster-7 membership.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans

from quant.data.windows import build_window_index
from quant.tracks.embedding_clustering import _embed_all_windows, _find_latest_encoder, _load_encoder
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.train import RunStatus
from quant.tracks import make_run_id

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B v3 — Walk-forward cluster identification (honest baseline)"
    )
    p.add_argument(
        "--encoder-path", type=Path, default=None,
        help="Path to Track F encoder.pt. Defaults to latest step3f_foundation_pretrain run.",
    )
    p.add_argument(
        "--features", type=Path, default=Path("data/features/features.parquet"),
        help="Labeled feature matrix.",
    )
    p.add_argument(
        "--cutoff", type=date.fromisoformat, default=date(2024, 12, 31),
        help="Train on windows with date ≤ this; predict on > this.",
    )
    p.add_argument(
        "--k", type=int, default=10,
        help="K-means k. Default 10 to match original Track 7.",
    )
    p.add_argument(
        "--kmeans-fit-sample", type=int, default=200_000,
        help="Subsample size for k-means fit (sklearn KMeans on 1.6M × 768-dim OOMs "
             "even at n_init=10). Centroids transfer to full prediction afterwards. "
             "Set to 0 to disable subsampling.",
    )
    p.add_argument(
        "--original-cluster-membership", type=Path,
        default=Path("runs/2026-05-14-step3g_embedding_clustering/cluster-membership.parquet"),
        help="Original Track 7 cluster-membership.parquet, for overlap diagnostics.",
    )
    p.add_argument(
        "--original-cluster-id", type=int, default=7,
        help="The cluster ID in the ORIGINAL Track 7 that we're comparing against.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
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
    pipeline_step = "step4_walkforward_cluster_id"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(
        dir=run_dir,
        run_id=make_run_id(run_date_str, pipeline_step),
        pipeline_step=pipeline_step,
        epoch_total=4,  # encode-train, kmeans-fit, encode-holdout, predict
    )
    status.update(state="training", epoch_current=0)

    try:
        print(f"walk-forward cluster identification — cutoff = {args.cutoff}")
        encoder_path = args.encoder_path or _find_latest_encoder()
        if encoder_path is None:
            raise FileNotFoundError("no Track F encoder found")
        # Resolve relative paths against repo root (same convention as --features).
        # Without this, --encoder-path runs/... crashes the .relative_to(_REPO_ROOT)
        # call below because the path is relative-but-not-rooted-at-repo.
        if not encoder_path.is_absolute():
            encoder_path = _REPO_ROOT / encoder_path
        if not encoder_path.exists():
            raise FileNotFoundError(f"Track F encoder not found at {encoder_path}")
        print(f"  encoder: {encoder_path.relative_to(_REPO_ROOT)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_encoder(encoder_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  loaded encoder: {n_params/1e6:.2f}M params on {device}")

        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        labeled = pl.read_parquet(features_path).filter(pl.col("is_winner").is_not_null())
        labeled, _ = _replay_feature_selection(labeled)
        print(f"  labeled features: {labeled.height:,} rows")

        # Split into ≤ cutoff (train k-means) and > cutoff (project).
        train_df = labeled.filter(pl.col("date") <= args.cutoff).sort(["symbol", "date"])
        holdout_df = labeled.filter(pl.col("date") > args.cutoff).sort(["symbol", "date"])
        print(f"  train (≤ {args.cutoff}): {train_df.height:,} rows")
        print(f"  holdout (> {args.cutoff}): {holdout_df.height:,} rows")

        # Build window indices.
        print(f"  building train window index ...")
        train_idx = build_window_index(train_df)
        print(f"    {train_idx.n_windows:,} windows")
        print(f"  building holdout window index ...")
        holdout_idx = build_window_index(holdout_df)
        print(f"    {holdout_idx.n_windows:,} windows")

        # Encode train windows (this is where we get the embeddings to fit k-means on).
        print(f"  embedding {train_idx.n_windows:,} train windows ...")
        t_emb = time.perf_counter()
        train_embs = _embed_all_windows(model, train_idx, device)
        print(f"    embedded in {time.perf_counter() - t_emb:.1f}s → shape {train_embs.shape}")
        status.update(state="training", epoch_current=1)

        # Fit k-means on train embeddings.
        # sklearn.KMeans on the full 1.6M × 768-dim with n_init=10 OOMs (each
        # init keeps a (1.6M, 10) distance matrix + intermediate centroids).
        # Subsample for the fit, then predict on all train + holdout afterwards.
        # 200K is plenty for stable k=10 centroids on this embedding distribution.
        rng = np.random.default_rng(42)
        if args.kmeans_fit_sample > 0 and train_embs.shape[0] > args.kmeans_fit_sample:
            sample_idx = rng.choice(train_embs.shape[0], size=args.kmeans_fit_sample, replace=False)
            fit_embs = train_embs[sample_idx]
            print(f"  fitting k-means k={args.k} on {fit_embs.shape[0]:,} subsample of {train_embs.shape[0]:,} ...")
        else:
            fit_embs = train_embs
            print(f"  fitting k-means k={args.k} on all {fit_embs.shape[0]:,} train embeddings ...")
        t_km = time.perf_counter()
        km = KMeans(n_clusters=args.k, n_init=10, random_state=42)
        km.fit(fit_embs)
        print(f"    fit in {time.perf_counter() - t_km:.1f}s")

        print(f"  predicting cluster ids for {train_embs.shape[0]:,} train windows ...")
        t_pred = time.perf_counter()
        train_labels = km.predict(train_embs)
        print(f"    predicted in {time.perf_counter() - t_pred:.1f}s")
        status.update(state="training", epoch_current=2)

        # Compute per-cluster winner_fraction on the TRAIN data (≤ cutoff).
        # This is the cluster-of-interest identification step, fully out-of-sample
        # w.r.t. the holdout.
        train_is_winner = train_idx.labels.astype(bool)
        train_base_rate = float(train_is_winner.mean())
        print(f"  train base winner rate: {train_base_rate:.4f}")

        train_cluster_summary = []
        for cid in range(args.k):
            mask = train_labels == cid
            n_members = int(mask.sum())
            n_winners = int(train_is_winner[mask].sum())
            wf = n_winners / max(n_members, 1)
            train_cluster_summary.append({
                "cluster_id": cid,
                "split": "train",
                "size": n_members,
                "n_winners": n_winners,
                "winner_fraction": round(wf, 6),
                "lift_vs_base": round(wf / max(train_base_rate, 1e-9), 4),
            })

        # Identify wf-cluster-of-interest = cluster with highest winner_fraction on TRAIN.
        train_cluster_summary_sorted = sorted(
            train_cluster_summary, key=lambda r: r["winner_fraction"], reverse=True
        )
        wf_cluster_id = train_cluster_summary_sorted[0]["cluster_id"]
        wf_cluster_train_wf = train_cluster_summary_sorted[0]["winner_fraction"]
        print(f"  ★ wf-cluster-of-interest (highest train winner_fraction): cluster {wf_cluster_id} "
              f"(train winner_fraction = {wf_cluster_train_wf:.4f}, "
              f"lift = {train_cluster_summary_sorted[0]['lift_vs_base']:.3f}x)")

        # Encode holdout windows.
        print(f"  embedding {holdout_idx.n_windows:,} holdout windows ...")
        t_emb = time.perf_counter()
        holdout_embs = _embed_all_windows(model, holdout_idx, device)
        print(f"    embedded in {time.perf_counter() - t_emb:.1f}s → shape {holdout_embs.shape}")
        status.update(state="training", epoch_current=3)

        # Predict holdout cluster assignments using trained k-means.
        print(f"  predicting holdout cluster assignments ...")
        holdout_labels = km.predict(holdout_embs)

        # Compute per-cluster winner_fraction on the HOLDOUT (diagnostic — this is
        # the "do the train-trained clusters retain their winner-density on
        # holdout?" check).
        holdout_is_winner = holdout_idx.labels.astype(bool)
        holdout_base_rate = float(holdout_is_winner.mean())
        print(f"  holdout base winner rate: {holdout_base_rate:.4f}")

        holdout_cluster_summary = []
        for cid in range(args.k):
            mask = holdout_labels == cid
            n_members = int(mask.sum())
            n_winners = int(holdout_is_winner[mask].sum())
            wf = n_winners / max(n_members, 1)
            holdout_cluster_summary.append({
                "cluster_id": cid,
                "split": "holdout",
                "size": n_members,
                "n_winners": n_winners,
                "winner_fraction": round(wf, 6),
                "lift_vs_base": round(wf / max(holdout_base_rate, 1e-9), 4),
            })

        # Build cluster-membership-walkforward.parquet — same schema as Track 7's
        # cluster-membership.parquet, for use by paper_sleeve_simulate.py.
        print(f"  building cluster-membership-walkforward.parquet ...")
        holdout_symbols = np.array([holdout_idx.symbols[s] for s in holdout_idx.endpoints[:, 0]])
        holdout_dates = holdout_idx.dates.astype("datetime64[D]").astype(str)
        algorithm_name = f"kmeans_k{args.k}_walkforward_cutoff_{args.cutoff.isoformat()}"
        membership_rows = []
        for i in range(len(holdout_labels)):
            membership_rows.append({
                "symbol": str(holdout_symbols[i]),
                "date": holdout_dates[i],
                "algorithm": algorithm_name,
                "k": int(args.k),
                "cluster_id": int(holdout_labels[i]),
            })
        membership_df = pl.DataFrame(membership_rows).with_columns(pl.col("date").str.to_date())
        membership_path = run_dir / "cluster-membership-walkforward.parquet"
        membership_df.write_parquet(membership_path)
        print(f"    wrote {membership_path.relative_to(_REPO_ROOT)} ({membership_df.height:,} rows)")

        # Cluster summary parquet — combines train + holdout views.
        summary_df = pl.DataFrame(train_cluster_summary + holdout_cluster_summary)
        summary_df.write_parquet(run_dir / "walkforward-cluster-summary.parquet")
        print(f"    wrote walkforward-cluster-summary.parquet")
        status.update(state="training", epoch_current=4)

        # Overlap diagnostic vs original Track 7 cluster-7.
        orig_path = args.original_cluster_membership if args.original_cluster_membership.is_absolute() \
            else (_REPO_ROOT / args.original_cluster_membership)
        if orig_path.exists():
            print(f"  computing overlap vs original Track 7 cluster {args.original_cluster_id} ...")
            orig_cm = pl.read_parquet(orig_path)
            orig_cluster_set = orig_cm.filter(pl.col("cluster_id") == args.original_cluster_id).select(
                ["symbol", "date"]
            ).unique()
            wf_cluster_set = membership_df.filter(pl.col("cluster_id") == wf_cluster_id).select(
                ["symbol", "date"]
            ).unique()

            # Two-way overlap.
            both = orig_cluster_set.join(wf_cluster_set, on=["symbol", "date"], how="inner")
            print(f"    original cluster {args.original_cluster_id}: {orig_cluster_set.height:,} (symbol, date)")
            print(f"    walk-forward cluster {wf_cluster_id}: {wf_cluster_set.height:,} (symbol, date)")
            print(f"    overlap: {both.height:,} pairs")

            # Coverage metrics.
            if orig_cluster_set.height > 0:
                pct_orig_recovered = both.height / orig_cluster_set.height
                print(f"    fraction of original recovered: {pct_orig_recovered:.4f}")
            else:
                pct_orig_recovered = 0.0
            if wf_cluster_set.height > 0:
                pct_wf_in_orig = both.height / wf_cluster_set.height
                print(f"    fraction of walk-forward in original: {pct_wf_in_orig:.4f}")
            else:
                pct_wf_in_orig = 0.0
            jaccard = both.height / max(
                (orig_cluster_set.height + wf_cluster_set.height - both.height), 1
            )
            print(f"    Jaccard overlap: {jaccard:.4f}")

            # Write the diagnostic.
            overlap_full = orig_cluster_set.with_columns(
                pl.lit(True).alias("in_original_cluster_7")
            ).join(
                wf_cluster_set.with_columns(pl.lit(True).alias("in_walkforward_cluster")),
                on=["symbol", "date"], how="full"
            ).with_columns([
                pl.col("in_original_cluster_7").fill_null(False),
                pl.col("in_walkforward_cluster").fill_null(False),
            ])
            overlap_full.write_parquet(run_dir / "overlap-vs-original-cluster-7.parquet")
            print(f"    wrote overlap-vs-original-cluster-7.parquet ({overlap_full.height:,} rows)")
        else:
            print(f"  WARNING: original cluster-membership at {orig_path} not found — skipping overlap diagnostic")
            pct_orig_recovered = None
            pct_wf_in_orig = None
            jaccard = None
            orig_cluster_set_height = 0
            wf_cluster_set_height = 0

        # Manifest.
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "encoder_d_model": int(train_embs.shape[1]),
            "cutoff_date": args.cutoff.isoformat(),
            "n_train_windows": int(train_idx.n_windows),
            "n_holdout_windows": int(holdout_idx.n_windows),
            "k": int(args.k),
            "wf_cluster_of_interest_id": int(wf_cluster_id),
            "wf_cluster_train_winner_fraction": float(wf_cluster_train_wf),
            "wf_cluster_holdout_winner_fraction": float(
                next(r["winner_fraction"] for r in holdout_cluster_summary if r["cluster_id"] == wf_cluster_id)
            ),
            "wf_cluster_holdout_lift_vs_base": float(
                next(r["lift_vs_base"] for r in holdout_cluster_summary if r["cluster_id"] == wf_cluster_id)
            ),
            "wf_cluster_holdout_size": int(
                next(r["size"] for r in holdout_cluster_summary if r["cluster_id"] == wf_cluster_id)
            ),
            "train_base_rate": round(train_base_rate, 6),
            "holdout_base_rate": round(holdout_base_rate, 6),
            "original_cluster_id_compared": int(args.original_cluster_id),
            "overlap_fraction_of_original_recovered": (
                None if pct_orig_recovered is None else round(pct_orig_recovered, 4)
            ),
            "overlap_fraction_of_wf_in_original": (
                None if pct_wf_in_orig is None else round(pct_wf_in_orig, 4)
            ),
            "overlap_jaccard": (None if jaccard is None else round(jaccard, 4)),
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote manifest.json")

        print()
        print(f"=== WALK-FORWARD CLUSTER ID RESULT ===")
        print(f"  wf-cluster-of-interest:        cluster {wf_cluster_id}")
        print(f"  train winner_fraction:         {wf_cluster_train_wf:.4f}")
        print(f"  holdout winner_fraction:       {manifest['wf_cluster_holdout_winner_fraction']:.4f}")
        print(f"  holdout lift vs base:          {manifest['wf_cluster_holdout_lift_vs_base']:.3f}x")
        if jaccard is not None:
            print(f"  overlap Jaccard vs original 7: {jaccard:.4f}")
            print(f"  fraction of original recovered:{pct_orig_recovered:.4f}")
        print(f"  wall clock:                    {wall_clock_s/60:.1f} min")
        status.record_checkpoint(epoch=4)
        status.update(state="done", epoch_current=4)
        return 0
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
