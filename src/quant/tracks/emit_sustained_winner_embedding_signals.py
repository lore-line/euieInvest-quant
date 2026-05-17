"""Emit ENTRY signals from encoder-embedding clusters — DL Angle 1 emit path.

For each candidate (symbol, date) in the emission window:
  1. Build a 60-day window of OHLCV ending at `date`
  2. Embed via the same FoundationTransformer used to build clusters
  3. Find the nearest cluster centroid (by L2 distance)
  4. If nearest is a WINNER-RICH cluster (lift >= min_cluster_lift) AND
     the distance is within the cluster's "tightness" radius (typically
     the 90th percentile of within-cluster distances), emit a signal:
       pattern = "sw1_{spec}_emb_cl{cluster_id}"
       expected_return_pct = cluster's empirical mean realized return
       signal_strength = clamp(lift / 3.0, 0, 1)
  5. Per-(symbol, pattern) 30-day dedup, same as rule-emit

Inputs:
  - `runs/{date}-sustained_winner_emb_clusters_{spec}/clusters.parquet`
    (centroid_emb, winner_fraction, lift, n_members)
  - `runs/{date}-sustained_winner_emb_clusters_{spec}/cluster_membership.parquet`
    (used to derive cluster_radius_p90 and per-cluster expected_return_pct)
  - encoder.pt (must match the one used to build the clusters)
  - features.parquet (for the candidate (symbol, date) window OHLCV)

Output: same contract-v1 schema as `emit_sustained_winner_signals`.

Note: This is the DL companion to `emit_sustained_winner_signals` (XGB
rule track). Both can emit alongside each other — they're distinguished
by pattern prefix (`sw1_g06_rule_{id}` vs `sw1_g06_emb_cl{K}`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.tracks.embedding_clustering import _embed_all_windows, _load_encoder
from quant.tracks.emit_quant_signals import _CONTRACT_SCHEMA, _apply_dedup, _next_sequence, _validate_signal_row
from quant.tracks.sustained_winner_label import SPECS, sweep_specs
from quant.tracks.sustained_winner_walkforward import _label_features_once

_REPO_ROOT = Path(__file__).resolve().parents[3]

CONTRACT_VERSION = "v1"
PIPELINE_STEP = "sustained_winner_emb_signal_emission_v1"
DEFAULT_DEDUP_WINDOW_DAYS = 30
DEFAULT_MIN_CLUSTER_LIFT = 1.5
DEFAULT_LIFT_STRENGTH_DIVISOR = 3.0
# Distance percentile from cluster_membership that defines "close enough"
# to fire (e.g. 0.90 → fire if candidate's distance to centroid is within
# the cluster's 90th-percentile of member distances).
DEFAULT_DISTANCE_PERCENTILE = 0.90


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument(
        "--clusters-dir", type=Path, required=True,
        help="Path to runs/{date}-sustained_winner_emb_clusters_{spec}/ directory.",
    )
    p.add_argument("--encoder-path", type=Path, default=None,
                   help="FoundationTransformer encoder. Default: latest step3f run.")
    p.add_argument("--signal-date", type=date.fromisoformat, default=None)
    p.add_argument("--backfill-days", type=int, default=0)
    p.add_argument("--dedup-window-days", type=int, default=DEFAULT_DEDUP_WINDOW_DAYS)
    p.add_argument("--min-cluster-lift", type=float, default=DEFAULT_MIN_CLUSTER_LIFT)
    p.add_argument("--strength-divisor", type=float, default=DEFAULT_LIFT_STRENGTH_DIVISOR)
    p.add_argument("--distance-percentile", type=float, default=DEFAULT_DISTANCE_PERCENTILE,
                   help="Candidate fires if dist to nearest centroid <= cluster's "
                        "Pth-percentile of member-distance. Default 0.90.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-sequence", type=int, default=None)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_latest_encoder() -> Path | None:
    runs = sorted(_REPO_ROOT.glob("runs/*step3f_foundation_pretrain*"))
    if not runs:
        return None
    for d in reversed(runs):
        for fname in ("encoder.safetensors", "encoder.pt"):
            p = d / fname
            if p.exists():
                return p
    return None


def _compute_cluster_radii(
    membership_df: pl.DataFrame,
    embs: np.ndarray,
    clusters_df: pl.DataFrame,
    distance_percentile: float,
) -> dict[int, dict]:
    """For each cluster, compute the Pth-percentile distance from member
    embeddings to the centroid. This becomes the "fire radius" tau.

    Also computes the cluster's empirical mean realized return — used as
    expected_return_pct in emitted signals. Requires membership_df to
    have a `forward_endpoint_pct` column (joined from labeled features).
    """
    centroids = {
        int(row["cluster_id"]): np.asarray(row["centroid_emb"], dtype=np.float32)
        for row in clusters_df.iter_rows(named=True)
    }
    member_cluster_ids = membership_df["cluster_id"].to_numpy()

    out: dict[int, dict] = {}
    for cid in centroids:
        if cid == -1:  # HDBSCAN outlier
            continue
        member_mask = member_cluster_ids == cid
        if not member_mask.any():
            continue
        c_embs = embs[member_mask]
        centroid = centroids[cid]
        dists = np.linalg.norm(c_embs - centroid[None, :], axis=1)
        radius = float(np.quantile(dists, distance_percentile))
        # Mean realized return — fall back to NaN if membership doesn't
        # have endpoint information
        mean_endpoint = float("nan")
        if "forward_endpoint_pct" in membership_df.columns:
            endp = membership_df.filter(pl.col("cluster_id") == cid)["forward_endpoint_pct"]
            if endp.len() > 0:
                mean_endpoint = float(endp.mean())
        out[cid] = {
            "centroid": centroid,
            "radius": radius,
            "n_members": int(member_mask.sum()),
            "mean_endpoint_pct": mean_endpoint,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    encoder_path = _resolve(args.encoder_path) if args.encoder_path else _find_latest_encoder()
    if encoder_path is None or not encoder_path.exists():
        print(f"ERROR: encoder not found ({encoder_path})")
        return 1
    clusters_dir = _resolve(args.clusters_dir)
    if not clusters_dir.exists():
        print(f"ERROR: clusters_dir not found: {clusters_dir}")
        return 1

    # Load cluster artifacts
    clusters_df = pl.read_parquet(clusters_dir / "clusters.parquet")
    membership_df = pl.read_parquet(clusters_dir / "cluster_membership.parquet")
    with open(clusters_dir / "manifest.json") as f:
        cluster_manifest = json.load(f)
    spec_name = cluster_manifest["spec_name"]
    spec = next(
        (s for s in sweep_specs() if s.name == spec_name),
        SPECS.get(spec_name),
    )
    if spec is None:
        raise ValueError(f"unknown spec '{spec_name}' from manifest")
    horizon = int(cluster_manifest["spec_horizon_days"])

    winner_rich = clusters_df.filter(pl.col("lift") >= args.min_cluster_lift)
    n_winner_rich = winner_rich.height
    print(f"sustained-winner embedding emit v{CONTRACT_VERSION} — spec {spec_name}")
    print(f"  clusters_dir:       {clusters_dir}")
    print(f"  encoder:            {encoder_path}")
    print(f"  total clusters:     {clusters_df.height}")
    print(f"  winner-rich (lift >= {args.min_cluster_lift}): {n_winner_rich}")
    if n_winner_rich == 0:
        print("  no winner-rich clusters — nothing to emit")
        return 0

    # Enrich membership with forward_endpoint_pct so we can compute per-
    # cluster expected_return. Requires re-loading features + label.
    features_path = _resolve(args.features)
    features_raw = pl.read_parquet(features_path)
    labeled = _label_features_once(features_raw, horizon)
    end_lookup = labeled.select(["symbol", "date", "forward_endpoint_pct"])
    # membership_df.date is a string; cast for join
    membership_df = membership_df.with_columns(
        pl.col("date").str.to_date().alias("date")
    )
    membership_df = membership_df.join(end_lookup, on=["symbol", "date"], how="left")
    print(f"  membership rows: {membership_df.height:,} (after endpoint join)")

    # We need the embeddings to compute per-cluster radii. Re-embed the
    # sample (deterministic — same encoder + same sample selection).
    # For now we cheat: instead of re-embedding, we compute radii using
    # the centroid_emb from clusters.parquet and the sample's cluster
    # assignment via TRUE within-cluster distances. But we don't HAVE
    # the embeddings persisted — clusters.parquet only has centroids.
    #
    # Two options:
    #  (a) Persist sample embeddings to disk in the cluster module
    #  (b) Re-embed the sample here
    #
    # For now we do (b) — small cost (~5 min for 30K sample on CPU).
    # PRODUCTION TODO: persist embeddings to clusters_dir/sample_embeddings.npz
    # so emit is fast on subsequent runs.
    print(f"  re-embedding sample to compute cluster radii ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_encoder(encoder_path, device)
    # Rebuild WindowIndex over the SAMPLE rows
    sample_rows = membership_df.select(["symbol", "date"]).rename({})
    # Get OHLCV for those (symbol, date) tuples from features
    needed = [*CHANNELS, "symbol", "date"]
    feat_subset = features_raw.select(needed)
    # Mark which rows are in the sample
    membership_keys = set(
        (r["symbol"], r["date"]) for r in membership_df.iter_rows(named=True)
    )
    # Have to dummy-add is_winner so WindowIndex doesn't complain
    feat_subset = feat_subset.with_columns(pl.lit(0).cast(pl.Int8).alias("is_winner"))
    # Build full index, then filter endpoints to just our sample
    full_idx = build_window_index(feat_subset)
    # The membership has window endpoints — find which positional indices
    # in full_idx correspond. Build lookup.
    print(f"  WARNING: re-embedding currently skipped — pipeline plumbing TODO. "
          f"Cluster radii fall back to placeholder.")
    # PLACEHOLDER: use a wide radius so the path runs end-to-end.
    # Real impl needs to either (i) persist embeddings in cluster module
    # or (ii) implement the WindowIndex slice + re-embed pattern.
    cluster_info: dict[int, dict] = {}
    centroids_lookup = {
        int(r["cluster_id"]): np.asarray(r["centroid_emb"], dtype=np.float32)
        for r in clusters_df.iter_rows(named=True)
    }
    for cid in centroids_lookup:
        cluster_rows = membership_df.filter(pl.col("cluster_id") == cid)
        if cluster_rows.height == 0:
            continue
        endpoints = cluster_rows["forward_endpoint_pct"].drop_nulls()
        mean_endpoint = float(endpoints.mean()) if endpoints.len() > 0 else float("nan")
        cluster_info[cid] = {
            "centroid": centroids_lookup[cid],
            "radius": float("inf"),  # PLACEHOLDER — see TODO above
            "n_members": int(cluster_rows.height),
            "mean_endpoint_pct": mean_endpoint,
            "lift": float(clusters_df.filter(pl.col("cluster_id") == cid)["lift"][0]),
        }

    winner_rich_ids = set(winner_rich["cluster_id"].to_list())
    print(f"  cluster_info populated for {len(cluster_info)} clusters "
          f"({len(winner_rich_ids)} winner-rich)")
    print()
    print("=== EMBEDDING-EMIT PIPELINE (scaffolded, not production-ready) ===")
    print(f"   total wall clock: {time.perf_counter() - t0:.1f}s")
    print()
    print("TODOs to make this emit production signals:")
    print(" 1. Persist sample embeddings in sustained_winner_embedding_clusters")
    print("    (write sample_embeddings.npz next to clusters.parquet)")
    print(" 2. Implement candidate (symbol, date) → 60-day window OHLCV → embed → ")
    print("    1-NN-centroid lookup + radius gate in this module")
    print(" 3. Choose distance_percentile tau via validation (likely 0.50-0.75)")
    print(" 4. Write contract rows + dedup + manifest + parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
