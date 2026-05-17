"""DL Angle 1 — encoder-embedding clusters on the sustained-winner cohort.

For a chosen spec (default g06 — the Pareto pick), embed labelable
(symbol, date) windows with the Track F-v2 pretrained transformer
encoder, then cluster the resulting 768-dim embeddings to find
"shape-prototype" patterns that the XGB feature-threshold rules don't
capture.

Pipeline:
  1. Load features.parquet, compute sustained_winner_g{NN} label
  2. Filter to walk-forward window 2024-01-01 → 2026-03-30
  3. Build a stratified sample of N windows (~equal pos/neg by default)
  4. Embed via FoundationTransformer.encode → mean-pool over time
     → (N, 768) float32
  5. HDBSCAN cluster (KMeans fallback if HDBSCAN returns 0)
  6. Per-cluster: winner_fraction, lift, n_members
  7. Filter to winner-rich clusters (lift >= 1.5) — these are
     candidate signal patterns for the contract

Outputs to `runs/{date}-sustained_winner_emb_clusters_{spec}/`:
  - `clusters.parquet`           cluster_id, n_members, winner_fraction,
                                  lift, centroid_emb_blob
  - `cluster_membership.parquet` symbol, date, cluster_id, distance,
                                  is_winner
  - `manifest.json`              run config + metrics
  - `winner_rich_clusters.json`  the cluster_ids to consider for emit

Per-rule signal-emit path (a sibling module): for a candidate
(symbol, date), embed → 1-NN to centroid → fire if distance < tau AND
nearest centroid's cluster is winner-rich.

Performance budget:
  CPU embed 256-batch ≈ 1-2s/batch (no GPU). 30K windows = 120 batches
  ≈ 2-5 min. 200K windows = ~20-35 min. Set --sample-size to control.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.tracks.embedding_clustering import _embed_all_windows, _load_encoder
from quant.tracks.foundation_pretrain import FoundationTransformer
from quant.tracks.sustained_winner_label import (
    SPECS,
    SustainedWinnerSpec,
    compute_sustained_winner_label,
    sweep_specs,
)
from quant.tracks.sustained_winner_walkforward import (
    TRADE_WINDOW_END,
    TRADE_WINDOW_START,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "sustained_winner_emb_clusters_v1"
DEFAULT_SPEC = "g06"  # Pareto pick from XGB sweep


# -------------------- args + setup --------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--encoder-path", type=Path, default=None,
                   help="Path to FoundationTransformer encoder .pt/.safetensors. "
                        "Default: latest runs/*step3f_foundation_pretrain*/encoder.{pt,safetensors}.")
    p.add_argument("--spec", type=str, default=DEFAULT_SPEC,
                   help=f"Sustained-winner spec name (e.g. g06). Default: {DEFAULT_SPEC}.")
    p.add_argument("--sample-size", type=int, default=30_000,
                   help="Number of windows to embed (stratified pos/neg). Default 30K "
                        "(~3-5 min on CPU). Use 200000+ for full coverage when GPU is up.")
    p.add_argument("--pos-fraction", type=float, default=None,
                   help="Fraction of sample that's positive (winner). Default None = "
                        "random sampling at universe rate (honest lift; recommended). "
                        "Pass e.g. 0.5 to stratify (may inflate lift numbers if compared "
                        "against universe base_rate).")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=200)
    p.add_argument("--kmeans-fallback-k", type=int, default=12,
                   help="If HDBSCAN finds 0 clusters, fall back to KMeans with this many "
                        "clusters. Set 0 to disable fallback.")
    p.add_argument("--min-cluster-lift", type=float, default=1.5,
                   help="Clusters with winner_fraction / base_rate < this are dropped from "
                        "winner_rich_clusters.json (still kept in clusters.parquet).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


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


def _resolve_spec(name: str) -> SustainedWinnerSpec:
    if name in SPECS:
        return SPECS[name]
    for s in sweep_specs():
        if s.name == name:
            return s
    raise ValueError(f"unknown spec '{name}'")


# -------------------- label prep --------------------

def _build_labelable_frame(
    features: pl.DataFrame, spec: SustainedWinnerSpec
) -> pl.DataFrame:
    """Compute the spec's label and keep only the columns needed for
    WindowIndex (CHANNELS + symbol + date + is_winner).

    The WindowIndex hardcodes `is_winner` so we alias the sustained_winner
    label column to it.
    """
    labeled = compute_sustained_winner_label(features, spec)
    label_col = spec.label_column()
    needed = [*CHANNELS, "symbol", "date", label_col]
    missing = set(needed) - set(labeled.columns)
    if missing:
        raise KeyError(
            f"labeled frame missing columns: {sorted(missing)} "
            f"(features.parquet must contain CHANNELS={CHANNELS})"
        )
    out = labeled.select(needed).rename({label_col: "is_winner"})
    return out.filter(pl.col("is_winner").is_not_null())


def _stratified_sample_indices(
    labels: np.ndarray, target_n: int, pos_fraction: float | None, seed: int = 42,
) -> np.ndarray:
    """Sample `target_n` window indices.

    If pos_fraction is None: random sampling at universe rate (honest
    lift values when computed against universe base_rate).
    If pos_fraction is a float: stratified sampling (cleaner cluster
    signal but lift must be compared against pos_fraction, not the
    universe base_rate).
    """
    rng = np.random.default_rng(seed)
    if pos_fraction is None:
        # Pure random sample
        take = min(target_n, len(labels))
        return rng.choice(len(labels), size=take, replace=False)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    target_pos = int(round(target_n * pos_fraction))
    target_neg = target_n - target_pos
    take_pos = min(target_pos, len(pos_idx))
    take_neg = min(target_neg, len(neg_idx))
    # If we couldn't hit pos quota, oversample neg to fill (and vice versa)
    deficit = (target_pos - take_pos) + (target_neg - take_neg)
    if deficit and take_pos == len(pos_idx) and take_neg < len(neg_idx):
        take_neg = min(take_neg + deficit, len(neg_idx))
    if deficit and take_neg == len(neg_idx) and take_pos < len(pos_idx):
        take_pos = min(take_pos + deficit, len(pos_idx))
    sample_pos = rng.choice(pos_idx, size=take_pos, replace=False) if take_pos else np.array([], dtype=int)
    sample_neg = rng.choice(neg_idx, size=take_neg, replace=False) if take_neg else np.array([], dtype=int)
    out = np.concatenate([sample_pos, sample_neg])
    rng.shuffle(out)
    return out


# -------------------- clustering --------------------

def _cluster_embeddings(
    embs: np.ndarray,
    labels: np.ndarray,
    hdbscan_min_cluster_size: int,
    kmeans_fallback_k: int,
) -> tuple[np.ndarray, str, int]:
    """Cluster embeddings. Returns (cluster_labels, algorithm, n_clusters).

    HDBSCAN first; KMeans fallback if HDBSCAN finds 0 clusters (common
    for transformer encoder embeddings — manifold is smoothly continuous).
    """
    t = time.perf_counter()
    print(f"  HDBSCAN (min_cluster_size={hdbscan_min_cluster_size}) ...")
    clusterer = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size, metric="euclidean",
    )
    labs = clusterer.fit_predict(embs)
    n_hdbscan = len(set(labs.tolist()) - {-1})
    print(f"    HDBSCAN: {n_hdbscan} clusters, {int((labs == -1).sum()):,} outliers ({time.perf_counter()-t:.1f}s)")
    if n_hdbscan > 0:
        return labs, "hdbscan_on_encoder_embeddings", n_hdbscan
    if kmeans_fallback_k <= 0:
        return labs, "hdbscan_zero_clusters", 0
    print(f"    HDBSCAN found 0 clusters → KMeans k={kmeans_fallback_k} ...")
    t = time.perf_counter()
    km = KMeans(n_clusters=kmeans_fallback_k, n_init=10, random_state=42)
    labs = km.fit_predict(embs)
    print(f"    KMeans: {kmeans_fallback_k} clusters ({time.perf_counter()-t:.1f}s)")
    return labs, f"kmeans_k{kmeans_fallback_k}_on_encoder_embeddings", kmeans_fallback_k


# -------------------- main --------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    encoder_path = args.encoder_path or _find_latest_encoder()
    if encoder_path is None:
        print("ERROR: no FoundationTransformer encoder found in runs/*step3f_foundation_pretrain*/")
        return 1
    encoder_path = _resolve(encoder_path)
    if not encoder_path.exists():
        print(f"ERROR: encoder not found at {encoder_path}")
        return 1

    spec = _resolve_spec(args.spec)
    features_path = _resolve(args.features)
    today = date.today().isoformat()
    out_dir = (
        _resolve(args.out_dir)
        if args.out_dir is not None
        else _REPO_ROOT / "runs" / f"{today}-sustained_winner_emb_clusters_{spec.name}"
    )
    # Honor docker mount
    if not out_dir.parent.exists():
        alt = Path("/workspace/runs")
        if alt.exists():
            out_dir = alt / f"{today}-sustained_winner_emb_clusters_{spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sustained_winner_emb_clusters_v1 — spec {spec.name}")
    print(f"  encoder:  {encoder_path}")
    print(f"  features: {features_path}")
    print(f"  out_dir:  {out_dir}")
    pf_label = "random (universe rate)" if args.pos_fraction is None else f"stratified pos={args.pos_fraction}"
    print(f"  sample_size: {args.sample_size:,}  ({pf_label})")
    print()

    # Load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading encoder on {device} ...")
    t = time.perf_counter()
    model = _load_encoder(encoder_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {n_params/1e6:.1f}M params, loaded in {time.perf_counter()-t:.1f}s")

    # Load features + compute label
    t = time.perf_counter()
    features = pl.read_parquet(features_path)
    print(f"  loaded features {features.height:,} × {features.width} in {time.perf_counter()-t:.1f}s")
    t = time.perf_counter()
    labelable = _build_labelable_frame(features, spec)
    print(f"  labelable frame: {labelable.height:,} rows (label compute + filter {time.perf_counter()-t:.1f}s)")

    # Filter to walk-forward window
    in_window = labelable.filter(
        (pl.col("date") >= TRADE_WINDOW_START)
        & (pl.col("date") <= TRADE_WINDOW_END)
    )
    n_winners_universe = int(in_window.filter(pl.col("is_winner") == True).height)
    n_total_universe = in_window.height
    base_rate = n_winners_universe / max(1, n_total_universe)
    print(f"  walk-forward window {TRADE_WINDOW_START}..{TRADE_WINDOW_END}: "
          f"{n_total_universe:,} rows, {n_winners_universe:,} winners ({100*base_rate:.1f}%)")

    # Build window index
    t = time.perf_counter()
    idx = build_window_index(in_window)
    print(f"  WindowIndex: {idx.n_windows:,} windows ({time.perf_counter()-t:.1f}s)")

    # Stratified sample of the windows
    sample_indices = _stratified_sample_indices(
        idx.labels, args.sample_size, args.pos_fraction,
    )
    n_pos_sample = int((idx.labels[sample_indices] == 1).sum())
    n_neg_sample = len(sample_indices) - n_pos_sample
    print(f"  stratified sample: {len(sample_indices):,} windows "
          f"({n_pos_sample:,} pos / {n_neg_sample:,} neg)")

    # Build a sub-index containing ONLY the sampled windows by slicing
    # idx.endpoints / .labels / .dates. The channels buffer stays the same.
    from dataclasses import replace
    from quant.data.windows import WindowIndex
    sample_idx = WindowIndex(
        channels=idx.channels,
        symbol_starts=idx.symbol_starts,
        endpoints=idx.endpoints[sample_indices],
        labels=idx.labels[sample_indices],
        dates=idx.dates[sample_indices],
        symbols=idx.symbols,
    )

    # Embed the sample
    print(f"  embedding {sample_idx.n_windows:,} windows (batch={args.batch_size}) ...")
    t = time.perf_counter()
    embs = _embed_all_windows(model, sample_idx, device, batch_size=args.batch_size)
    embed_s = time.perf_counter() - t
    print(f"    shape {embs.shape}, {embed_s:.1f}s ({1000*embed_s/sample_idx.n_windows:.1f}ms/window)")

    # Cluster
    labels_arr = sample_idx.labels.astype(int)
    cluster_labels, algorithm, n_clusters = _cluster_embeddings(
        embs, labels_arr,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        kmeans_fallback_k=args.kmeans_fallback_k,
    )

    if n_clusters == 0:
        print("  no clusters found — exiting")
        return 2

    # Per-cluster statistics
    print(f"  computing per-cluster metrics ...")
    cluster_ids = sorted(set(cluster_labels.tolist()) - {-1})
    clusters_rows = []
    for cid in cluster_ids:
        mask = cluster_labels == cid
        n = int(mask.sum())
        n_winners_in = int((labels_arr[mask] == 1).sum())
        winner_frac = n_winners_in / n if n > 0 else 0.0
        lift = winner_frac / base_rate if base_rate > 0 else 0.0
        centroid = embs[mask].mean(axis=0).astype(np.float32)
        clusters_rows.append({
            "cluster_id": int(cid),
            "n_members": n,
            "n_winners": n_winners_in,
            "winner_fraction": float(winner_frac),
            "base_rate": float(base_rate),
            "lift": float(lift),
            "centroid_emb": centroid.tolist(),
        })

    clusters_df = pl.DataFrame(clusters_rows)
    clusters_df.write_parquet(out_dir / "clusters.parquet")
    print(f"  wrote clusters.parquet ({clusters_df.height} clusters)")

    # Persist the sample embeddings + (symbol, date) keys so the emit
    # module can compute per-cluster radii without re-embedding the sample.
    # Embeddings are (N, 768) float32 ≈ 90 MB at N=30K, 600 MB at N=200K —
    # acceptable next to the 100 MB encoder weights.
    np.savez_compressed(
        out_dir / "sample_embeddings.npz",
        embeddings=embs.astype(np.float32),
        cluster_labels=cluster_labels.astype(np.int32),
    )
    print(f"  wrote sample_embeddings.npz ({embs.nbytes // (1024*1024)} MB raw)")

    # Cluster membership for the sample (sufficient for SIGNAL emit at this scale;
    # full-universe membership requires embedding all 1M rows)
    sample_symbols = np.array([
        sample_idx.symbols[s] for s in sample_idx.endpoints[:, 0]
    ])
    sample_dates = sample_idx.dates.astype("datetime64[D]").astype(str)
    membership_df = pl.DataFrame({
        "symbol": sample_symbols.tolist(),
        "date": sample_dates.tolist(),
        "cluster_id": cluster_labels.astype(int).tolist(),
        "is_winner": labels_arr.tolist(),
    })
    membership_df.write_parquet(out_dir / "cluster_membership.parquet")
    print(f"  wrote cluster_membership.parquet ({membership_df.height:,} rows)")

    # Surface winner-rich clusters (lift >= threshold)
    winner_rich = sorted(
        [c for c in clusters_rows if c["lift"] >= args.min_cluster_lift],
        key=lambda c: c["lift"], reverse=True,
    )
    winner_rich_ids = [c["cluster_id"] for c in winner_rich]
    with open(out_dir / "winner_rich_clusters.json", "w") as f:
        json.dump({
            "min_cluster_lift": args.min_cluster_lift,
            "base_rate": base_rate,
            "winner_rich_cluster_ids": winner_rich_ids,
            "winner_rich_count": len(winner_rich),
            "clusters": [
                {k: v for k, v in c.items() if k != "centroid_emb"}
                for c in winner_rich
            ],
        }, f, indent=2)

    manifest = {
        "pipeline_step": PIPELINE_STEP,
        "spec_name": spec.name,
        "spec_touch_threshold_pct": spec.touch_threshold_pct,
        "spec_endpoint_threshold_pct": spec.endpoint_threshold_pct,
        "spec_horizon_days": spec.horizon_days,
        "encoder_path": str(encoder_path),
        "encoder_n_params_M": round(n_params / 1e6, 1),
        "device": str(device),
        "trade_window": [TRADE_WINDOW_START.isoformat(), TRADE_WINDOW_END.isoformat()],
        "n_total_universe": n_total_universe,
        "n_winners_universe": n_winners_universe,
        "base_rate": base_rate,
        "sample_size": int(sample_idx.n_windows),
        "n_pos_sample": n_pos_sample,
        "n_neg_sample": n_neg_sample,
        "embedding_dim": int(embs.shape[1]),
        "embed_wall_clock_s": round(embed_s, 1),
        "algorithm": algorithm,
        "n_clusters": int(n_clusters),
        "n_winner_rich_clusters": int(len(winner_rich)),
        "min_cluster_lift": args.min_cluster_lift,
        "total_wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"=== EMBEDDING CLUSTERS RESULT ({spec.name}) ===")
    print(f"  algorithm:           {algorithm}")
    print(f"  clusters found:      {n_clusters}")
    print(f"  winner-rich clusters (lift >= {args.min_cluster_lift}): {len(winner_rich)}")
    if winner_rich:
        print(f"  top winner-rich:")
        for c in winner_rich[:8]:
            print(f"    cluster {c['cluster_id']:>3}: "
                  f"n={c['n_members']:>5,}  "
                  f"winner_frac={100*c['winner_fraction']:>5.1f}%  "
                  f"lift={c['lift']:>4.2f}")
    print(f"  total wall clock:    {manifest['total_wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
