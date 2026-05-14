"""Track 7 — Embedding-space clustering.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 7.

Run the Track F pretrained encoder on the holdout windows; cluster
the resulting 768-dim embeddings with HDBSCAN; project to 2D with
UMAP for visualization. Compare against Track 2's hand-crafted-
feature clusters — clusters that survive both methods are stronger
thesis material.

Pre-req: Track F (step3f_foundation_pretrain) must have completed
and written encoder.safetensors (or .pt) to its run dir.

Outputs:
  clusters.parquet           — same schema as Track 2's clusters.parquet
  cluster-membership.parquet — same schema as Track 2's
  umap-projection.parquet    — (symbol, date, umap_x, umap_y, cluster_id, is_winner)
  clustering-comparison.md   — narrative comparing Track 2's and Track 7's clusters

CPU-bound after the GPU encoder pass (~5-10 min for encoder forward
on 624K holdout windows; clustering on 30K-row sample takes seconds).
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
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.tracks.foundation_pretrain import FoundationTransformer
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOLDOUT_EMBED_BATCH = 256


def _load_encoder(encoder_path: Path, device: torch.device) -> FoundationTransformer:
    model = FoundationTransformer().to(device)
    if encoder_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(encoder_path))
    else:
        state = torch.load(encoder_path, map_location=device, weights_only=True)
    # fp16 weights → fp32 for the model (encoder runs forward in mixed precision anyway).
    state = {k: v.to(torch.float32) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _embed_all_windows(
    model: FoundationTransformer,
    index,
    device: torch.device,
    batch_size: int = _HOLDOUT_EMBED_BATCH,
    mixed_precision: bool = True,
) -> np.ndarray:
    """Run the encoder over every window in ``index``; return (N, d_model)
    mean-pooled embeddings as float32."""
    from quant.models.cnn_discovery import WindowDataset
    from torch.utils.data import DataLoader

    ds = WindowDataset(index)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    all_embs: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
            h = model.encode(xb)
            pooled = h.mean(dim=1)
        all_embs.append(pooled.float().cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 7 — embedding clustering")
    p.add_argument("--encoder-path", type=Path, required=False, default=None,
                   help="Path to Track F's encoder.safetensors (default: latest step3f_foundation_pretrain run)")
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=200)
    p.add_argument("--hdbscan-sample-size", type=int, default=30_000)
    p.add_argument(
        "--kmeans-fallback-k",
        type=int,
        default=10,
        help=(
            "If HDBSCAN finds 0 clusters (the encoder's manifold lacks "
            "density gaps but may still carry label signal — diagnosed for "
            "Track F's encoder via scripts/diagnose_track_f_encoder.py), "
            "fall back to k-means with this many clusters. Set to 0 to "
            "disable the fallback and let the run exit with zero clusters."
        ),
    )
    p.add_argument("--umap-sample-size", type=int, default=20_000)
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


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


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step3g_embedding_clustering"
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
        encoder_path = args.encoder_path or _find_latest_encoder()
        if encoder_path is None or not encoder_path.exists():
            raise FileNotFoundError(
                "no Track F encoder found — run step3f_foundation_pretrain first or pass --encoder-path"
            )
        print(f"track 7 (embedding clustering) — encoder: {encoder_path.relative_to(_REPO_ROOT)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_encoder(encoder_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  loaded encoder: {n_params/1e6:.2f}M params on {device}")

        labeled = pl.read_parquet(
            args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        ).filter(pl.col("is_winner").is_not_null())
        labeled, _ = _replay_feature_selection(labeled)
        holdout = labeled.filter(pl.col("date") > args.val_end).sort(["symbol", "date"])
        print(f"  holdout windows: {holdout.height:,}")
        holdout_idx = build_window_index(holdout)

        print(f"  embedding {holdout_idx.n_windows:,} windows ...")
        t_emb = time.perf_counter()
        embs = _embed_all_windows(model, holdout_idx, device)
        print(f"    embedded in {time.perf_counter() - t_emb:.1f}s  → shape {embs.shape}")

        if stop_flag["stop"]:
            raise KeyboardInterrupt

        symbols = np.array([holdout_idx.symbols[s] for s in holdout_idx.endpoints[:, 0]])
        dates = holdout_idx.dates.astype("datetime64[D]").astype(str)
        is_winner_arr = holdout_idx.labels.astype(bool)

        # HDBSCAN on a 30K random sample of embeddings, then 1-NN propagate to all N.
        rng = np.random.default_rng(42)
        n = embs.shape[0]
        sample_n = min(args.hdbscan_sample_size, n)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        embs_sample = embs[sample_idx]
        print(f"  HDBSCAN (min_cluster_size={args.hdbscan_min_cluster_size}) on {sample_n} sample ...")
        t_clu = time.perf_counter()
        clusterer = HDBSCAN(min_cluster_size=args.hdbscan_min_cluster_size, metric="euclidean")
        sample_labels = clusterer.fit_predict(embs_sample)
        nn = NearestNeighbors(n_neighbors=1).fit(embs_sample)
        _, neighbor_idx = nn.kneighbors(embs)
        labels = sample_labels[neighbor_idx[:, 0]]
        n_hdbscan_clusters = len(set(labels.tolist()) - {-1})
        print(
            f"    {n_hdbscan_clusters} clusters, "
            f"{int((labels == -1).sum()):,} outliers, "
            f"{time.perf_counter() - t_clu:.1f}s"
        )

        # K-means fallback. HDBSCAN can return 0 clusters when the embedding
        # manifold is smoothly continuous without density gaps — which is the
        # case for Track F's encoder (diagnosed via
        # scripts/diagnose_track_f_encoder.py: k-means k=10 finds clusters
        # with winner_fraction spread 0.117-0.411 vs base 0.227). When HDBSCAN
        # finds nothing, fall back to k-means so the run produces usable
        # clusters.parquet instead of crashing on the downstream empty-DF
        # path. The algorithm field in clusters.parquet reflects which method
        # actually produced the clusters.
        algorithm = "hdbscan_on_encoder_embeddings"
        if n_hdbscan_clusters == 0 and args.kmeans_fallback_k > 0:
            k = args.kmeans_fallback_k
            print(f"  HDBSCAN found 0 clusters; falling back to k-means k={k} on full embedding set ...")
            t_km = time.perf_counter()
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(embs)
            algorithm = f"kmeans_k{k}_on_encoder_embeddings"
            print(f"    k-means done in {time.perf_counter() - t_km:.1f}s — {k} clusters")

        # Cluster signatures (means of embedding values per cluster — abstract,
        # but the winners-fraction-per-cluster is the meaningful signal).
        unique_clusters = sorted(set(labels.tolist()) - {-1})
        cluster_rows = []
        membership_rows = []
        for cid in unique_clusters:
            members = np.flatnonzero(labels == cid)
            n_members = len(members)
            n_winners = int(is_winner_arr[members].sum())
            winner_frac = n_winners / n_members if n_members else 0.0
            examples = [
                {"symbol": str(symbols[i]), "date": dates[i], "is_winner": bool(is_winner_arr[i])}
                for i in members[:20]
            ]
            cluster_rows.append({
                "algorithm": algorithm,
                "k": len(unique_clusters) if algorithm.startswith("kmeans") else 0,
                "cluster_id": int(cid),
                "size": n_members,
                "n_winners": n_winners,
                "winner_fraction": round(winner_frac, 6),
                "signature_features_json": json.dumps([]),  # encoder embeddings aren't named features
                "example_symbol_dates_json": json.dumps(examples),
            })
            for i in members:
                membership_rows.append({
                    "symbol": str(symbols[i]),
                    "date": dates[i],
                    "algorithm": algorithm,
                    "k": len(unique_clusters) if algorithm.startswith("kmeans") else 0,
                    "cluster_id": int(cid),
                })

        clusters_df = pl.DataFrame(cluster_rows)
        clusters_path = run_dir / "clusters.parquet"
        clusters_df.write_parquet(clusters_path)
        print(f"  wrote {clusters_path.relative_to(_REPO_ROOT)}  ({clusters_df.height} clusters)")
        # Guard against zero clusters (e.g. HDBSCAN finds nothing and
        # --kmeans-fallback-k=0). The .with_columns(date.str.to_date) call
        # below explodes on a 0-row 0-column DataFrame ("unable to find
        # column 'date'") — the original bug that crashed the first Track 7
        # run before the k-means fallback existed.
        if membership_rows:
            membership_df = pl.DataFrame(membership_rows).with_columns(pl.col("date").str.to_date())
        else:
            membership_df = pl.DataFrame(
                schema={"symbol": pl.Utf8, "date": pl.Date, "algorithm": pl.Utf8, "k": pl.Int64, "cluster_id": pl.Int64}
            )
        membership_path = run_dir / "cluster-membership.parquet"
        membership_df.write_parquet(membership_path)
        print(f"  wrote {membership_path.relative_to(_REPO_ROOT)}  ({membership_df.height:,} rows)")

        # UMAP 2D projection on a 20K sample (UMAP on 600K is slow).
        # UMAP is purely informational — used for human-eyeballing the embedding
        # space; nothing downstream consumes umap-projection.parquet. So any
        # failure here is non-fatal.
        #
        # ImportError: UMAP isn't installed (some environments skip it).
        # OSError: Python's import-cache (`_fill_cache` → `os.listdir`) can fail
        #   with EINVAL on Windows when a cloud-sync engine (Nextcloud, OneDrive,
        #   etc.) holds a transient handle on a sys.path directory.
        #   Observed: `OSError: [Errno 22] Invalid argument: '/workspace/src'`.
        # Either way, the cluster artifacts are already on disk; we don't want
        # the manifest write below to be skipped because of a visualization
        # detail.
        try:
            import umap  # type: ignore[import]
            print("  UMAP 2D projection ...")
            t_u = time.perf_counter()
            umap_n = min(args.umap_sample_size, n)
            u_idx = rng.choice(n, size=umap_n, replace=False)
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
            xy = reducer.fit_transform(embs[u_idx])
            umap_rows = [
                {
                    "symbol": str(symbols[u_idx[j]]),
                    "date": dates[u_idx[j]],
                    "umap_x": float(xy[j, 0]),
                    "umap_y": float(xy[j, 1]),
                    "cluster_id": int(labels[u_idx[j]]),
                    "is_winner": bool(is_winner_arr[u_idx[j]]),
                }
                for j in range(umap_n)
            ]
            umap_df = pl.DataFrame(umap_rows).with_columns(pl.col("date").str.to_date())
            umap_path = run_dir / "umap-projection.parquet"
            umap_df.write_parquet(umap_path)
            print(f"    UMAP in {time.perf_counter() - t_u:.1f}s → wrote {umap_path.relative_to(_REPO_ROOT)}")
        except ImportError:
            print("  UMAP skipped — install umap-learn for the umap-projection.parquet artifact")
        except OSError as exc:
            print(f"  UMAP skipped — OS-level import failure (likely cloud-sync handle on sys.path): {exc}")
        except Exception as exc:  # noqa: BLE001 — UMAP is informational; never block the run
            print(f"  UMAP skipped — unexpected error in projection: {type(exc).__name__}: {exc}")

        # Skeleton clustering-comparison.md — full narrative requires Track 2 cluster IDs to compare.
        comparison_md = run_dir / "clustering-comparison.md"
        comparison_md.write_text(
            "# Track 7 vs Track 2 clustering comparison\n\n"
            f"Track 7 ({algorithm}): "
            f"{len(unique_clusters)} clusters, {int((labels == -1).sum()):,} outliers.\n\n"
            "Cross-method comparison (Jaccard overlap of cluster memberships) — "
            "left for the synthesis stage once both tracks' cluster-membership.parquet "
            "files are available. The synthesis script joins both on (symbol, date), "
            "computes the contingency table, and identifies clusters that survive "
            "both methods (Jaccard ≥ 0.5).\n"
        )
        print(f"  wrote {comparison_md.relative_to(_REPO_ROOT)} (skeleton)")

        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "encoder_d_model": int(embs.shape[1]),
            "n_holdout_windows": int(embs.shape[0]),
            "algorithm": algorithm,
            "n_hdbscan_clusters": int(n_hdbscan_clusters),
            "n_clusters": len(unique_clusters),
            "n_outliers": int((labels == -1).sum()),
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_sample_size": sample_n,
            "kmeans_fallback_k": args.kmeans_fallback_k,
            "umap_sample_size": args.umap_sample_size,
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(f"=== TRACK 7 RESULT: {len(unique_clusters)} embedding clusters ({wall_clock_s:.1f}s) ===")
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
