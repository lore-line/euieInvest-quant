"""K-means diagnostic on Track F's encoder embeddings.

Purpose: when HDBSCAN (Track 7) finds no density structure, this script
answers the critical follow-up — does the encoder embedding space have
ANY structure correlated with the label, or is it hollow?

K-means doesn't require density gaps; it just partitions space by
proximity. If the encoder learned useful representations:
  - We expect k-means clusters to have varying winner_fraction
    (e.g. 0.10 -- 0.30 spread on a 0.20 holdout base rate).
If the encoder is hollow (random):
  - All k-means clusters will have winner_fraction ~= base rate
    (0.20 +/- noise).

Reads:
  data/features/features.parquet  (uses default --features)
  D:/quant-runs/2026-05-13-step3f_foundation_pretrain/encoder.pt (default)

Writes (small + cheap):
  D:/quant-runs/encoder-diagnostic/kmeans-summary.parquet

Cost: ~30s GPU (re-embed) + ~1 min CPU (k-means k=5,8,10).

Usage:
    docker compose run --rm dev python scripts/diagnose_track_f_encoder.py
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
from sklearn.cluster import KMeans

from quant.data.windows import build_window_index
from quant.tracks.embedding_clustering import _embed_all_windows, _load_encoder
from quant.tracks.xgb_rule_extraction import _replay_feature_selection

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-means diagnostic on Track F embeddings")
    p.add_argument(
        "--encoder-path",
        type=Path,
        default=_REPO_ROOT / "runs" / "2026-05-13-step3f_foundation_pretrain" / "encoder.pt",
    )
    p.add_argument("--features", type=Path, default=_REPO_ROOT / "data" / "features" / "features.parquet")
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "encoder-diagnostic",
    )
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 8, 10],
        help="k values to try for k-means",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"encoder: {args.encoder_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_encoder(args.encoder_path, device)
    print(f"  loaded {sum(p.numel() for p in model.parameters())/1e6:.2f}M params on {device}")

    print(f"features: {args.features}")
    labeled = pl.read_parquet(args.features).filter(pl.col("is_winner").is_not_null())
    labeled, _ = _replay_feature_selection(labeled)
    holdout = labeled.filter(pl.col("date") > args.val_end).sort(["symbol", "date"])
    print(f"  holdout windows: {holdout.height:,}")

    holdout_idx = build_window_index(holdout)
    print(f"  building {holdout_idx.n_windows:,} windows ...")

    t_emb = time.perf_counter()
    embs = _embed_all_windows(model, holdout_idx, device)
    print(f"  embedded in {time.perf_counter() - t_emb:.1f}s -> shape {embs.shape}")

    is_winner = holdout_idx.labels.astype(bool)
    base_rate = float(is_winner.mean())
    print(f"  base winner rate: {base_rate:.4f}  ({int(is_winner.sum()):,} of {len(is_winner):,})")

    # K-means per k value. Reports winner_fraction per cluster + spread metrics.
    summary_rows: list[dict] = []
    for k in args.k_values:
        print()
        print(f"k-means k={k} ...")
        t_k = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embs)
        print(f"  fit in {time.perf_counter() - t_k:.1f}s")

        cluster_winner_fractions = []
        for cid in range(k):
            mask = labels == cid
            n_members = int(mask.sum())
            n_winners = int(is_winner[mask].sum())
            wf = n_winners / n_members if n_members > 0 else 0.0
            cluster_winner_fractions.append(wf)
            summary_rows.append(
                {
                    "k": k,
                    "cluster_id": int(cid),
                    "size": n_members,
                    "n_winners": n_winners,
                    "winner_fraction": round(wf, 6),
                    "lift_vs_base": round(wf / base_rate, 4) if base_rate > 0 else 0.0,
                }
            )

        wfs = np.array(cluster_winner_fractions)
        print(f"  winner_fractions per cluster (k={k}):")
        for cid, wf in enumerate(sorted(zip(range(k), wfs), key=lambda x: x[1])):
            cid_, wf_ = wf
            mask = labels == cid_
            print(f"    cluster {cid_}: size={int(mask.sum()):>7,}  winner_frac={wf_:.4f}  lift={wf_/base_rate:.3f}x")

        print(f"  spread: min={wfs.min():.4f}  max={wfs.max():.4f}  range={wfs.max() - wfs.min():.4f}")
        print(f"          stdev={wfs.std():.4f}  ratio(max/min)={wfs.max()/max(wfs.min(),1e-9):.2f}")

    summary_df = pl.DataFrame(summary_rows)
    out_path = args.out_dir / "kmeans-summary.parquet"
    summary_df.write_parquet(out_path)
    print()
    print(f"wrote {out_path}")

    meta = {
        "encoder_path": str(args.encoder_path),
        "n_windows": int(embs.shape[0]),
        "embedding_dim": int(embs.shape[1]),
        "base_winner_rate": round(base_rate, 6),
        "k_values": args.k_values,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (args.out_dir / "diagnostic-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out_dir / 'diagnostic-meta.json'}")

    print()
    print("=== INTERPRETATION GUIDE ===")
    print("  Encoder is HOLLOW if:    all clusters have winner_fraction ≈ base_rate (±2pp)")
    print("                            i.e. max-min spread < 0.04")
    print("  Encoder is USABLE if:    at least one cluster reaches winner_fraction >= 0.25")
    print("                            (well above base_rate); confirms downstream tracks are worth running")
    print("  Encoder is STRONG if:    max-min spread >= 0.15 across all k values")
    print(f"  Base rate for reference: {base_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
