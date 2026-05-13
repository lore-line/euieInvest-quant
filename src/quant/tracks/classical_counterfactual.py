"""Track 6 — Classical counterfactual analysis.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 6.

For every winner in the holdout, find the 5 nearest non-winners in
the 47-dim hand-crafted feature space (cosine distance). The mean
feature delta from a winner to its 5 nearest losers tells us which
features *flip* the outcome — they're candidate causal features for
the winner pattern.

Outputs:
  winner-deltas.parquet           — per-feature mean delta, std,
                                    aggregated across all winners
  nearest-non-winners.parquet     — per-winner, list of nearest losers
                                    (deserialized JSON column)

CPU only; ~3-5 min on the ~128K holdout winners with NN-on-standardized
features. NN build is O(N log N), query is O(log N) per winner with
a KDTree on the loser pool.

Note: nearest-non-winners.parquet can be large (128K winners × 5 NNs
~ 640K rows). We keep it slim — only `(winner_symbol, winner_date,
neighbor_symbol, neighbor_date, neighbor_rank, distance)`.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.tracks import make_run_id

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _prepare_matrix(df: pl.DataFrame, feature_cols: list[str], scaler: StandardScaler | None) -> tuple[np.ndarray, StandardScaler]:
    """Convert to numpy, median-impute NaN, standardize. If scaler is
    None we fit it; otherwise we apply (consistency between winner +
    loser pools)."""
    X = df.select(feature_cols).to_numpy().astype(np.float64, copy=False)
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        medians = np.nanmedian(np.where(nan_mask, np.nan, X), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        for c in range(X.shape[1]):
            X[nan_mask[:, c], c] = medians[c]
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 6 — classical counterfactuals")
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--k", type=int, default=5, help="Number of nearest non-winners per winner.")
    p.add_argument(
        "--metric", default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance metric.",
    )
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step3e_classical_counterfactual"
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
        print(f"track 6 (classical counterfactuals) — run dir: {run_dir.relative_to(_REPO_ROOT)}")
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        if not features_path.exists():
            raise FileNotFoundError(features_path)

        labeled = pl.read_parquet(features_path).filter(pl.col("is_winner").is_not_null())
        labeled, feature_cols = _replay_feature_selection(labeled)
        holdout = labeled.filter(pl.col("date") > args.val_end).sort(["symbol", "date"])
        winners = holdout.filter(pl.col("is_winner") == True)
        losers  = holdout.filter(pl.col("is_winner") == False)
        print(
            f"  holdout winners: {winners.height:,}  losers: {losers.height:,}  "
            f"({holdout['date'].min()}→{holdout['date'].max()})"
        )

        # Fit scaler on losers (the larger pool), apply to both.
        X_losers, scaler = _prepare_matrix(losers, feature_cols, scaler=None)
        X_winners, _ = _prepare_matrix(winners, feature_cols, scaler=scaler)

        print(f"  building NN index on losers ({args.metric}, k={args.k}) ...")
        t_idx = time.perf_counter()
        nn = NearestNeighbors(n_neighbors=args.k, metric=args.metric, algorithm="brute" if args.metric == "cosine" else "auto")
        nn.fit(X_losers)
        print(f"    fit in {time.perf_counter() - t_idx:.1f}s")

        if stop_flag["stop"]:
            raise KeyboardInterrupt

        print(f"  querying nearest non-winners for {winners.height:,} winners ...")
        # Batch the query so memory stays bounded and progress is visible.
        batch = 5000
        all_dist = np.empty((winners.height, args.k), dtype=np.float32)
        all_idx = np.empty((winners.height, args.k), dtype=np.int64)
        t_q = time.perf_counter()
        for start in range(0, winners.height, batch):
            if stop_flag["stop"]:
                raise KeyboardInterrupt
            end = min(start + batch, winners.height)
            d, ii = nn.kneighbors(X_winners[start:end], n_neighbors=args.k)
            all_dist[start:end] = d.astype(np.float32, copy=False)
            all_idx[start:end] = ii
        print(f"    queried in {time.perf_counter() - t_q:.1f}s")

        # Per-feature delta: mean(winner - mean(its 5 nearest losers)) in z-space.
        # We compute on un-standardized features for human readability;
        # population_std normalizes for cross-feature comparison.
        X_winners_raw = winners.select(feature_cols).to_numpy().astype(np.float64, copy=False)
        X_losers_raw  = losers.select(feature_cols).to_numpy().astype(np.float64, copy=False)
        # Median-impute (same as _prepare_matrix would; mirror).
        for X_ in (X_winners_raw, X_losers_raw):
            nan_mask = ~np.isfinite(X_)
            if nan_mask.any():
                medians = np.nanmedian(np.where(nan_mask, np.nan, X_), axis=0)
                medians = np.where(np.isfinite(medians), medians, 0.0)
                for c in range(X_.shape[1]):
                    X_[nan_mask[:, c], c] = medians[c]

        # Per-winner: average over its k nearest losers, then delta.
        loser_means_per_winner = X_losers_raw[all_idx].mean(axis=1)  # (n_winners, n_features)
        deltas = X_winners_raw - loser_means_per_winner               # (n_winners, n_features)

        pop_std = X_winners_raw.std(axis=0) + 1e-9
        mean_delta = deltas.mean(axis=0)
        std_delta = deltas.std(axis=0)
        z_delta = mean_delta / pop_std

        # Rank features by |z_delta|.
        order = np.argsort(-np.abs(z_delta))
        delta_rows = []
        for rank, i in enumerate(order):
            delta_rows.append(
                {
                    "feature_name": feature_cols[i],
                    "rank": rank,
                    "mean_delta_winner_minus_nearest_losers": round(float(mean_delta[i]), 6),
                    "std_delta": round(float(std_delta[i]), 6),
                    "z_delta": round(float(z_delta[i]), 6),
                    "direction": "+" if z_delta[i] > 0 else "-",
                }
            )
        deltas_df = pl.DataFrame(delta_rows)
        deltas_path = run_dir / "winner-deltas.parquet"
        deltas_df.write_parquet(deltas_path)
        print(f"  wrote {deltas_path.relative_to(_REPO_ROOT)}  ({deltas_df.height} features)")
        print(f"  top-5 by |z_delta|:")
        for row in deltas_df.head(5).iter_rows(named=True):
            print(f"    {row['feature_name']:<28} z={row['z_delta']:+.3f}  ({row['direction']})")

        # Per-winner nearest-non-winners file. Slim schema.
        loser_symbols = losers["symbol"].to_numpy()
        loser_dates = losers["date"].to_numpy().astype("datetime64[D]").astype(str)
        winner_symbols = winners["symbol"].to_numpy()
        winner_dates = winners["date"].to_numpy().astype("datetime64[D]").astype(str)
        nn_rows: list[dict[str, Any]] = []
        for wi in range(winners.height):
            for rank in range(args.k):
                lj = all_idx[wi, rank]
                nn_rows.append(
                    {
                        "winner_symbol": str(winner_symbols[wi]),
                        "winner_date": winner_dates[wi],
                        "neighbor_symbol": str(loser_symbols[lj]),
                        "neighbor_date": loser_dates[lj],
                        "neighbor_rank": rank + 1,
                        "distance": float(all_dist[wi, rank]),
                    }
                )
        nn_df = pl.DataFrame(nn_rows).with_columns(
            pl.col("winner_date").str.to_date(),
            pl.col("neighbor_date").str.to_date(),
        )
        nn_path = run_dir / "nearest-non-winners.parquet"
        nn_df.write_parquet(nn_path)
        print(f"  wrote {nn_path.relative_to(_REPO_ROOT)}  ({nn_df.height:,} (winner, neighbor) pairs)")

        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "val_end": args.val_end.isoformat(),
            "holdout_start": str(holdout["date"].min()),
            "holdout_end": str(holdout["date"].max()),
            "n_winners": int(winners.height),
            "n_losers": int(losers.height),
            "k_nearest": args.k,
            "metric": args.metric,
            "feature_count": len(feature_cols),
            "runtime_device": "cpu",
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(f"=== TRACK 6 RESULT: top z_delta {deltas_df['feature_name'][0]} ({deltas_df['z_delta'][0]:+.3f})  ({wall_clock_s:.1f}s) ===")
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
