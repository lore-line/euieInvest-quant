"""Walk-forward stability check for breakout_seq_v1 CNN.

Unlike sustained_winner_walkforward (which re-scored thousands of XGB
rules per window), this just buckets the single trained CNN's val
predictions into 5 chronological windows and computes per-window val_auc
+ threshold-based precision/recall. A model whose val_auc is stable
within ±0.05 across all 5 buckets is generalizing; one whose val_auc
collapses on later windows is overfit to the train period.

Reuses `val_predictions.parquet` written by `breakout_seq_train` —
no re-training, just re-aggregation.

Output `runs/{date}-breakout_seq_v1_g20/walk_forward.parquet` (one row
per window) + `walk_forward_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from quant.tracks.sustained_winner_walkforward import WINDOWS

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "breakout_seq_walkforward_v1"
DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to runs/{date}-breakout_seq_v1_g20/ produced by breakout_seq_train.",
    )
    p.add_argument(
        "--thresholds", type=str, default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated decision thresholds for precision/recall reporting.",
    )
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    run_dir = _resolve(args.run_dir)
    val_pred_path = run_dir / "val_predictions.parquet"
    if not val_pred_path.exists():
        print(f"ERROR: {val_pred_path} not found — run breakout_seq_train first")
        return 1

    val_df = pl.read_parquet(val_pred_path)
    if "date" not in val_df.columns:
        print(f"ERROR: val_predictions.parquet missing 'date' column")
        return 1
    val_df = val_df.with_columns(pl.col("date").str.to_date().alias("date"))
    print(f"loaded {val_df.height:,} val predictions from {val_pred_path}")

    overall_scores = val_df["score"].to_numpy()
    overall_labels = val_df["label"].to_numpy()
    overall_auc = (
        float(roc_auc_score(overall_labels, overall_scores))
        if len(set(overall_labels.tolist())) > 1 else 0.5
    )
    base_rate = float(overall_labels.mean())
    print(f"overall val_auc: {overall_auc:.4f}, base_rate: {base_rate:.3f}")

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    print(f"\n  window           |   n   | base | val_auc | threshold breakdown (precision/recall/n_fires)")
    print(f"  -----------------+-------+------+---------+-----------------------------------------------")
    rows = []
    for w_idx, (w_start, w_end) in enumerate(WINDOWS):
        bucket = val_df.filter(
            (pl.col("date") >= w_start) & (pl.col("date") <= w_end)
        )
        if bucket.height == 0:
            continue
        scores = bucket["score"].to_numpy()
        labels = bucket["label"].to_numpy()
        if len(set(labels.tolist())) > 1:
            window_auc = float(roc_auc_score(labels, scores))
        else:
            window_auc = float("nan")
        window_base = float(labels.mean())

        per_threshold = []
        for tau in thresholds:
            fires = scores >= tau
            n_fires = int(fires.sum())
            if n_fires == 0:
                prec = recall = 0.0
            else:
                prec = float(labels[fires].mean())
                recall = float(((labels == 1) & fires).sum() / max(1, (labels == 1).sum()))
            per_threshold.append({
                "threshold": tau,
                "precision": prec,
                "recall": recall,
                "n_fires": n_fires,
                "lift": float(prec / base_rate) if base_rate > 0 else 0.0,
            })

        rows.append({
            "window_idx": w_idx,
            "window_start": w_start,
            "window_end": w_end,
            "n_samples": int(bucket.height),
            "n_positives": int((labels == 1).sum()),
            "base_rate": window_base,
            "val_auc": window_auc,
            "per_threshold": per_threshold,
        })

        tau75 = next(t for t in per_threshold if t["threshold"] == 0.75) if any(t["threshold"] == 0.75 for t in per_threshold) else per_threshold[len(per_threshold)//2]
        print(f"  {w_start}..{w_end} | {bucket.height:>5,} | {window_base:>4.2f} | {window_auc:>7.4f} | "
              f"@0.75: p={tau75['precision']:.3f} n={tau75['n_fires']:,} lift={tau75['lift']:.2f}")

    # Flatten per-threshold for parquet
    flat_rows = []
    for r in rows:
        for tp in r["per_threshold"]:
            flat_rows.append({
                "window_idx": r["window_idx"],
                "window_start": r["window_start"],
                "window_end": r["window_end"],
                "n_samples": r["n_samples"],
                "val_auc": r["val_auc"],
                "base_rate": r["base_rate"],
                "threshold": tp["threshold"],
                "precision": tp["precision"],
                "recall": tp["recall"],
                "n_fires": tp["n_fires"],
                "lift": tp["lift"],
            })
    pl.DataFrame(flat_rows).write_parquet(run_dir / "walk_forward.parquet")
    print(f"\nwrote {run_dir / 'walk_forward.parquet'}")

    aucs = [r["val_auc"] for r in rows if not np.isnan(r["val_auc"])]
    summary = {
        "pipeline_step": PIPELINE_STEP,
        "overall_val_auc": overall_auc,
        "base_rate": base_rate,
        "per_window_aucs": aucs,
        "auc_min": float(min(aucs)) if aucs else None,
        "auc_max": float(max(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
        "n_windows_scored": len(rows),
        "is_stable": bool((max(aucs) - min(aucs)) < 0.10) if len(aucs) > 1 else None,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (run_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote walk_forward_summary.json")
    print(f"\n=== WALK_FORWARD SUMMARY ===")
    print(f"  overall val_auc: {overall_auc:.4f}")
    print(f"  per-window AUCs: {[round(a, 3) for a in aucs]}")
    if summary["is_stable"] is not None:
        print(f"  stability (max-min < 0.10): {'PASS' if summary['is_stable'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
