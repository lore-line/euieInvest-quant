"""1D CNN training driver for breakout_seq_v1 — DL Angle 2.

Per PR #1 issuecomment-4469607665. Trains BreakoutSeqCNN on all
labelable (symbol, date) windows, splits chronologically at
2024-12-31, optimizes BCE on the breakout_seq_g20 label, tracks val_auc
each epoch, early-stops on plateau.

Output:
  runs/{date}-breakout_seq_v1_g20/
    model.pt          — final BreakoutSeqCNN state_dict
    train_log.json    — per-epoch train_loss / val_loss / val_auc
    manifest.json     — spec, hyperparams, final metrics
    val_predictions.parquet — (symbol, date, score, label) for downstream
                              walkforward + joint validation

Performance: full training on ~1.7M train rows is GPU-required
(~30-60 min). CPU smoke-test mode via --max-symbols N runs a tiny
subset for wiring validation.

Per server-team spec, the CNN is trained from scratch (no F-v2 warm-start
in v1 — architectures differ: encoder is transformer, this is CNN).
A future v2 could try transferring F-v2's pretrained embeddings as
auxiliary input, but v1 keeps things simple.
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
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.tracks.breakout_seq_label import (
    PIPELINE_STEP,
    SPEC_DEFAULT,
    BreakoutSeqSpec,
    compute_breakout_seq_label,
    label_statistics,
)
from quant.tracks.breakout_seq_model import BreakoutSeqCNN, count_parameters
from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TRAIN_CUTOFF = date(2024, 12, 31)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 3  # early stop after N epochs with no val_auc improvement


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--train-cutoff", type=date.fromisoformat, default=DEFAULT_TRAIN_CUTOFF)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument(
        "--max-symbols", type=int, default=None,
        help="CPU smoke-test: cap to N symbols (e.g. 10). Default None = all.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


# -------------------- dataset --------------------

class WindowedFeatureDataset(torch.utils.data.Dataset):
    """Yields (window: (C, T) float32, label: float32 in {0, 1}) for each
    valid endpoint in a WindowIndex.

    Caches per-symbol contiguous channel buffers in numpy; per-window
    slicing is O(C * T) = 360 floats.
    """

    def __init__(self, idx) -> None:
        self.idx = idx

    def __len__(self) -> int:
        return self.idx.n_windows

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        sym_id, row_within_sym = self.idx.endpoints[i]
        start_row_global = self.idx.symbol_starts[sym_id] + row_within_sym - (WINDOW - 1)
        end_row_global = self.idx.symbol_starts[sym_id] + row_within_sym + 1
        # idx.channels is (total_rows, n_channels); slice + transpose to (C, T)
        window = self.idx.channels[start_row_global:end_row_global].T.astype(np.float32)
        label = float(self.idx.labels[i])
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.float32)


# -------------------- training loop --------------------

def train_one_epoch(
    model: BreakoutSeqCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stop_flag: dict,
) -> float:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        if stop_flag.get("stop"):
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = bce(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: BreakoutSeqCNN, loader: DataLoader, device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (val_loss, val_auc, scores, labels)."""
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = bce(logits, yb)
        total_loss += loss.item()
        n_batches += 1
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    val_auc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else 0.5
    return total_loss / max(1, n_batches), val_auc, scores, labels


# -------------------- main --------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = SPEC_DEFAULT  # g20 fixed for v1
    features_path = _resolve(args.features)
    today = date.today().isoformat()
    out_dir = (
        _resolve(args.out_dir)
        if args.out_dir is not None
        else _REPO_ROOT / "runs" / f"{today}-{PIPELINE_STEP}_{spec.name}"
    )
    if not out_dir.parent.exists():
        alt = Path("/workspace/runs")
        if alt.exists():
            out_dir = alt / f"{today}-{PIPELINE_STEP}_{spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_step = f"{PIPELINE_STEP}_{spec.name}"
    status = RunStatus(
        dir=out_dir,
        run_id=make_run_id(today, pipeline_step),
        pipeline_step=pipeline_step,
        epoch_total=args.epochs,
    )
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        print(f"{PIPELINE_STEP} — spec {spec.name}")
        print(f"  features:    {features_path}")
        print(f"  out_dir:     {out_dir}")
        print(f"  epochs:      {args.epochs}")
        print(f"  batch_size:  {args.batch_size}  lr={args.lr}  wd={args.weight_decay}")
        if args.max_symbols:
            print(f"  max_symbols: {args.max_symbols} (SMOKE TEST MODE)")
        print()

        # Load features + compute label
        t = time.perf_counter()
        features = pl.read_parquet(features_path)
        print(f"  loaded {features.height:,} rows × {features.width} cols ({time.perf_counter()-t:.1f}s)")

        t = time.perf_counter()
        labeled = compute_breakout_seq_label(features, spec)
        label_col = spec.label_column()
        # Drop Phase A's `is_winner` if present (collision)
        if "is_winner" in labeled.columns:
            labeled = labeled.drop("is_winner")
        labeled = labeled.rename({label_col: "is_winner"}).filter(
            pl.col("is_winner").is_not_null()
        )
        print(f"  labelable: {labeled.height:,} rows ({time.perf_counter()-t:.1f}s)")
        stats = label_statistics(
            labeled.rename({"is_winner": label_col}).with_columns(
                pl.col(label_col).cast(pl.Boolean)
            ),
            spec,
        )
        print(f"  winner_rate: {100*stats['winner_rate']:.1f}%")

        # Smoke-test mode: subsample symbols
        if args.max_symbols:
            rng = np.random.default_rng(args.seed)
            all_symbols = labeled["symbol"].unique().to_list()
            keep = rng.choice(
                all_symbols,
                size=min(args.max_symbols, len(all_symbols)),
                replace=False,
            ).tolist()
            labeled = labeled.filter(pl.col("symbol").is_in(keep))
            print(f"  subsampled to {labeled.height:,} rows ({args.max_symbols} symbols)")

        # Chronological split
        train_df = labeled.filter(pl.col("date") <= args.train_cutoff)
        val_df = labeled.filter(pl.col("date") > args.train_cutoff)
        print(f"  train: {train_df.height:,} rows ({train_df.filter(pl.col('is_winner')).height:,} positive)")
        print(f"  val:   {val_df.height:,} rows ({val_df.filter(pl.col('is_winner')).height:,} positive)")

        # Build WindowIndex for each split
        t = time.perf_counter()
        train_idx = build_window_index(train_df)
        val_idx = build_window_index(val_df)
        print(f"  WindowIndex: train={train_idx.n_windows:,}, val={val_idx.n_windows:,} ({time.perf_counter()-t:.1f}s)")

        if train_idx.n_windows == 0:
            raise RuntimeError("no valid train windows — increase max_symbols or check features")

        # Model + optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device:      {device}")
        model = BreakoutSeqCNN().to(device)
        n_params = count_parameters(model)
        print(f"  model:       BreakoutSeqCNN, {n_params/1e3:.1f}K params")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

        train_ds = WindowedFeatureDataset(train_idx)
        val_ds = WindowedFeatureDataset(val_idx)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        print(f"\n  epoch | train_loss | val_loss | val_auc | wall_s")
        print(f"  ------+------------+----------+---------+-------")
        train_log: list[dict] = []
        best_val_auc = 0.0
        best_state = None
        epochs_no_improvement = 0
        best_scores = None
        best_labels = None
        for epoch in range(1, args.epochs + 1):
            if stop_flag.get("stop"):
                print(f"  graceful interrupt after epoch {epoch-1}")
                break
            t_ep = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, device, stop_flag)
            val_loss, val_auc, scores, labels = evaluate(model, val_loader, device)
            wall = time.perf_counter() - t_ep
            print(f"  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_auc:>7.4f} | {wall:>5.1f}")
            status.update(state="training", epoch_current=epoch)
            train_log.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "wall_s": round(wall, 1),
            })
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improvement = 0
                best_scores = scores.copy()
                best_labels = labels.copy()
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= args.patience:
                    print(f"  early stop: no val_auc improvement in {args.patience} epochs")
                    break

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Persist
        torch.save(model.state_dict(), out_dir / "model.pt")
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)

        # Val predictions for downstream walkforward + joint validation
        val_symbols = np.array([val_idx.symbols[s] for s in val_idx.endpoints[:, 0]])
        val_dates = val_idx.dates.astype("datetime64[D]").astype(str)
        # best_scores corresponds to the best epoch's val pass — if validate
        # ran in a different order than val_idx, we'd be misaligned. The
        # DataLoader has shuffle=False, so order matches val_idx.endpoints.
        if best_scores is None:
            # No epoch improved beyond initial — score the final model on val
            _, _, best_scores, best_labels = evaluate(model, val_loader, device)
        val_pred_df = pl.DataFrame({
            "symbol": val_symbols.tolist(),
            "date": val_dates.tolist(),
            "score": best_scores.tolist(),
            "label": best_labels.tolist(),
        })
        val_pred_df.write_parquet(out_dir / "val_predictions.parquet")

        manifest = {
            "pipeline_step": pipeline_step,
            "spec": {"name": spec.name, "touch_threshold_pct": spec.touch_threshold_pct,
                     "horizon_days": spec.horizon_days,
                     "min_entry_price_usd": spec.min_entry_price_usd},
            "features_path": str(features_path),
            "train_cutoff": args.train_cutoff.isoformat(),
            "max_symbols": args.max_symbols,
            "hyperparams": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "seed": args.seed,
            },
            "model_params_K": round(n_params / 1e3, 1),
            "device": str(device),
            "label_statistics": stats,
            "n_train_windows": int(train_idx.n_windows),
            "n_val_windows": int(val_idx.n_windows),
            "n_train_positives": int((train_idx.labels == 1).sum()),
            "n_val_positives": int((val_idx.labels == 1).sum()),
            "best_val_auc": float(best_val_auc),
            "epochs_run": len(train_log),
            "wall_clock_s": round(time.perf_counter() - t0, 1),
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        status.update(state="done", epoch_current=len(train_log))
        print()
        print(f"=== BREAKOUT_SEQ_V1 TRAINING RESULT ({spec.name}) ===")
        print(f"  best val_auc:    {best_val_auc:.4f}")
        print(f"  epochs run:      {len(train_log)}")
        print(f"  wall clock:      {manifest['wall_clock_s']}s")
        print(f"  out_dir:         {out_dir}")
    except Exception as e:
        status.update(state="failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
