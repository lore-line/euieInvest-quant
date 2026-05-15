"""Track 8 — Prototype learning (ProtoPNet-style).

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 8.

Add a prototype layer on top of the frozen Track F encoder. Train 50
prototypes with the standard ProtoPNet objective (clustering pulls
each example toward its nearest prototype; separation pushes prototypes
of different classes apart; the final classifier predicts winner from
prototype similarities). The trained prototypes are 50 specific
(symbol, date) windows the model points to as archetypal winners.

Pre-req: Track F encoder available.

Outputs:
  prototypes.parquet         — (prototype_id, symbol, date, embedding_vector,
                                n_winners_nearest, n_losers_nearest)
  prototype-windows.parquet  — raw 60×6 OHLCV for each prototype window
                                (for charting in synthesis)
  losses.parquet             — per-epoch train/val clustering+sep+CE losses

Note: ProtoPNet originally uses image patches; here a "prototype" is
a single window embedding. The clustering+separation+CE recipe
generalizes directly. Implementation uses 50 prototypes shared
across both classes (the brief calls for 50 total, not 50-per-class).

GPU-bound; ~2-3h on 5090.
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
from torch import nn
from torch.utils.data import DataLoader

from quant.data.windows import CHANNELS, WINDOW, WindowIndex, build_window_index
from quant.models.cnn_discovery import WindowDataset
from quant.tracks.embedding_clustering import _find_latest_encoder, _load_encoder
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.train import CheckpointManager, RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["PrototypeLayer", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


class PrototypeLayer(nn.Module):
    """Distance-based prototype layer over encoder embeddings.

    Holds N learnable prototype vectors of dimension D. For each input
    embedding, computes negative squared L2 distance to each prototype;
    those distances feed a linear classifier head.
    """

    def __init__(self, n_prototypes: int = 50, d_model: int = 768) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.02)
        self.classifier = nn.Linear(n_prototypes, 1)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """emb: (B, D). Returns (logits: (B,), neg_sq_dists: (B, N))."""
        # Negative squared L2 distance — high when input is close to a prototype.
        d2 = torch.cdist(emb, self.prototypes, p=2.0) ** 2  # (B, N)
        sims = -d2
        logits = self.classifier(sims).squeeze(-1)
        return logits, sims


def proto_clustering_loss(sims: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Clustering: every WINNER input must be close to AT LEAST ONE prototype.
    Loss = mean over winners of (- max similarity)."""
    winner_mask = labels.bool()
    if winner_mask.sum() == 0:
        return sims.sum() * 0.0  # zero with gradient
    return (-sims[winner_mask]).min(dim=-1).values.mean()


def proto_separation_loss(sims: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Separation: every LOSER input must be FAR from every prototype.
    Loss = mean over losers of (- min distance = - (-max sim)) = mean(max sim)."""
    loser_mask = (~labels.bool())
    if loser_mask.sum() == 0:
        return sims.sum() * 0.0
    return sims[loser_mask].max(dim=-1).values.mean()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 8 — prototype learning")
    p.add_argument("--encoder-path", type=Path, default=None)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--n-prototypes", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-cluster", type=float, default=0.8)
    p.add_argument("--lambda-sep", type=float, default=0.08)
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
    pipeline_step = "step3h_prototype_learning"
    from quant.tracks import resolve_run_dir
    run_dir, run_date_str = resolve_run_dir(
        pipeline_step=pipeline_step,
        out_dir_arg=args.out_dir,
        repo_root=_REPO_ROOT,
        resume_checkpoint_filename="latest.pt",
    )

    status = RunStatus(dir=run_dir, run_id=make_run_id(run_date_str, pipeline_step), pipeline_step=pipeline_step, epoch_total=args.epochs)
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        encoder_path = args.encoder_path or _find_latest_encoder()
        if encoder_path is None:
            raise FileNotFoundError("no Track F encoder found; run step3f_foundation_pretrain first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = _load_encoder(encoder_path, device)
        for p in encoder.parameters():
            p.requires_grad = False  # freeze
        proto = PrototypeLayer(n_prototypes=args.n_prototypes, d_model=encoder.d_model).to(device)
        opt = torch.optim.AdamW(proto.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss()
        print(f"track 8 (prototype learning) — encoder {encoder_path.relative_to(_REPO_ROOT)} (frozen)")

        labeled = pl.read_parquet(
            args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        ).filter(pl.col("is_winner").is_not_null())
        labeled, _ = _replay_feature_selection(labeled)
        train_df = labeled.filter(pl.col("date") <= args.val_end)
        val_df = labeled.filter(pl.col("date") > args.val_end)
        train_idx = build_window_index(train_df)
        val_idx = build_window_index(val_df)
        train_loader = DataLoader(WindowDataset(train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(WindowDataset(val_idx), batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
        print(f"  windows: train={train_idx.n_windows:,} val={val_idx.n_windows:,}")

        ckpt = CheckpointManager(dir=run_dir)
        loss_history = []
        for epoch in range(1, args.epochs + 1):
            if stop_flag["stop"]:
                break
            proto.train()
            ep_loss = 0.0
            n = 0
            for xb, yb in train_loader:
                if stop_flag["stop"]:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xb).mean(dim=1)
                h = h.float()
                logits, sims = proto(h)
                ce = bce(logits, yb)
                lc = proto_clustering_loss(sims, yb)
                ls = proto_separation_loss(sims, yb)
                loss = ce + args.lambda_cluster * lc + args.lambda_sep * ls
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n += 1
            train_loss = ep_loss / max(n, 1)

            # Validation: precision@top-decile on val embeddings.
            proto.eval()
            with torch.no_grad():
                val_logits = []
                val_labels = []
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        h = encoder.encode(xb).mean(dim=1)
                    logits, _ = proto(h.float())
                    val_logits.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels.append(yb.numpy())
                v_logits = np.concatenate(val_logits)
                v_labels = np.concatenate(val_labels).astype(np.int8)
                k = max(1, len(v_logits) // 10)
                top_idx = np.argpartition(-v_logits, k - 1)[:k]
                val_prec = float(v_labels[top_idx].sum() / k)

            print(f"  epoch {epoch:>2}/{args.epochs}  train_loss={train_loss:.4f}  val_prec@TD={val_prec:.4f}")
            loss_history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_prec_topd": round(val_prec, 6)})
            ckpt.save(epoch=epoch, model=proto, optimizer=opt, extras={"loss_history": loss_history})
            status.record_checkpoint(epoch=epoch)
            status.update(state="training", epoch_current=epoch)

        # Identify each prototype's archetypal window: scan train+val for nearest embedding.
        print("  finding archetypal window per prototype ...")
        proto.eval()
        prototype_vecs = proto.prototypes.detach().cpu().numpy()  # (N, D)
        all_embs: list[np.ndarray] = []
        all_meta: list[tuple[str, str, bool]] = []
        full_loader = DataLoader(WindowDataset(train_idx), batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
        symbols = train_idx.symbols
        with torch.no_grad():
            i = 0
            for xb, yb in full_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xb).mean(dim=1)
                h_np = h.float().cpu().numpy()
                all_embs.append(h_np)
                for j in range(h_np.shape[0]):
                    sym_id, local_end = train_idx.endpoints[i + j]
                    all_meta.append((
                        str(symbols[sym_id]),
                        str(train_idx.dates[i + j].astype("datetime64[D]")),
                        bool(yb[j].item()),
                    ))
                i += h_np.shape[0]
        all_embs_arr = np.concatenate(all_embs, axis=0)
        # Nearest training-window per prototype. The naive broadcast
        # `((all_embs_arr[None, :, :] - prototype_vecs[:, None, :]) ** 2).sum(axis=-1)`
        # creates a (50, 1.6M, 768) intermediate array ≈ 235 GB on a typical
        # Phase A train set; the OS OOM-kills the process. Compute pairwise
        # squared distances in CPU chunks of 8K rows × 50 prototypes (≈ 1.5 MB
        # per chunk after sum) so peak RAM stays bounded regardless of dataset
        # size. d2 result itself is (50, N_train) ≈ 320 MB for 1.6M windows —
        # fine to materialize once and reuse for the top-10 neighbor count below.
        n_train = all_embs_arr.shape[0]
        d2 = np.empty((args.n_prototypes, n_train), dtype=np.float32)
        chunk = 8192
        proto_sq = (prototype_vecs ** 2).sum(axis=1)  # (N,)
        for s in range(0, n_train, chunk):
            e = min(s + chunk, n_train)
            x = all_embs_arr[s:e]  # (C, D)
            x_sq = (x ** 2).sum(axis=1)  # (C,)
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b — faster + bounded memory.
            cross = prototype_vecs @ x.T  # (N, C)
            d2[:, s:e] = (proto_sq[:, None] + x_sq[None, :] - 2.0 * cross).astype(np.float32)
        nearest_idx = d2.argmin(axis=1)  # (N,)
        proto_rows = []
        for pid in range(args.n_prototypes):
            j = int(nearest_idx[pid])
            sym, dt, is_win = all_meta[j]
            # Top-10 nearest training windows to this prototype, count winners.
            top10 = np.argpartition(d2[pid], 10)[:10]
            n_w = int(sum(all_meta[k][2] for k in top10))
            proto_rows.append({
                "prototype_id": pid,
                "archetype_symbol": sym,
                "archetype_date": dt,
                "archetype_is_winner": is_win,
                "n_winners_in_top10_neighbors": n_w,
                "n_losers_in_top10_neighbors": 10 - n_w,
            })
        proto_df = pl.DataFrame(proto_rows)
        proto_df.write_parquet(run_dir / "prototypes.parquet")
        print(f"  wrote prototypes.parquet  ({proto_df.height} prototypes)")

        # prototype-windows.parquet: the raw 60×6 OHLCV for each prototype's archetypal window.
        proto_window_rows = []
        for pid, j in enumerate(nearest_idx):
            sym_id, local_end = train_idx.endpoints[int(j)]
            global_end = train_idx.symbol_starts[sym_id] + local_end
            w = train_idx.channels[global_end - WINDOW + 1 : global_end + 1]
            for t in range(WINDOW):
                proto_window_rows.append({
                    "prototype_id": pid,
                    "t_offset": t - (WINDOW - 1),  # 0 = day t, -59 = day t-59
                    "open": float(w[t, 0]),
                    "high": float(w[t, 1]),
                    "low": float(w[t, 2]),
                    "close": float(w[t, 3]),
                    "close_adj": float(w[t, 4]),
                    "volume": float(w[t, 5]),
                })
        pl.DataFrame(proto_window_rows).write_parquet(run_dir / "prototype-windows.parquet")
        print(f"  wrote prototype-windows.parquet  ({args.n_prototypes * WINDOW} rows)")

        pl.DataFrame(loss_history).write_parquet(run_dir / "losses.parquet")
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "n_prototypes": args.n_prototypes,
            "architecture": "frozen_encoder + prototype_layer",
            "epochs_trained": len(loss_history),
            "lambda_cluster": args.lambda_cluster,
            "lambda_sep": args.lambda_sep,
            "final_val_prec_topd": loss_history[-1]["val_prec_topd"] if loss_history else None,
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"=== TRACK 8 RESULT: {args.n_prototypes} prototypes ({wall_clock_s/60:.1f}min) ===")
        status.update(state="done", epoch_current=len(loss_history))
        return 0
    except KeyboardInterrupt:
        status.update(state="paused", epoch_current=0)
        return 130
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
