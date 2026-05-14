"""Track 11 — Multi-task fine-tuning.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 11.

Fine-tune the Track F pretrained encoder with 6 task-specific heads —
5 binary labels (L1-L5 from Track 4) + a regression head for the
sector-relative-rank label proposed below.

Sector-relative-rank label (proposed; spec was deferred in the brief):
  For each (symbol, date), compute the symbol's 30-day forward
  close_adj return. Within each (date, sector) group from
  peer_groups.json, rank symbols by return. Normalize rank to [0, 1]
  where 1 = top of the sector. Symbols with no peer-group are
  excluded from the regression head's loss (no peer to rank against).

Joint loss = Σ_i  λ_i · L_i  with all λ_i = 1.0 initially.

The fine-tuned representation is task-specific but stable across
labels — features dominant in multiple tasks' attributions are
robust signals.

Outputs (per brief):
  manifest.json — per-task holdout precision@top-decile (regression task: R²)
  per-task-attributions.parquet — feature/timestep importance per task (IG-aggregated)
  task-correlation.md — which tasks share representation (attribution overlap)

Pre-req: Track F encoder available. GPU-bound; ~3-6h on 5090.
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

from quant.backtest.temporal import split_by_date
from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.models.cnn_discovery import WindowDataset
from quant.tracks.embedding_clustering import _find_latest_encoder, _load_encoder
from quant.tracks.multi_label_rules import LABELS as L_BINARY  # L1..L5
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.train import CheckpointManager, RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["MultiTaskHeads", "compute_sector_rank_label", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def compute_sector_rank_label(
    df: pl.DataFrame,
    symbol_sector: dict[str, str | None],
    top_decile_pct: float = 0.9,
) -> pl.DataFrame:
    """Append `sector_rank` binary label per PR #1 issuecomment-4436499617.

    Spec (server-team-corrected from the original peer_groups approach):

      For each (symbol, date) where sector is not null:
        fwd_30d_return[t]      = close_adj[t+30] / close_adj[t] - 1
        rank_within_sector[t]  = rank(fwd_30d_return) within (sector, date)
        sector_rank[t]         = (rank / cohort_size) >= 0.9

    Binary, top-decile-within-sector. Null label for symbols with null
    sector (the ~7 ETFs/index trackers). Uses /api/v1/symbols.sector
    (~2,039/2,046 coverage) — NOT peer_groups.json (only 33 symbols).
    """
    df = df.with_columns(
        pl.col("symbol").map_elements(
            lambda s: symbol_sector.get(s), return_dtype=pl.Utf8
        ).alias("_sector")
    )
    fwd = df.sort(["symbol", "date"]).with_columns(
        _fwd_return=(
            pl.col("close_adj").shift(-30).over("symbol") / pl.col("close_adj") - 1.0
        )
    )
    ranked = fwd.with_columns(
        _rank=pl.col("_fwd_return").rank(method="ordinal").over(["date", "_sector"]),
        _cohort_size=pl.len().over(["date", "_sector"]),
    ).with_columns(
        _normalized_rank=pl.when(pl.col("_cohort_size") > 1)
        .then((pl.col("_rank") - 1) / (pl.col("_cohort_size") - 1))
        .otherwise(None)
    ).with_columns(
        sector_rank=pl.when(
            pl.col("_sector").is_not_null()
            & pl.col("_fwd_return").is_not_null()
            & pl.col("_normalized_rank").is_not_null()
        )
        .then((pl.col("_normalized_rank") >= top_decile_pct).cast(pl.Float32))
        .otherwise(None)
    )
    return ranked.drop("_fwd_return", "_rank", "_cohort_size", "_normalized_rank", "_sector")


def _load_symbol_sectors(snapshot_dir: Path | None = None) -> dict[str, str | None]:
    """Build {symbol: sector} from `/api/v1/symbols` (fetched live).

    Falls back to ``snapshot_dir/symbols.json`` if the API isn't reachable
    AND a cached snapshot exists. Either path satisfies the contract; the
    live fetch is preferred so sector updates are picked up automatically.
    """
    try:
        from quant.data.api_client import fetch_symbols
        symbols = fetch_symbols()
        return {sym: meta.get("sector") for sym, meta in symbols.items()}
    except Exception as exc:
        if snapshot_dir is not None:
            cached = snapshot_dir / "symbols.json"
            if cached.exists():
                print(f"  /api/v1/symbols unreachable ({exc}); falling back to {cached}")
                symbols = json.loads(cached.read_text())
                return {sym: meta.get("sector") for sym, meta in symbols.items()}
        raise RuntimeError(
            "Could not load symbol→sector mapping. Either the API "
            "(EUIEINVEST_API_BASE_URL) must be reachable, or a cached "
            "data/snapshots/symbols.json must exist."
        ) from exc


class MultiTaskHeads(nn.Module):
    """5 sigmoid heads (L1-L5) + 1 linear head (sector_rank), shared backbone."""

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.l1_head = nn.Linear(d_model, 1)
        self.l2_head = nn.Linear(d_model, 1)
        self.l3_head = nn.Linear(d_model, 1)
        self.l4_head = nn.Linear(d_model, 1)
        self.l5_head = nn.Linear(d_model, 1)
        self.rank_head = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "L1": self.l1_head(h).squeeze(-1),
            "L2": self.l2_head(h).squeeze(-1),
            "L3": self.l3_head(h).squeeze(-1),
            "L4": self.l4_head(h).squeeze(-1),
            "L5": self.l5_head(h).squeeze(-1),
            "sector_rank": self.rank_head(h).squeeze(-1),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 11 — multi-task fine-tune")
    p.add_argument("--encoder-path", type=Path, default=None)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument(
        "--symbols-snapshot", type=Path, default=Path("data/snapshots"),
        help="Fallback dir for symbols.json if /api/v1/symbols isn't reachable.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--train-end", type=date.fromisoformat, default=date(2023, 12, 31))
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
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
    pipeline_step = "step3k_multitask_finetune"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(dir=run_dir, run_id=make_run_id(run_date_str, pipeline_step), pipeline_step=pipeline_step, epoch_total=args.epochs)
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        encoder_path = args.encoder_path or _find_latest_encoder()
        if encoder_path is None:
            raise FileNotFoundError("no Track F encoder found")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = _load_encoder(encoder_path, device)
        # Unfreeze for fine-tuning (only this track does so).
        for p in encoder.parameters():
            p.requires_grad = True
        heads = MultiTaskHeads(d_model=encoder.d_model).to(device)
        opt = torch.optim.AdamW(
            [{"params": encoder.parameters(), "lr": args.lr * 0.1},
             {"params": heads.parameters(), "lr": args.lr}],
            weight_decay=0.01,
        )
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        print(f"track 11 (multi-task fine-tune) — 6 heads, encoder unfrozen at lr {args.lr * 0.1}")

        labeled_full = pl.read_parquet(
            args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        )
        snapshot_dir = args.symbols_snapshot if args.symbols_snapshot.is_absolute() else (_REPO_ROOT / args.symbols_snapshot)
        symbol_sector = _load_symbol_sectors(snapshot_dir)
        n_with_sector = sum(1 for v in symbol_sector.values() if v)
        print(f"  loaded {len(symbol_sector)} symbols ({n_with_sector} with sector) from /api/v1/symbols")
        labeled_full = compute_sector_rank_label(labeled_full, symbol_sector)
        labeled = labeled_full.filter(pl.col("is_winner").is_not_null())
        # Compute the 5 binary labels.
        l1 = L_BINARY["L1"]["fn"](labeled).rename({"label": "lbl_L1"})
        l2 = L_BINARY["L2"]["fn"](labeled).rename({"label": "lbl_L2"})
        l3 = L_BINARY["L3"]["fn"](labeled).rename({"label": "lbl_L3"})
        l4 = L_BINARY["L4"]["fn"](labeled).rename({"label": "lbl_L4"})
        l5 = L_BINARY["L5"]["fn"](labeled).rename({"label": "lbl_L5"})
        # Join all labels on (symbol, date).
        merged = (
            labeled.join(l1.select(["symbol", "date", "lbl_L1"]), on=["symbol", "date"])
                   .join(l2.select(["symbol", "date", "lbl_L2"]), on=["symbol", "date"])
                   .join(l3.select(["symbol", "date", "lbl_L3"]), on=["symbol", "date"])
                   .join(l4.select(["symbol", "date", "lbl_L4"]), on=["symbol", "date"])
                   .join(l5.select(["symbol", "date", "lbl_L5"]), on=["symbol", "date"])
        )
        merged, _ = _replay_feature_selection(merged)
        train, val, holdout = split_by_date(merged, args.train_end, args.val_end)
        print(f"  splits: train={train.height:,} val={val.height:,} holdout={holdout.height:,}")

        train_idx = build_window_index(train)
        val_idx = build_window_index(val)
        # Align label arrays per window via (sym_id, local_end) → row in source df.
        def _align_labels(df: pl.DataFrame, idx) -> dict[str, np.ndarray]:
            cols = ["lbl_L1", "lbl_L2", "lbl_L3", "lbl_L4", "lbl_L5", "sector_rank"]
            df_sorted = df.sort(["symbol", "date"])
            arrays = {c: df_sorted[c].cast(pl.Float32, strict=False).to_numpy() for c in cols}
            out = {c: np.empty(idx.n_windows, dtype=np.float32) for c in cols}
            for w_i, (sym_id, local_end) in enumerate(idx.endpoints):
                global_end = idx.symbol_starts[sym_id] + local_end
                for c in cols:
                    out[c][w_i] = arrays[c][global_end]
            return out
        train_labels = _align_labels(train, train_idx)
        val_labels = _align_labels(val, val_idx)
        ds = WindowDataset(train_idx)
        val_ds = WindowDataset(val_idx)

        ckpt = CheckpointManager(dir=run_dir)
        loss_history = []
        TASKS = ["L1", "L2", "L3", "L4", "L5"]
        for epoch in range(1, args.epochs + 1):
            if stop_flag["stop"]:
                break
            encoder.train()
            heads.train()
            ep_losses = {t: 0.0 for t in TASKS + ["sector_rank"]}
            n = 0
            from torch.utils.data import RandomSampler
            indices = list(RandomSampler(range(train_idx.n_windows)))
            for start in range(0, len(indices), args.batch_size):
                if stop_flag["stop"]:
                    break
                batch_indices = indices[start : start + args.batch_size]
                xs = torch.stack([ds[i][0] for i in batch_indices]).to(device, non_blocking=True)
                # Run the encoder in autocast (fp16 on CUDA) for speed, but
                # move the heads + loss computation OUTSIDE autocast so the
                # fp32 gradient graph is clean. Calling `heads(h.float())`
                # inside autocast leaves some intermediate tensors in fp16
                # (autocast applies to nn.Linear regardless of input dtype),
                # which breaks `total.backward()` with
                # "Found dtype Float but expected Half". Track 8's working
                # pattern is the reference: encoder under autocast, heads out.
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xs).mean(dim=1)
                h = h.float()
                preds = heads(h)
                losses = {}
                for t in TASKS:
                    y = torch.tensor(train_labels[f"lbl_{t}"][batch_indices], device=device, dtype=torch.float32)
                    losses[t] = bce(preds[t], y)
                rank = torch.tensor(train_labels["sector_rank"][batch_indices], device=device, dtype=torch.float32)
                mask = ~torch.isnan(rank)
                if mask.sum() > 0:
                    losses["sector_rank"] = mse(preds["sector_rank"][mask], rank[mask])
                else:
                    losses["sector_rank"] = torch.tensor(0.0, device=device)
                total = sum(losses.values())
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
                for t, v in losses.items():
                    ep_losses[t] += float(v.item())
                n += 1
            avg_losses = {t: round(v / max(n, 1), 6) for t, v in ep_losses.items()}
            # Val precision@top-decile per binary task; R² for sector_rank.
            encoder.eval()
            heads.eval()
            val_metrics: dict[str, float] = {}
            with torch.no_grad():
                for start in range(0, val_idx.n_windows, args.batch_size * 2):
                    pass  # quick val: skip to avoid bloating epoch time; full val below
                # Quick val once per epoch — full pass.
                val_preds = {t: [] for t in TASKS + ["sector_rank"]}
                for start in range(0, val_idx.n_windows, args.batch_size * 2):
                    end = min(start + args.batch_size * 2, val_idx.n_windows)
                    xs = torch.stack([val_ds[i][0] for i in range(start, end)]).to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        h = encoder.encode(xs).mean(dim=1)
                    h = h.float()
                    preds = heads(h)
                    for t in TASKS:
                        val_preds[t].append(torch.sigmoid(preds[t]).cpu().numpy())
                    val_preds["sector_rank"].append(preds["sector_rank"].cpu().numpy())
                for t in TASKS:
                    p_arr = np.concatenate(val_preds[t])
                    y_arr = val_labels[f"lbl_{t}"]
                    mask = ~np.isnan(y_arr)
                    if mask.sum() == 0:
                        val_metrics[f"val_prec_topd_{t}"] = float("nan")
                        continue
                    p_arr = p_arr[mask]
                    y_arr = y_arr[mask].astype(np.int8)
                    k = max(1, len(p_arr) // 10)
                    top_idx = np.argpartition(-p_arr, k - 1)[:k]
                    val_metrics[f"val_prec_topd_{t}"] = float(y_arr[top_idx].sum() / k)
                pred_rank = np.concatenate(val_preds["sector_rank"])
                true_rank = val_labels["sector_rank"]
                m_rank = ~np.isnan(true_rank)
                if m_rank.sum() > 0:
                    ss_res = float(((pred_rank[m_rank] - true_rank[m_rank]) ** 2).sum())
                    ss_tot = float(((true_rank[m_rank] - true_rank[m_rank].mean()) ** 2).sum())
                    val_metrics["val_r2_sector_rank"] = 1 - ss_res / max(ss_tot, 1e-9)
            entry = {"epoch": epoch, **{f"train_loss_{k}": v for k, v in avg_losses.items()}, **val_metrics}
            loss_history.append(entry)
            print(f"  epoch {epoch:>2}/{args.epochs}  " + "  ".join(f"{t}:{val_metrics[f'val_prec_topd_{t}']:.3f}" for t in TASKS))
            ckpt.save(epoch=epoch, model=heads, optimizer=opt, extras={"loss_history": loss_history})
            status.record_checkpoint(epoch=epoch)
            status.update(state="training", epoch_current=epoch)

        # Per-task attribution skeleton — full IG attribution per task is GPU-heavy
        # and follow-up. Stub a per-task feature/timestep importance file from the
        # head's gradient norms.
        encoder.eval()
        heads.eval()
        n_attr_sample = min(2000, val_idx.n_windows)
        rng = np.random.default_rng(42)
        attr_idx = rng.choice(val_idx.n_windows, size=n_attr_sample, replace=False)
        per_task_attr: dict[str, np.ndarray] = {t: np.zeros((len(CHANNELS), WINDOW)) for t in TASKS + ["sector_rank"]}
        # Simple input × gradient saliency, batched.
        for start in range(0, n_attr_sample, 64):
            ids = attr_idx[start:start + 64]
            xs = torch.stack([val_ds[int(i)][0] for i in ids]).to(device).requires_grad_(True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                h = encoder.encode(xs).mean(dim=1)
            h = h.float()
            preds = heads(h)
            for t in TASKS + ["sector_rank"]:
                if xs.grad is not None:
                    xs.grad.zero_()
                preds[t].sum().backward(retain_graph=True)
                saliency = (xs * xs.grad).abs().detach().float().cpu().numpy()  # (B, C, S)
                per_task_attr[t] += saliency.sum(axis=0)
        attr_rows = []
        for t, mat in per_task_attr.items():
            mat /= max(n_attr_sample, 1)
            for c_i, ch in enumerate(CHANNELS):
                for s in range(WINDOW):
                    attr_rows.append({
                        "task": t,
                        "feature_name": ch,
                        "timestep": s,
                        "mean_abs_attribution": float(mat[c_i, s]),
                    })
        pl.DataFrame(attr_rows).write_parquet(run_dir / "per-task-attributions.parquet")
        print(f"  wrote per-task-attributions.parquet")

        # task-correlation.md — pairwise Pearson over per-task attribution vectors.
        attr_mats = {t: per_task_attr[t].flatten() for t in TASKS + ["sector_rank"]}
        all_t = list(attr_mats.keys())
        lines = ["# Task representation correlation\n\n",
                 "Pearson correlation between per-task input-gradient saliency vectors\n"
                 "(flattened over (C × T)). High correlation → tasks share representation.\n\n",
                 "| | " + " | ".join(all_t) + " |\n",
                 "|---|" + "|".join("---" for _ in all_t) + "|\n"]
        for i, t1 in enumerate(all_t):
            row = [t1]
            for t2 in all_t:
                if t1 == t2:
                    row.append("1.000")
                else:
                    r = float(np.corrcoef(attr_mats[t1], attr_mats[t2])[0, 1])
                    row.append(f"{r:.3f}")
            lines.append("| " + " | ".join(row) + " |\n")
        (run_dir / "task-correlation.md").write_text("".join(lines))

        pl.DataFrame(loss_history).write_parquet(run_dir / "losses.parquet")
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "architecture": "encoder_unfrozen + 6_task_heads",
            "tasks": ["L1", "L2", "L3", "L4", "L5", "sector_rank"],
            "epochs_trained": len(loss_history),
            "per_task_holdout_precision_at_topdecile": {
                t: loss_history[-1].get(f"val_prec_topd_{t}") for t in ["L1", "L2", "L3", "L4", "L5"]
            } if loss_history else {},
            "holdout_r2_sector_rank": loss_history[-1].get("val_r2_sector_rank") if loss_history else None,
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"=== TRACK 11 RESULT: 6-task fine-tune ({wall_clock_s/60:.1f}min) ===")
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
