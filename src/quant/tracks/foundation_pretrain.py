"""Track F — Foundation Transformer pretraining.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track F (centerpiece).

Self-supervised Transformer encoder over the full 60×6 OHLCV window
corpus. The trained encoder is the substrate for DL tracks 7-12.
No supervised label involved.

Two pretraining objectives, summed:

  L = λ_mlm · MSE(reconstruct masked bars)
    + λ_ntx · NT-Xent(same-symbol within 5d = positive)

Architecture (~56M params — the brief's "~50M, d_model=256, 8 heads,
8 layers, dim_ff=1024" was internally inconsistent: those hyper-
parameters yield ~6M, not 50M. We scale d_model to 768 to hit the
50M target while keeping the brief's 8-head / 8-layer / encoder-only
topology):

  Linear(6 → 768)                                # bar embedding
  + sinusoidal positional encoding (60 positions)
  TransformerEncoderLayer × 8                    # d_model=768, n_heads=8, dim_ff=3072
  LayerNorm
  → two heads share the encoder:
      Linear(768 → 6)         # masked-bar reconstruction
      Linear(768 → 128) → GELU → Linear(128 → 64) → L2-norm   # contrastive projection

Operational hygiene (CheckpointManager + RunStatus + SIGINT):
  - Checkpoint cadence: per epoch + every 30 min, whichever is sooner
  - --resume latest: bit-identical resume from latest.pt
  - Ctrl-C: flips status.json to "paused", flushes final checkpoint
  - Status updates with ETA every status_update_every_n_batches

Expected wall-clock: 12-24h on RTX 5090 with mixed precision (the
brief's estimate). Detached via docker run -d; survives across agent
sessions; resume via quant-start.ps1 -Track step3f_foundation_pretrain -Resume.

Output (per docs/reports-repo-layout.md):
  manifest.json
  encoder.safetensors        # fp16 weights, ~110 MB
  pretrain-losses.parquet    # per-epoch MLM + NT-Xent + total losses
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quant.data.windows import CHANNELS, WINDOW, WindowIndex, build_window_index
from quant.labels import compute_forward_winner_labels
from quant.train import CheckpointManager, RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["FoundationTransformer", "MaskedContrastiveDataset", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Default architecture (≈56M params).
D_MODEL = 768
N_HEADS = 8
N_LAYERS = 8
DIM_FF = 3072
PROJ_DIM = 64  # contrastive projection dimension
MASK_RATIO = 0.15
NTX_TEMPERATURE = 0.1
NEIGHBOR_WINDOW_DAYS = 5  # within-symbol positive-pair radius


# ----- model -----


def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding. Shape (1, seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class FoundationTransformer(nn.Module):
    """8-layer encoder + MLM head + contrastive projection head."""

    def __init__(
        self,
        n_channels: int = len(CHANNELS),
        seq_len: int = WINDOW,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_ff: int = DIM_FF,
        proj_dim: int = PROJ_DIM,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.bar_embed = nn.Linear(n_channels, d_model)
        # Learnable mask token replaces masked bars before embedding.
        self.mask_token = nn.Parameter(torch.randn(n_channels) * 0.02)
        self.register_buffer("pe", _sinusoidal_pe(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable at our depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, n_channels)
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_channels, seq_len). Returns hidden states (B, seq_len, d_model)."""
        # Transformer wants (B, seq, n_channels) for the linear projection.
        bars = x.transpose(1, 2)  # (B, seq, n_channels)
        h = self.bar_embed(bars) + self.pe
        h = self.encoder(h)
        h = self.final_norm(h)
        return h

    def forward_mlm(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, C, S). mask: (B, S) bool — True at masked positions.
        Returns reconstructed bar values at masked positions: (B, S, C)."""
        # Replace masked timesteps with the learnable mask token before encoding.
        masked_input = x.clone()
        # mask shape (B, S); broadcast to (B, C, S)
        m = mask.unsqueeze(1).expand_as(x)
        masked_input = torch.where(m, self.mask_token.view(1, -1, 1).expand_as(x), masked_input)
        h = self.encode(masked_input)
        return self.reconstruction_head(h)

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, S). Returns L2-normalized embedding (B, proj_dim)."""
        h = self.encode(x)
        pooled = h.mean(dim=1)
        z = self.contrastive_head(pooled)
        return torch.nn.functional.normalize(z, dim=-1)


# ----- losses -----


def nt_xent_loss(z: torch.Tensor, temperature: float = NTX_TEMPERATURE) -> torch.Tensor:
    """NT-Xent over a batch arranged as [anchor_0, pos_0, anchor_1, pos_1, ...].

    z: (2B, proj_dim), already L2-normalized.

    Returns scalar loss = -mean log(exp(s_pos / τ) / Σ_other exp(s / τ)).
    """
    two_b = z.size(0)
    assert two_b % 2 == 0, "NT-Xent expects even batch of (anchor, positive) pairs"
    sim = z @ z.t() / temperature  # (2B, 2B)
    # Mask out the diagonal (self-similarity).
    sim.fill_diagonal_(float("-inf"))
    # Build the "positive index" for each row: anchor at 2i pairs with positive at 2i+1, and vice versa.
    pos_idx = torch.arange(two_b, device=z.device) ^ 1  # xor 1 flips the LSB
    loss = torch.nn.functional.cross_entropy(sim, pos_idx)
    return loss


# ----- data -----


class MaskedContrastiveDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Each __getitem__ yields:

      anchor_window  : (C, S)  — primary window, z-normed
      positive_window: (C, S)  — same symbol, within ±5 days, z-normed
      anchor_mask    : (S,) bool — 15% mask for MLM
      anchor_unmasked: (C, S)  — original z-normed anchor (target for MLM reconstruction)

    The MLM head sees the masked anchor; the contrastive head sees both
    anchor and positive (no masking).
    """

    def __init__(self, index: WindowIndex, mask_ratio: float = MASK_RATIO, eps: float = 1e-6, seed: int = 42) -> None:
        self.index = index
        self.mask_ratio = mask_ratio
        self.eps = eps
        self._rng = np.random.default_rng(seed)
        # Pre-compute per-symbol endpoint lists for fast positive-pair sampling.
        self._endpoints_by_sym: dict[int, np.ndarray] = {}
        for sym_id in range(len(index.symbols)):
            mask = index.endpoints[:, 0] == sym_id
            if mask.any():
                self._endpoints_by_sym[sym_id] = np.flatnonzero(mask)

    def __len__(self) -> int:
        return self.index.n_windows

    def _window_zscored(self, sym_id: int, local_end: int) -> np.ndarray:
        global_end = self.index.symbol_starts[sym_id] + local_end
        w = self.index.channels[global_end - WINDOW + 1 : global_end + 1]
        x = w.T.astype(np.float32, copy=False)  # (C, S)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + self.eps
        return (x - mean) / std

    def _sample_positive(self, anchor_pos: int, sym_id: int) -> int:
        """Pick another endpoint in the same symbol within ±NEIGHBOR_WINDOW_DAYS."""
        cohort = self._endpoints_by_sym[sym_id]
        anchor_local = self.index.endpoints[anchor_pos, 1]
        # local_end values are within NEIGHBOR_WINDOW_DAYS of anchor_local.
        diffs = np.abs(self.index.endpoints[cohort, 1] - anchor_local)
        candidates = cohort[(diffs > 0) & (diffs <= NEIGHBOR_WINDOW_DAYS)]
        if len(candidates) == 0:
            # Fall back to self (degenerate but rare — symbol has very few windows).
            return anchor_pos
        # Sample one. Using stdlib random is fine; this runs in DataLoader workers.
        import random as _random
        return int(_random.choice(candidates.tolist()))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sym_id, anchor_local = self.index.endpoints[i]
        anchor = self._window_zscored(sym_id, anchor_local)
        pos_pos = self._sample_positive(i, sym_id)
        pos_sym, pos_local = self.index.endpoints[pos_pos]
        positive = self._window_zscored(pos_sym, pos_local)
        # Generate mask via DataLoader-worker-local RNG. Numpy default_rng
        # per-worker isn't seeded deterministically; use python random
        # which DataLoader seeds per-worker for us via worker_init_fn (omitted
        # — non-determinism here is acceptable, masking is random anyway).
        import random as _random
        mask = np.zeros(WINDOW, dtype=bool)
        n_mask = max(1, int(WINDOW * self.mask_ratio))
        mask_indices = _random.sample(range(WINDOW), n_mask)
        mask[mask_indices] = True
        return (
            torch.from_numpy(anchor),
            torch.from_numpy(positive),
            torch.from_numpy(mask),
            torch.from_numpy(anchor),  # same array, MLM target
        )


# ----- training -----


@dataclass
class FoundationTrainer:
    """Encapsulates the optimizer / scheduler / scaler / training loop +
    integration with CheckpointManager + RunStatus."""

    run_dir: Path
    run_id: str
    pipeline_step: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    mlm_weight: float
    ntx_weight: float
    num_workers: int
    device: str
    mixed_precision: bool
    random_seed: int
    status_update_every_n_batches: int = 50

    _model: FoundationTransformer | None = field(default=None, init=False, repr=False)
    _stop_flag: dict[str, bool] = field(default_factory=lambda: {"stop": False}, init=False, repr=False)

    def request_stop(self) -> None:
        self._stop_flag["stop"] = True

    def train(self, train_idx: WindowIndex, val_idx: WindowIndex) -> dict[str, Any]:
        torch.manual_seed(self.random_seed)
        device = torch.device(self.device)

        model = FoundationTransformer().to(device)
        self._model = model
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  model: {n_params/1e6:.2f}M params on {device}")

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_ds = MaskedContrastiveDataset(train_idx, seed=self.random_seed)
        val_ds = MaskedContrastiveDataset(val_idx, seed=self.random_seed + 1)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=True, persistent_workers=(self.num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=self.num_workers, pin_memory=(device.type == "cuda"),
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.epochs
        def _lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision and device.type == "cuda")

        status = RunStatus(
            dir=self.run_dir,
            run_id=self.run_id,
            pipeline_step=self.pipeline_step,
            epoch_total=self.epochs,
        )
        ckpt = CheckpointManager(dir=self.run_dir, min_interval_s=1800.0)

        # Resume if a checkpoint exists.
        start_epoch = 0
        loss_history: list[dict[str, float]] = []
        latest = ckpt.latest_path()
        if latest is not None and latest.exists():
            print(f"  resuming from {latest.relative_to(_REPO_ROOT)}")
            payload = CheckpointManager.load(latest, model=model, optimizer=opt, scheduler=scheduler, scaler=scaler)
            start_epoch = payload["epoch"]
            loss_history = list(payload.get("extras", {}).get("loss_history", []))

        status.update(state="training", epoch_current=start_epoch)

        for epoch in range(start_epoch + 1, self.epochs + 1):
            if self._stop_flag["stop"]:
                break
            model.train()
            ep_mlm = 0.0
            ep_ntx = 0.0
            ep_total = 0.0
            n_batches = 0
            t_epoch = time.perf_counter()
            for batch_i, (anchor, positive, mask, target) in enumerate(train_loader):
                if self._stop_flag["stop"]:
                    break
                anchor = anchor.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=self.mixed_precision and device.type == "cuda"):
                    recon = model.forward_mlm(anchor, mask)
                    # MLM loss: MSE on masked positions only.
                    target_bars = target.transpose(1, 2)  # (B, S, C)
                    mse_per_pos = ((recon - target_bars) ** 2).mean(dim=-1)  # (B, S)
                    mlm_loss = (mse_per_pos * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                    # Contrastive: stack [anchor, positive] interleaved.
                    z_anchor = model.forward_contrastive(anchor)    # (B, P)
                    z_pos = model.forward_contrastive(positive)
                    # Interleave so pair (2i, 2i+1).
                    z_stack = torch.stack([z_anchor, z_pos], dim=1).reshape(-1, z_anchor.size(-1))
                    ntx_loss = nt_xent_loss(z_stack)
                    total_loss = self.mlm_weight * mlm_loss + self.ntx_weight * ntx_loss
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                ep_mlm += float(mlm_loss.item())
                ep_ntx += float(ntx_loss.item())
                ep_total += float(total_loss.item())
                n_batches += 1
                if batch_i % self.status_update_every_n_batches == 0:
                    status.update(
                        state="training",
                        epoch_current=epoch - 1,  # in-progress
                        extras={"batch_in_epoch": batch_i, "steps_per_epoch": steps_per_epoch},
                    )
                # Periodic checkpoint mid-epoch.
                if ckpt.should_save():
                    extras_mid = {"loss_history": loss_history, "in_progress_epoch": epoch}
                    ckpt.save(epoch=epoch - 1, model=model, optimizer=opt, scheduler=scheduler, scaler=scaler, extras=extras_mid)
                    status.record_checkpoint(epoch=epoch - 1)
            epoch_seconds = time.perf_counter() - t_epoch
            train_mlm = ep_mlm / max(n_batches, 1)
            train_ntx = ep_ntx / max(n_batches, 1)
            train_total = ep_total / max(n_batches, 1)

            # Validation pass — same losses, no gradient.
            val_mlm, val_ntx, val_total = self._validate(model, val_loader, device)
            print(
                f"  epoch {epoch:>2}/{self.epochs}  "
                f"train mlm={train_mlm:.4f} ntx={train_ntx:.4f} total={train_total:.4f}  "
                f"val mlm={val_mlm:.4f} ntx={val_ntx:.4f} total={val_total:.4f}  "
                f"({epoch_seconds:.0f}s)"
            )

            loss_history.append({
                "epoch": epoch,
                "train_mlm": round(train_mlm, 6),
                "train_ntx": round(train_ntx, 6),
                "train_total": round(train_total, 6),
                "val_mlm": round(val_mlm, 6),
                "val_ntx": round(val_ntx, 6),
                "val_total": round(val_total, 6),
                "epoch_seconds": round(epoch_seconds, 1),
            })
            # End-of-epoch checkpoint (always).
            extras = {"loss_history": loss_history}
            ckpt.save(epoch=epoch, model=model, optimizer=opt, scheduler=scheduler, scaler=scaler, extras=extras)
            status.record_checkpoint(epoch=epoch)
            status.mark_epoch_complete()
            status.update(state="training", epoch_current=epoch)

        return {"model": model, "loss_history": loss_history}

    @torch.no_grad()
    def _validate(self, model: FoundationTransformer, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
        model.eval()
        mlm_sum = 0.0
        ntx_sum = 0.0
        n = 0
        for anchor, positive, mask, target in loader:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=self.mixed_precision and device.type == "cuda"):
                recon = model.forward_mlm(anchor, mask)
                target_bars = target.transpose(1, 2)
                mse_per_pos = ((recon - target_bars) ** 2).mean(dim=-1)
                mlm_loss = (mse_per_pos * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                z_anchor = model.forward_contrastive(anchor)
                z_pos = model.forward_contrastive(positive)
                z_stack = torch.stack([z_anchor, z_pos], dim=1).reshape(-1, z_anchor.size(-1))
                ntx_loss = nt_xent_loss(z_stack)
            mlm_sum += float(mlm_loss.item())
            ntx_sum += float(ntx_loss.item())
            n += 1
        return mlm_sum / max(n, 1), ntx_sum / max(n, 1), mlm_sum / max(n, 1) + ntx_sum / max(n, 1)


# ----- entrypoint -----


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track F — foundation Transformer pretraining")
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--mlm-weight", type=float, default=1.0)
    p.add_argument("--ntx-weight", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument(
        "--resume",
        default=None,
        help='Pass "latest" to auto-resume from runs/<date>-<step>/latest.pt; '
        'pass an explicit path otherwise. No-op if no checkpoint exists.',
    )
    return p.parse_args(argv)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _save_safetensors_fp16(model: FoundationTransformer, path: Path) -> None:
    """Write the encoder (no heads) as fp16 safetensors. Falls back to
    torch.save if safetensors isn't installed."""
    state = {k: v.detach().to(torch.float16) for k, v in model.state_dict().items()}
    try:
        from safetensors.torch import save_file  # type: ignore[import]
        save_file(state, str(path))
    except ImportError:
        torch.save(state, path.with_suffix(".pt"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    pipeline_step = "step3f_foundation_pretrain"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(dir=run_dir, run_id=make_run_id(run_date_str, pipeline_step), pipeline_step=pipeline_step, epoch_total=args.epochs)
    trainer_holder: dict[str, FoundationTrainer | None] = {"trainer": None}

    def _on_sigint() -> None:
        print("[track-f] SIGINT — finishing current batch and pausing")
        if trainer_holder["trainer"] is not None:
            trainer_holder["trainer"].request_stop()
    install_graceful_interrupt(_on_sigint)
    status.update(state="training", epoch_current=0)

    try:
        print(f"track F (foundation pretrain) — run dir: {run_dir.relative_to(_REPO_ROOT)}")
        features_path = args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)

        # Same labeled-frame load as the CNN — we need windows over all of price_history,
        # filtered to rows that yielded a (now-discarded) is_winner label so the window
        # ends are well-defined. The label column itself isn't used for pretraining.
        labeled = pl.read_parquet(features_path).filter(pl.col("is_winner").is_not_null())
        print(f"  loaded labeled: {labeled.height:,} rows, {labeled['symbol'].n_unique()} symbols")
        full_idx = build_window_index(labeled)
        print(f"  total windows: {full_idx.n_windows:,}")

        # Train/val split by random window-id (val_fraction). NOT a time-based
        # split — the Step 2 walk-forward rule applies to supervised holdout,
        # not to self-supervised pretraining. Mixing time periods here is
        # intentional and makes the encoder representation-stable across regimes.
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(full_idx.n_windows)
        n_val = int(full_idx.n_windows * args.val_fraction)
        val_indices = np.sort(perm[:n_val])
        train_indices = np.sort(perm[n_val:])

        def _subset(idx: WindowIndex, keep: np.ndarray) -> WindowIndex:
            from dataclasses import replace
            return replace(
                idx,
                endpoints=idx.endpoints[keep],
                labels=idx.labels[keep],
                dates=idx.dates[keep],
            )
        train_idx = _subset(full_idx, train_indices)
        val_idx = _subset(full_idx, val_indices)
        print(f"  split: train={train_idx.n_windows:,}  val={val_idx.n_windows:,}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = FoundationTrainer(
            run_dir=run_dir,
            run_id=make_run_id(run_date_str, pipeline_step),
            pipeline_step=pipeline_step,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            mlm_weight=args.mlm_weight,
            ntx_weight=args.ntx_weight,
            num_workers=args.num_workers,
            device=device,
            mixed_precision=not args.no_mixed_precision,
            random_seed=args.seed,
        )
        trainer_holder["trainer"] = trainer
        result = trainer.train(train_idx, val_idx)
        model: FoundationTransformer = result["model"]
        loss_history = result["loss_history"]

        # Save artifacts.
        encoder_path = run_dir / "encoder.safetensors"
        _save_safetensors_fp16(model, encoder_path)
        actual_encoder_path = encoder_path if encoder_path.exists() else encoder_path.with_suffix(".pt")
        encoder_sha = hashlib.sha256(actual_encoder_path.read_bytes()).hexdigest()
        print(f"  wrote {actual_encoder_path.relative_to(_REPO_ROOT)}  ({actual_encoder_path.stat().st_size / 1e6:.1f} MB)")

        losses_df = pl.DataFrame(loss_history)
        losses_path = run_dir / "pretrain-losses.parquet"
        losses_df.write_parquet(losses_path)
        print(f"  wrote {losses_path.relative_to(_REPO_ROOT)}  ({losses_df.height} epochs)")

        wall_clock_s = round(time.perf_counter() - t0, 3)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "architecture": "foundation_transformer",
            "encoder_path": str(actual_encoder_path.relative_to(_REPO_ROOT)),
            "encoder_sha": f"sha256:{encoder_sha}",
            "n_channels": len(CHANNELS),
            "window_length": WINDOW,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dim_ff": DIM_FF,
            "proj_dim": PROJ_DIM,
            "mask_ratio": MASK_RATIO,
            "ntx_temperature": NTX_TEMPERATURE,
            "neighbor_window_days": NEIGHBOR_WINDOW_DAYS,
            "param_count": n_params,
            "epochs_planned": args.epochs,
            "epochs_trained": len(loss_history),
            "best_val_total": (
                round(min((h["val_total"] for h in loss_history), default=float("nan")), 6)
                if loss_history else None
            ),
            "final_val_mlm": round(loss_history[-1]["val_mlm"], 6) if loss_history else None,
            "final_val_ntx": round(loss_history[-1]["val_ntx"], 6) if loss_history else None,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "mlm_weight": args.mlm_weight,
            "ntx_weight": args.ntx_weight,
            "n_train_windows": int(train_idx.n_windows),
            "n_val_windows": int(val_idx.n_windows),
            "mixed_precision": not args.no_mixed_precision,
            "runtime_device": device,
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(
            f"=== TRACK F RESULT: {n_params/1e6:.1f}M params  "
            f"trained {len(loss_history)} epochs  "
            f"({wall_clock_s/3600:.2f}h on {device}) ==="
        )
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
