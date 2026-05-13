"""Track 10 — Generative winner modeling (VAE).

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 10.

Train a VAE on winner-only 60×6 windows. Encoder is initialized from
Track F's pretrained encoder; decoder is small from-scratch. Three
artifacts after training:

  synthetic-winners.parquet — 100 generated 60×6 windows + VAE log-likelihoods
  latent-traversal.parquet  — 20 interpolation paths between real winner pairs
  density-scores.parquet    — per-holdout-window VAE log-density (high = looks like a typical winner setup)

Pre-req: Track F encoder available.
GPU-bound; ~3-6h on 5090.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.models.cnn_discovery import WindowDataset
from quant.tracks.embedding_clustering import _find_latest_encoder, _load_encoder
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.train import CheckpointManager, RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["WindowVAE", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
LATENT_DIM = 64


class WindowVAE(nn.Module):
    """Frozen encoder → mean-pool → small variational head → small decoder.

    Decoder produces 60×6 reconstructed windows via Linear(latent → 60×6).
    Simple architecture intentional — the heavy lifting was in pretraining.
    """

    def __init__(self, d_model: int = 768, latent_dim: int = LATENT_DIM, seq_len: int = WINDOW, n_channels: int = len(CHANNELS)) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, seq_len * n_channels),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode_to_latent(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.view(-1, self.n_channels, self.seq_len)

    def log_density(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Approximate log p(x) via VAE ELBO (1-sample estimate)."""
        mu, logvar = self.encode_to_latent(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon_ll = -((recon - x) ** 2).mean(dim=(-1, -2))  # per-window mean recon
        kl = -0.5 * (1.0 + logvar - mu ** 2 - logvar.exp()).sum(dim=-1)
        return recon_ll - kl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 10 — generative winner VAE")
    p.add_argument("--encoder-path", type=Path, default=None)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--beta", type=float, default=0.5, help="β-VAE weight on KL term")
    p.add_argument("--n-synthetic", type=int, default=100)
    p.add_argument("--n-traversals", type=int, default=20)
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
    pipeline_step = "step3j_generative_winners"
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
        for p in encoder.parameters():
            p.requires_grad = False
        vae = WindowVAE(d_model=encoder.d_model).to(device)
        opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
        print(f"track 10 (winner VAE) — encoder frozen, vae latent_dim={LATENT_DIM}")

        labeled = pl.read_parquet(
            args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        ).filter(pl.col("is_winner").is_not_null())
        labeled, _ = _replay_feature_selection(labeled)
        # Winners-only for training; full holdout for density scoring.
        winners_train = labeled.filter(
            (pl.col("date") <= args.val_end) & (pl.col("is_winner") == True)
        )
        holdout = labeled.filter(pl.col("date") > args.val_end)
        train_idx = build_window_index(winners_train)
        holdout_idx = build_window_index(holdout)
        print(f"  windows: train_winners={train_idx.n_windows:,}  holdout(all)={holdout_idx.n_windows:,}")

        ds = WindowDataset(train_idx)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        ckpt = CheckpointManager(dir=run_dir)
        loss_history = []
        for epoch in range(1, args.epochs + 1):
            if stop_flag["stop"]:
                break
            vae.train()
            ep_recon = ep_kl = 0.0
            n = 0
            for xb, _ in loader:
                if stop_flag["stop"]:
                    break
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xb).mean(dim=1).float()
                mu, logvar = vae.encode_to_latent(h)
                z = vae.reparameterize(mu, logvar)
                recon = vae.decode(z)
                recon_loss = ((recon - xb) ** 2).mean()
                kl_loss = -0.5 * (1.0 + logvar - mu ** 2 - logvar.exp()).mean()
                loss = recon_loss + args.beta * kl_loss
                loss.backward()
                opt.step()
                ep_recon += float(recon_loss.item())
                ep_kl += float(kl_loss.item())
                n += 1
            loss_history.append({
                "epoch": epoch,
                "train_recon": round(ep_recon / max(n, 1), 6),
                "train_kl": round(ep_kl / max(n, 1), 6),
            })
            print(f"  epoch {epoch:>2}/{args.epochs}  recon={ep_recon/max(n,1):.4f}  kl={ep_kl/max(n,1):.4f}")
            ckpt.save(epoch=epoch, model=vae, optimizer=opt, extras={"loss_history": loss_history})
            status.record_checkpoint(epoch=epoch)
            status.update(state="training", epoch_current=epoch)

        # synthetic-winners.parquet — sample N from prior, decode.
        print(f"  generating {args.n_synthetic} synthetic winners ...")
        vae.eval()
        with torch.no_grad():
            z_samples = torch.randn(args.n_synthetic, LATENT_DIM, device=device)
            synthetic = vae.decode(z_samples).cpu().numpy()  # (N, C, S)
        syn_rows = []
        for sid in range(args.n_synthetic):
            for t in range(WINDOW):
                syn_rows.append({
                    "synthetic_id": sid,
                    "t_offset": t - (WINDOW - 1),
                    "open":      float(synthetic[sid, 0, t]),
                    "high":      float(synthetic[sid, 1, t]),
                    "low":       float(synthetic[sid, 2, t]),
                    "close":     float(synthetic[sid, 3, t]),
                    "close_adj": float(synthetic[sid, 4, t]),
                    "volume":    float(synthetic[sid, 5, t]),
                })
        pl.DataFrame(syn_rows).write_parquet(run_dir / "synthetic-winners.parquet")
        print(f"  wrote synthetic-winners.parquet  ({args.n_synthetic * WINDOW} rows)")

        # latent-traversal.parquet — N pairs of real winners, interpolate.
        print(f"  computing {args.n_traversals} latent traversals ...")
        rng = np.random.default_rng(42)
        n_steps = 11
        traversal_rows = []
        with torch.no_grad():
            for tid in range(args.n_traversals):
                i, j = rng.choice(train_idx.n_windows, size=2, replace=False)
                xs_pair = torch.stack([ds[int(i)][0], ds[int(j)][0]]).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h_pair = encoder.encode(xs_pair).mean(dim=1).float()
                mu_pair, _ = vae.encode_to_latent(h_pair)
                for s in range(n_steps):
                    alpha = s / (n_steps - 1)
                    z = (1 - alpha) * mu_pair[0] + alpha * mu_pair[1]
                    recon = vae.decode(z.unsqueeze(0)).squeeze(0).cpu().numpy()  # (C, S)
                    for t in range(WINDOW):
                        traversal_rows.append({
                            "traversal_id": tid,
                            "step": s,
                            "alpha": round(alpha, 4),
                            "t_offset": t - (WINDOW - 1),
                            "open":      float(recon[0, t]),
                            "high":      float(recon[1, t]),
                            "low":       float(recon[2, t]),
                            "close":     float(recon[3, t]),
                            "close_adj": float(recon[4, t]),
                            "volume":    float(recon[5, t]),
                        })
        pl.DataFrame(traversal_rows).write_parquet(run_dir / "latent-traversal.parquet")
        print(f"  wrote latent-traversal.parquet")

        # density-scores.parquet — every holdout window's log-density under the VAE.
        print(f"  scoring {holdout_idx.n_windows:,} holdout windows under VAE ...")
        holdout_ds = WindowDataset(holdout_idx)
        holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size * 4, shuffle=False, num_workers=2, pin_memory=True)
        all_lp = []
        with torch.no_grad():
            for xb, _ in holdout_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xb).mean(dim=1).float()
                lp = vae.log_density(xb, h).cpu().numpy()
                all_lp.append(lp)
        log_p = np.concatenate(all_lp)
        symbols = np.array([holdout_idx.symbols[s] for s in holdout_idx.endpoints[:, 0]])
        dates = holdout_idx.dates.astype("datetime64[D]").astype(str)
        is_winner_arr = holdout_idx.labels.astype(bool)
        pl.DataFrame({
            "symbol": symbols,
            "date": dates,
            "log_density": log_p.astype(np.float64),
            "is_winner": is_winner_arr,
        }).with_columns(pl.col("date").str.to_date()).write_parquet(run_dir / "density-scores.parquet")
        print(f"  wrote density-scores.parquet  ({log_p.size:,} rows)")

        pl.DataFrame(loss_history).write_parquet(run_dir / "losses.parquet")
        wall_clock_s = round(time.perf_counter() - t0, 3)
        n_params = sum(p.numel() for p in vae.parameters())
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "architecture": "frozen_encoder + small_VAE",
            "latent_dim": LATENT_DIM,
            "param_count": n_params,
            "epochs_trained": len(loss_history),
            "beta": args.beta,
            "n_synthetic": args.n_synthetic,
            "n_traversals": args.n_traversals,
            "winner_mean_log_density": round(float(log_p[is_winner_arr].mean()), 6) if is_winner_arr.any() else None,
            "loser_mean_log_density": round(float(log_p[~is_winner_arr].mean()), 6) if (~is_winner_arr).any() else None,
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"=== TRACK 10 RESULT: VAE trained ({wall_clock_s/60:.1f}min) ===")
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
