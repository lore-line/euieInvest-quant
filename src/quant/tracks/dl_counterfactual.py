"""Track 12 — DL counterfactual generation.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 12.

For every winner in the holdout, find the minimum-perturbation
counterfactual: an additive δ on the input window such that the
classifier flips its prediction from winner → loser. The
per-(timestep, channel) perturbation magnitude is the local
explanation — small δ means "this timestep/channel is *just barely*
classifying it as a winner; flip easily" — large δ means
"the model is confident about this dimension's contribution".

Method: PGD (Projected Gradient Descent) with L2-ball projection
to keep δ small. Step toward "loser" prediction; project δ back
into a small ball each step.

Pre-req: a trained binary classifier on top of the Track F encoder —
we use the Track 8 ProtoPNet classifier as the target model (any
classifier with sigmoid output works; ProtoPNet's interpretability
makes the counterfactual + prototype combination particularly
informative).

Outputs:
  counterfactual-perturbations.parquet — per-winner perturbation profile:
    (symbol, date, timestep, channel, original_value_z,
     perturbation_delta, original_proba, counterfactual_proba)
  aggregate-perturbation-stats.md — per-channel and per-timestep mean
    perturbation magnitudes; ranks (timestep, channel) cells by
    "how much pressure this dimension takes to flip the prediction"

GPU-bound; ~2-4h on 5090 for ~128K holdout winners.
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

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.models.cnn_discovery import WindowDataset
from quant.tracks.embedding_clustering import _find_latest_encoder, _load_encoder
from quant.tracks.prototype_learning import PrototypeLayer
from quant.tracks.xgb_rule_extraction import _replay_feature_selection
from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["main", "pgd_counterfactual"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def pgd_counterfactual(
    encoder: nn.Module,
    classifier: PrototypeLayer,
    x: torch.Tensor,
    target_proba: float = 0.4,
    eps: float = 2.0,
    step_size: float = 0.1,
    n_steps: int = 30,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """PGD search for the smallest L2-ball perturbation that drops the
    winner probability below ``target_proba``.

    x: (B, C, S) z-normed window. Returns (delta, final_proba).
    """
    delta = torch.zeros_like(x, requires_grad=True)
    for step in range(n_steps):
        x_adv = x + delta
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            h = encoder.encode(x_adv).mean(dim=1).float()
            logits, _ = classifier(h)
        probas = torch.sigmoid(logits)
        # Loss: drive probas DOWN. We use direct gradient on probas.
        loss = probas.sum()
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta - step_size * grad.sign()
            # Project to L2-ball.
            flat = delta.view(delta.size(0), -1)
            norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
            scale = torch.clamp(eps / norms, max=1.0)
            delta = (flat * scale).view_as(delta).detach().requires_grad_(True)
        # Stop early if all already below target.
        if (probas < target_proba).all():
            break
    with torch.no_grad():
        x_adv = x + delta
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            h = encoder.encode(x_adv).mean(dim=1).float()
            logits, _ = classifier(h)
        final_p = torch.sigmoid(logits)
    return delta.detach(), final_p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 12 — DL counterfactual")
    p.add_argument("--encoder-path", type=Path, default=None)
    p.add_argument(
        "--classifier-checkpoint", type=Path, default=None,
        help="Path to Track 8 latest.pt (or compatible PrototypeLayer checkpoint). "
        "Defaults to the latest step3h_prototype_learning run.",
    )
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--target-proba", type=float, default=0.4)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-winners-sample", type=int, default=10_000,
                   help="Per-winner PGD is expensive — subsample. Set 0 for all.")
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


def _find_latest_proto() -> Path | None:
    runs = sorted(_REPO_ROOT.glob("runs/*step3h_prototype_learning*"))
    if not runs:
        return None
    for d in reversed(runs):
        p = d / "latest.pt"
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
    pipeline_step = "step3l_dl_counterfactual"
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
        if encoder_path is None:
            raise FileNotFoundError("no Track F encoder found")
        if not encoder_path.is_absolute():
            encoder_path = _REPO_ROOT / encoder_path
        if not encoder_path.exists():
            raise FileNotFoundError(f"Track F encoder not found at {encoder_path}")
        clf_ckpt_path = args.classifier_checkpoint or _find_latest_proto()
        if clf_ckpt_path is None:
            raise FileNotFoundError(
                "no Track 8 prototype-layer checkpoint found — run step3h_prototype_learning first"
            )
        if not clf_ckpt_path.is_absolute():
            clf_ckpt_path = _REPO_ROOT / clf_ckpt_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = _load_encoder(encoder_path, device)
        for p in encoder.parameters():
            p.requires_grad = False
        classifier = PrototypeLayer(d_model=encoder.d_model).to(device)
        # Load the saved PrototypeLayer state (uses CheckpointManager's payload format).
        payload = torch.load(clf_ckpt_path, map_location=device, weights_only=False)
        classifier.load_state_dict(payload["model_state_dict"])
        for p in classifier.parameters():
            p.requires_grad = False
        classifier.eval()
        encoder.eval()
        print(f"track 12 (DL counterfactual)")
        print(f"  encoder:   {encoder_path.relative_to(_REPO_ROOT)}")
        print(f"  classifier: {clf_ckpt_path.relative_to(_REPO_ROOT)} (Track 8)")

        labeled = pl.read_parquet(
            args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        ).filter(pl.col("is_winner").is_not_null())
        labeled, _ = _replay_feature_selection(labeled)
        holdout = labeled.filter(pl.col("date") > args.val_end)
        winners = holdout.filter(pl.col("is_winner") == True).sort(["symbol", "date"])
        winner_idx = build_window_index(winners)
        print(f"  holdout winners: {winner_idx.n_windows:,}")

        if args.n_winners_sample > 0 and args.n_winners_sample < winner_idx.n_windows:
            rng = np.random.default_rng(42)
            sample = np.sort(rng.choice(winner_idx.n_windows, size=args.n_winners_sample, replace=False))
            from dataclasses import replace
            winner_idx = replace(
                winner_idx,
                endpoints=winner_idx.endpoints[sample],
                labels=winner_idx.labels[sample],
                dates=winner_idx.dates[sample],
            )
            print(f"  sampled to {winner_idx.n_windows:,} for PGD budget")

        ds = WindowDataset(winner_idx)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        symbols = np.array([winner_idx.symbols[s] for s in winner_idx.endpoints[:, 0]])
        dates = winner_idx.dates.astype("datetime64[D]").astype(str)

        # Accumulate per-(channel, timestep) mean perturbation magnitudes.
        accum = np.zeros((len(CHANNELS), WINDOW))
        perturbation_rows: list[dict[str, Any]] = []
        i = 0
        for xb, _ in loader:
            if stop_flag["stop"]:
                raise KeyboardInterrupt
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h0 = encoder.encode(xb).mean(dim=1).float()
                    logits0, _ = classifier(h0)
                orig_p = torch.sigmoid(logits0).cpu().numpy()
            delta, final_p = pgd_counterfactual(
                encoder, classifier, xb,
                target_proba=args.target_proba,
                eps=args.eps, step_size=args.step_size, n_steps=args.n_steps,
                device=device,
            )
            delta_np = delta.cpu().numpy()  # (B, C, S)
            final_p_np = final_p.cpu().numpy()
            accum += np.abs(delta_np).sum(axis=0)
            # Per-winner per-cell perturbation — slim only to non-zero cells.
            x_np = xb.cpu().numpy()
            for b in range(xb.size(0)):
                # Pick top-K cells by |delta| to keep the output manageable.
                k = 10
                flat = np.abs(delta_np[b]).flatten()
                top_cells = np.argpartition(-flat, k - 1)[:k]
                for cell in top_cells:
                    c_i, t_i = divmod(int(cell), WINDOW)
                    perturbation_rows.append({
                        "symbol": str(symbols[i + b]),
                        "date": dates[i + b],
                        "channel": CHANNELS[c_i],
                        "timestep": t_i - (WINDOW - 1),
                        "original_value_z": float(x_np[b, c_i, t_i]),
                        "perturbation_delta": float(delta_np[b, c_i, t_i]),
                        "original_proba": float(orig_p[b]),
                        "counterfactual_proba": float(final_p_np[b]),
                    })
            i += xb.size(0)
            if i % 1000 == 0 or i >= winner_idx.n_windows:
                status.update(state="training", epoch_current=0,
                              extras={"winners_processed": i, "winners_total": winner_idx.n_windows})

        # Write artifacts.
        pert_df = pl.DataFrame(perturbation_rows).with_columns(pl.col("date").str.to_date())
        pert_path = run_dir / "counterfactual-perturbations.parquet"
        pert_df.write_parquet(pert_path)
        print(f"  wrote {pert_path.relative_to(_REPO_ROOT)}  ({pert_df.height:,} rows)")

        # Aggregate stats.
        accum /= max(winner_idx.n_windows, 1)
        per_channel = accum.mean(axis=1)  # (C,)
        per_timestep = accum.mean(axis=0)  # (S,)
        lines = ["# Aggregate counterfactual perturbation stats\n\n",
                 "Mean per-cell |δ| across all sampled holdout winners.\n\n",
                 "## Per-channel\n\n",
                 "| channel | mean |δ| |\n",
                 "|---|---|\n"]
        for c_i, ch in enumerate(CHANNELS):
            lines.append(f"| {ch} | {per_channel[c_i]:.4f} |\n")
        lines.extend(["\n## Per-timestep (window-relative; -59 = oldest, 0 = day t-1)\n\n",
                      "| timestep | mean |δ| |\n",
                      "|---|---|\n"])
        for t in range(WINDOW):
            lines.append(f"| {t - (WINDOW - 1)} | {per_timestep[t]:.4f} |\n")
        (run_dir / "aggregate-perturbation-stats.md").write_text("".join(lines))
        print(f"  wrote aggregate-perturbation-stats.md")

        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "classifier_checkpoint": str(clf_ckpt_path.relative_to(_REPO_ROOT)),
            "pgd_target_proba": args.target_proba,
            "pgd_eps": args.eps,
            "pgd_step_size": args.step_size,
            "pgd_n_steps": args.n_steps,
            "n_winners_attacked": int(winner_idx.n_windows),
            "channel_with_max_mean_delta": str(CHANNELS[int(np.argmax(per_channel))]),
            "timestep_with_max_mean_delta": int(np.argmax(per_timestep) - (WINDOW - 1)),
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"=== TRACK 12 RESULT: PGD over {winner_idx.n_windows:,} winners ({wall_clock_s/60:.1f}min) ===")
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
