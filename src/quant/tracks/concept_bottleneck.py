"""Track 9 — Concept-bottleneck model.

CLAUDE.md §12 / PR #1 issuecomment-4436101547 §Track 9.

Train a model with an intermediate concept-prediction layer + final
winner classifier. The bottleneck forces winner predictions to flow
*through* a small set of human-readable concepts, making the final
prediction directly interpretable as a linear combination of
concept activations.

Concept list (30 concepts) — derived from the 47 hand-crafted
features + Track 1's top SHAP/rules + Track 6's z_delta finding.
Each concept is a thresholded combination of existing features,
computed deterministically at training time. The model is
*supervised* on these concept labels and on the winner label
simultaneously; the bottleneck ensures the winner prediction can
only use information that passes through the concept layer.

The brief deferred the concept list to a follow-up; this is the
proposed enumeration. Any concept the server team wants to add /
remove / threshold differently is a one-line edit to CONCEPTS.

Pre-req: Track F encoder available.

Outputs:
  concept-activations.parquet   — (symbol, date, concept_1..30, predicted_proba, is_winner)
  concept-importance.parquet    — (concept_name, weight_in_final_classifier,
                                    correlation_with_winner, mean_activation_winners,
                                    mean_activation_losers)
  concept-cluster-signatures.md — concept combinations characterizing each cluster
                                    from Tracks 2 and 7 (filled by synthesis)
  losses.parquet                — per-epoch concept_loss / winner_loss / val_prec
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

__all__ = ["CONCEPTS", "ConceptBottleneck", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


# Concept definitions: each maps a name to a callable `pl.Expr` that returns
# a boolean column for the input frame. Frames are the post-_replay_feature_selection
# rows, so all 47 features + market_regime dummies are available.
#
# Order MUST be stable across runs — the column index matches the model's
# concept-layer position. Changing this list invalidates trained checkpoints.

CONCEPTS: list[tuple[str, Any]] = [
    # Volume / volatility
    ("volume_breakout_5d", lambda: pl.col("vol_mult_5") > 2.0),
    ("volume_persistent",  lambda: pl.col("vol_mult_60") > 1.5),
    ("high_atr_regime",    lambda: pl.col("atr_pct_14") > 0.04),
    ("low_atr_regime",     lambda: pl.col("atr_pct_14") < 0.015),
    ("bb_squeeze_tight",   lambda: pl.col("bb_squeeze_20") < 0.05),
    ("bb_expansion",       lambda: pl.col("bb_squeeze_20") > 0.15),
    ("range_expansion",    lambda: pl.col("range_expansion_5d") > 1.5),
    # Momentum / trend
    ("rsi_oversold",       lambda: pl.col("rsi_14") < 30.0),
    ("rsi_overbought",     lambda: pl.col("rsi_14") > 70.0),
    ("macd_above_signal",  lambda: pl.col("macd_hist") > 0.0),
    ("roc_60_positive",    lambda: pl.col("roc_60") > 0.0),
    ("sma20_rising",       lambda: pl.col("sma20_slope_5d") > 0.0),
    ("sma50_rising",       lambda: pl.col("sma50_slope_5d") > 0.0),
    # Position relative to extremes
    ("near_year_high",     lambda: pl.col("pct_of_252d_high") > 0.95),
    ("off_year_high",      lambda: pl.col("pct_of_252d_high") < 0.5),
    ("near_year_low",      lambda: pl.col("pct_of_252d_low") < 1.1),
    ("above_sma200",       lambda: pl.col("close_over_sma_200") > 1.0),
    ("above_sma50",        lambda: pl.col("close_over_sma_50") > 1.0),
    # Peer / market
    ("peer_strength_high", lambda: pl.col("close_over_sma_20_peer_z") > 1.0),
    ("peer_strength_low",  lambda: pl.col("close_over_sma_20_peer_z") < -1.0),
    ("outperforming_spy",  lambda: pl.col("rs_spy_20d") > 5.0),
    ("market_uptrend",     lambda: pl.col("market_regime_uptrend") == 1),
    ("market_downtrend",   lambda: pl.col("market_regime_downtrend") == 1),
    ("market_chop",        lambda: pl.col("market_regime_chop") == 1),
    # Bar shape
    ("inside_bar",         lambda: pl.col("is_inside_bar") == 1),
    ("nr4_compression",    lambda: pl.col("is_nr4") == 1),
    ("nr7_compression",    lambda: pl.col("is_nr7") == 1),
    # Recency / setup
    ("fresh_setup",        lambda: pl.col("days_since_last_20pct") > 60),
    ("recent_winner_echo", lambda: pl.col("days_since_last_20pct") < 30),
    ("hv_short_above_long", lambda: pl.col("hv_ratio_10_60") > 1.0),
    # A/D-line negative distribution (PR #1 issuecomment-4436499617 ask) —
    # Track 1's signature finding: rules involving `ad_line < <negative>`
    # appeared in 5 of the top-10 rules. Captures the "smart-money
    # distributed, panic washout, now bouncing" pattern.
    ("ad_line_negative_distribution", lambda: pl.col("ad_line") < -1_000_000),
]
N_CONCEPTS = len(CONCEPTS)


def compute_concept_labels(df: pl.DataFrame) -> np.ndarray:
    """Apply each CONCEPT's expression to ``df``; return (N, n_concepts) int8."""
    cols = []
    for name, expr_fn in CONCEPTS:
        cols.append(expr_fn().fill_null(False).cast(pl.Int8).alias(name))
    df = df.with_columns(cols)
    return df.select([c.meta.output_name() for c in cols]).to_numpy().astype(np.int8)


class ConceptBottleneck(nn.Module):
    """Frozen encoder → concept head (sigmoid, 30 dims) → linear classifier.

    The final winner-logit is a *linear combination* of concept activations,
    so the classifier's weight vector directly attributes the prediction
    to concepts.
    """

    def __init__(self, d_model: int = 768, n_concepts: int = N_CONCEPTS) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.concept_head = nn.Linear(d_model, n_concepts)
        self.classifier = nn.Linear(n_concepts, 1)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """emb: (B, D). Returns (concept_logits: (B, n_concepts), winner_logit: (B,))."""
        c_logits = self.concept_head(emb)
        c_acts = torch.sigmoid(c_logits)
        w_logit = self.classifier(c_acts).squeeze(-1)
        return c_logits, w_logit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A Track 9 — concept bottleneck")
    p.add_argument("--encoder-path", type=Path, default=None)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-concept", type=float, default=1.0)
    p.add_argument("--lambda-winner", type=float, default=1.0)
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
    pipeline_step = "step3i_concept_bottleneck"
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
        model = ConceptBottleneck(d_model=encoder.d_model, n_concepts=N_CONCEPTS).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss()
        print(f"track 9 (concept bottleneck) — {N_CONCEPTS} concepts, encoder frozen")

        labeled = pl.read_parquet(
            args.features if args.features.is_absolute() else (_REPO_ROOT / args.features)
        ).filter(pl.col("is_winner").is_not_null())
        labeled, _ = _replay_feature_selection(labeled)
        train_df = labeled.filter(pl.col("date") <= args.val_end)
        val_df = labeled.filter(pl.col("date") > args.val_end)
        train_concepts = compute_concept_labels(train_df)
        val_concepts = compute_concept_labels(val_df)
        train_idx = build_window_index(train_df)
        val_idx = build_window_index(val_df)
        # Concept labels are aligned by row order — same as endpoints[:, 1] is the source-row index within symbol.
        # We need to map each window's (sym_id, local_end) back to its row in train_df. That's the global_end.
        train_concepts_per_window = np.empty((train_idx.n_windows, N_CONCEPTS), dtype=np.int8)
        for w_i, (sym_id, local_end) in enumerate(train_idx.endpoints):
            train_concepts_per_window[w_i] = train_concepts[train_idx.symbol_starts[sym_id] + local_end]
        val_concepts_per_window = np.empty((val_idx.n_windows, N_CONCEPTS), dtype=np.int8)
        for w_i, (sym_id, local_end) in enumerate(val_idx.endpoints):
            val_concepts_per_window[w_i] = val_concepts[val_idx.symbol_starts[sym_id] + local_end]

        print(f"  windows: train={train_idx.n_windows:,} val={val_idx.n_windows:,}")
        print(f"  concept activation rates (train, mean over concepts): {train_concepts_per_window.mean():.3f}")

        # Hand-roll the loader so we can pull concept labels alongside (xb, yb).
        # WindowDataset returns (window, is_winner); we attach concepts by index.
        ds = WindowDataset(train_idx)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_ds = WindowDataset(val_idx)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

        # We need indices to pull concept labels. Override the sampler with an index-aware loader.
        def _train_iter():
            from torch.utils.data import RandomSampler
            indices = list(RandomSampler(range(train_idx.n_windows)))
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                xs = torch.stack([ds[i][0] for i in batch_indices])
                ys = torch.tensor([train_idx.labels[i] for i in batch_indices], dtype=torch.float32)
                cs = torch.tensor(train_concepts_per_window[batch_indices], dtype=torch.float32)
                yield xs, ys, cs

        def _val_iter():
            for start in range(0, val_idx.n_windows, args.batch_size * 2):
                end = min(start + args.batch_size * 2, val_idx.n_windows)
                xs = torch.stack([val_ds[i][0] for i in range(start, end)])
                ys = torch.tensor([val_idx.labels[i] for i in range(start, end)], dtype=torch.float32)
                cs = torch.tensor(val_concepts_per_window[start:end], dtype=torch.float32)
                yield xs, ys, cs

        ckpt = CheckpointManager(dir=run_dir)
        loss_history = []
        for epoch in range(1, args.epochs + 1):
            if stop_flag["stop"]:
                break
            model.train()
            ep_c = ep_w = 0.0
            n = 0
            for xs, ys, cs in _train_iter():
                if stop_flag["stop"]:
                    break
                xs = xs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                cs = cs.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xs).mean(dim=1).float()
                c_logits, w_logit = model(h)
                loss_c = bce(c_logits, cs)
                loss_w = bce(w_logit, ys)
                loss = args.lambda_concept * loss_c + args.lambda_winner * loss_w
                loss.backward()
                opt.step()
                ep_c += float(loss_c.item())
                ep_w += float(loss_w.item())
                n += 1
            model.eval()
            val_logits = []
            val_labels = []
            with torch.no_grad():
                for xs, ys, cs in _val_iter():
                    xs = xs.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        h = encoder.encode(xs).mean(dim=1).float()
                    _, w_logit = model(h)
                    val_logits.append(torch.sigmoid(w_logit).cpu().numpy())
                    val_labels.append(ys.numpy())
            v_logits = np.concatenate(val_logits)
            v_labels = np.concatenate(val_labels).astype(np.int8)
            k = max(1, len(v_logits) // 10)
            top_idx = np.argpartition(-v_logits, k - 1)[:k]
            val_prec = float(v_labels[top_idx].sum() / k)
            loss_history.append({
                "epoch": epoch,
                "train_concept_loss": round(ep_c / max(n, 1), 6),
                "train_winner_loss": round(ep_w / max(n, 1), 6),
                "val_prec_topd": round(val_prec, 6),
            })
            print(f"  epoch {epoch:>2}/{args.epochs}  c_loss={ep_c/max(n,1):.4f}  w_loss={ep_w/max(n,1):.4f}  val_prec@TD={val_prec:.4f}")
            ckpt.save(epoch=epoch, model=model, optimizer=opt, extras={"loss_history": loss_history})
            status.record_checkpoint(epoch=epoch)
            status.update(state="training", epoch_current=epoch)

        # Compute concept activations + final predictions on val (= holdout) for the artifact.
        print("  computing holdout concept activations ...")
        model.eval()
        acts_all = []
        preds_all = []
        with torch.no_grad():
            for xs, ys, cs in _val_iter():
                xs = xs.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h = encoder.encode(xs).mean(dim=1).float()
                c_logits, w_logit = model(h)
                acts_all.append(torch.sigmoid(c_logits).cpu().numpy())
                preds_all.append(torch.sigmoid(w_logit).cpu().numpy())
        acts = np.concatenate(acts_all, axis=0)
        preds = np.concatenate(preds_all, axis=0)
        val_symbols = np.array([val_idx.symbols[s] for s in val_idx.endpoints[:, 0]])
        val_dates = val_idx.dates.astype("datetime64[D]").astype(str)
        is_winner_arr = val_idx.labels.astype(bool)

        # concept-activations.parquet
        act_data = {"symbol": val_symbols, "date": val_dates}
        for c_i, (name, _) in enumerate(CONCEPTS):
            act_data[name] = acts[:, c_i].astype(np.float64)
        act_data["predicted_proba"] = preds.astype(np.float64)
        act_data["is_winner"] = is_winner_arr
        act_df = pl.DataFrame(act_data).with_columns(pl.col("date").str.to_date())
        act_df.write_parquet(run_dir / "concept-activations.parquet")
        print(f"  wrote concept-activations.parquet  ({act_df.height:,} rows × {act_df.width} cols)")

        # concept-importance.parquet
        weights = model.classifier.weight.detach().cpu().numpy().squeeze(0)  # (n_concepts,)
        win_mean = acts[is_winner_arr].mean(axis=0)
        los_mean = acts[~is_winner_arr].mean(axis=0)
        # Pearson r between concept activation and winner label.
        corrs = []
        for c_i in range(N_CONCEPTS):
            corrs.append(float(np.corrcoef(acts[:, c_i], is_winner_arr.astype(np.float32))[0, 1]))
        imp_rows = []
        for c_i, (name, _) in enumerate(CONCEPTS):
            imp_rows.append({
                "concept_name": name,
                "weight_in_final_classifier": round(float(weights[c_i]), 6),
                "correlation_with_winner": round(corrs[c_i], 6),
                "mean_activation_winners": round(float(win_mean[c_i]), 6),
                "mean_activation_losers": round(float(los_mean[c_i]), 6),
                "abs_winner_loser_gap": round(abs(float(win_mean[c_i] - los_mean[c_i])), 6),
            })
        imp_rows.sort(key=lambda r: -abs(r["weight_in_final_classifier"]))
        pl.DataFrame(imp_rows).write_parquet(run_dir / "concept-importance.parquet")
        print(f"  wrote concept-importance.parquet")
        print(f"  top-5 by |weight|:")
        for row in imp_rows[:5]:
            print(f"    {row['concept_name']:<28} weight={row['weight_in_final_classifier']:+.3f}  corr={row['correlation_with_winner']:+.3f}")

        (run_dir / "concept-cluster-signatures.md").write_text(
            "# Concept signatures per cluster\n\n"
            "Per-cluster mean concept activations (Tracks 2 + 7 cluster IDs joined\n"
            "with this track's concept-activations.parquet). Skeleton — filled by\n"
            "the synthesis stage using the recipe:\n\n"
            "```python\n"
            "import polars as pl\n"
            "concepts = pl.read_parquet('runs/<date>-step3i_concept_bottleneck/concept-activations.parquet')\n"
            "t2_membership = pl.read_parquet('runs/<date>-step3b_handcrafted_clustering/cluster-membership.parquet')\n"
            "joined = concepts.join(t2_membership, on=['symbol', 'date'])\n"
            "joined.group_by(['algorithm', 'k', 'cluster_id']).agg([\n"
            "    pl.mean(c[0]).alias(c[0]) for c in CONCEPTS\n"
            "])\n"
            "```\n"
        )

        pl.DataFrame(loss_history).write_parquet(run_dir / "losses.parquet")
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "encoder_path": str(encoder_path.relative_to(_REPO_ROOT)),
            "n_concepts": N_CONCEPTS,
            "concept_list": [name for name, _ in CONCEPTS],
            "architecture": "frozen_encoder + concept_head + linear_classifier",
            "epochs_trained": len(loss_history),
            "final_val_prec_topd": loss_history[-1]["val_prec_topd"] if loss_history else None,
            "runtime_device": str(device),
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"=== TRACK 9 RESULT: {N_CONCEPTS} concepts, val_prec@TD={loss_history[-1]['val_prec_topd']:.4f} ({wall_clock_s/60:.1f}min) ===")
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
