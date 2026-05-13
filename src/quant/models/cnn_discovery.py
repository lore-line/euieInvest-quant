"""1D-CNN discovery model — Step 2b first architecture.

CLAUDE.md §12 (post-2026-05-12) authorizes DL as a parallel research
track. Spec is from PR #1 issuecomment-4435741839:

  Conv1d(6→64, k=3) → ReLU → MaxPool → Conv1d(64→128, k=3) →
  ReLU → GlobalAvgPool → Linear(128→1)

Same labels as XGB (`close_adj` +20%/30d, CLAUDE.md §6), same train/val/
holdout splits (CLAUDE.md §8), same `precision@top-decile` metric. The
goal is apples-to-apples vs the XGB 44.58% baseline.

Wraps in a `CnnDiscovery` class that mirrors `XGBDiscovery`'s interface:
fit / predict / shap_summary (here returning IntegratedGradients) so
discover.py can swap models cleanly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quant.data.windows import CHANNELS, WINDOW, WindowIndex

__all__ = ["Cnn1d", "CnnDiscovery", "WindowDataset"]


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy (window, label) producer over a :class:`WindowIndex`.

    Per-window z-normalization happens here (not in the model) so the
    DataLoader's worker processes do the normalization on CPU in
    parallel with GPU forward/backward — the network stays GPU-bound.
    """

    def __init__(self, index: WindowIndex, eps: float = 1e-6) -> None:
        self.index = index
        self.eps = float(eps)

    def __len__(self) -> int:
        return self.index.n_windows

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        sym_id, local_end = self.index.endpoints[i]
        global_end = self.index.symbol_starts[sym_id] + local_end
        # window: rows [global_end - WINDOW + 1, global_end] inclusive
        window = self.index.channels[
            global_end - WINDOW + 1 : global_end + 1
        ]  # (WINDOW, channels)
        x = window.T.astype(np.float32, copy=False)  # (channels, WINDOW)
        # Per-channel z-norm over the window. (channels, 1) broadcasting.
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + self.eps
        x = (x - mean) / std
        # Pathological case: a channel was perfectly constant in this window
        # (rare but possible for `volume == 0` on illiquid bars). std hits eps;
        # the result is 0/eps ≈ 0, which is the right "no signal" answer.
        y = self.index.labels[i].astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class Cnn1d(nn.Module):
    """1D-CNN per the PR #1 spec.

    Input shape: ``(B, 6, 60)``. Output: ``(B, 1)`` logit.
    """

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        conv1_out: int = 64,
        conv2_out: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, conv1_out, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size)
        self.head = nn.Linear(conv2_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = h.mean(dim=-1)  # global average pool over the window dim
        return self.head(h).squeeze(-1)


@dataclass
class CnnDiscovery:
    """Train, predict, and attribute a :class:`Cnn1d` over a labeled OHLCV frame.

    Mirrors :class:`quant.models.XGBDiscovery` so discover.py can treat
    them uniformly.
    """

    scale_pos_weight: float
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5  # early-stop after N epochs without val-prec@TD improvement
    top_decile_q: float = 0.10
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    random_seed: int = 42

    _model: Cnn1d | None = field(default=None, init=False, repr=False)
    _train_wall_clock_s: float | None = field(default=None, init=False, repr=False)
    _epochs_trained: int = field(default=0, init=False, repr=False)
    _best_val_precision: float = field(default=0.0, init=False, repr=False)

    @property
    def runtime_device(self) -> str:
        """Where the model actually lives, post-fit. Same self-attestation
        contract as :class:`XGBDiscovery`."""
        if self._model is None:
            raise RuntimeError("CnnDiscovery.runtime_device called before fit")
        return str(next(self._model.parameters()).device)

    @property
    def train_wall_clock_s(self) -> float | None:
        return self._train_wall_clock_s

    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    @property
    def param_count(self) -> int:
        if self._model is None:
            raise RuntimeError("CnnDiscovery.param_count called before fit")
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    # ----- core API -----

    def fit(
        self,
        train_idx: WindowIndex,
        val_idx: WindowIndex,
    ) -> "CnnDiscovery":
        torch.manual_seed(self.random_seed)
        device = torch.device(self.device)

        self._model = Cnn1d().to(device)
        opt = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.scale_pos_weight], device=device)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision and device.type == "cuda")

        train_ds = WindowDataset(train_idx)
        val_ds = WindowDataset(val_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size * 4,  # eval is forward-only, bigger fits
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        best_state: dict[str, torch.Tensor] | None = None
        epochs_since_improvement = 0
        t0 = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            running_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=self.mixed_precision and device.type == "cuda",
                ):
                    logits = self._model(xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running_loss += float(loss.item())
                n_batches += 1
            sched.step()
            train_loss = running_loss / max(n_batches, 1)

            val_prec, val_auc = self._eval_precision_at_topdecile(val_loader, device)
            improved = val_prec > self._best_val_precision + 1e-6
            print(
                f"  epoch {epoch:>2}/{self.epochs}: "
                f"train_loss={train_loss:.4f}  "
                f"val_prec@TD={val_prec:.4f}  "
                f"val_auc={val_auc:.4f}  "
                f"{'(new best)' if improved else ''}"
            )
            if improved:
                self._best_val_precision = val_prec
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.patience:
                    print(
                        f"  early stop: val_prec@TD did not improve for "
                        f"{self.patience} epochs (best {self._best_val_precision:.4f})"
                    )
                    break

            self._epochs_trained = epoch

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._train_wall_clock_s = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def _eval_precision_at_topdecile(
        self, loader: DataLoader, device: torch.device
    ) -> tuple[float, float]:
        """Walk the loader once; return (precision@top-decile, auc)."""
        from sklearn.metrics import roc_auc_score

        assert self._model is not None
        self._model.eval()
        probas = []
        labels = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=self.mixed_precision and device.type == "cuda",
            ):
                logits = self._model(xb)
            probas.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(yb.numpy())
        proba = np.concatenate(probas)
        y = np.concatenate(labels).astype(np.int8)

        n = len(proba)
        k = max(1, int(n * self.top_decile_q))
        top_idx = np.argpartition(-proba, k - 1)[:k]
        prec = float(y[top_idx].sum() / k)
        try:
            auc = float(roc_auc_score(y, proba))
        except ValueError:
            auc = float("nan")
        return prec, auc

    @torch.no_grad()
    def predict(self, index: WindowIndex) -> np.ndarray:
        """Return per-window predicted P(is_winner) aligned to ``index.endpoints``."""
        if self._model is None:
            raise RuntimeError("CnnDiscovery.predict called before fit")
        device = torch.device(self.device)
        loader = DataLoader(
            WindowDataset(index),
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        self._model.eval()
        out = []
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=self.mixed_precision and device.type == "cuda",
            ):
                logits = self._model(xb)
            out.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(out)

    # ----- attribution -----

    def attribution(
        self,
        index: WindowIndex,
        sample_size: int = 10_000,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Captum IntegratedGradients on a sample of ``index``.

        Returns ``(shap_summary, timestep_attribution)``:
        - ``shap_summary``: per-channel mean |IG|, matching the XGB
          ``shap-summary.parquet`` contract (column ``mean_abs_shap``
          re-used for cross-pipeline parser compat — see the layout doc).
        - ``timestep_attribution``: per-(channel, timestep) mean |IG|,
          for the Step 2b-only ``timestep-attribution.parquet`` artifact.
          Timestep 0 = day t-1 (most recent), 59 = day t-60 (oldest).
        """
        if self._model is None:
            raise RuntimeError("CnnDiscovery.attribution called before fit")
        from captum.attr import IntegratedGradients

        device = torch.device(self.device)
        n = index.n_windows
        sample_n = min(sample_size, n)
        rng = np.random.default_rng(self.random_seed)
        sample_idx = rng.choice(n, size=sample_n, replace=False)

        ds = WindowDataset(index)
        # Batched IG to stay in GPU memory.
        ig = IntegratedGradients(self._model)
        batch = 256
        attrs_accum = np.zeros((len(CHANNELS), WINDOW), dtype=np.float64)
        n_seen = 0
        self._model.eval()
        for start in range(0, sample_n, batch):
            chunk_idx = sample_idx[start : start + batch]
            xs = torch.stack([ds[int(j)][0] for j in chunk_idx]).to(device)
            # Baseline: per-window-zero (post-normalization, mean=0 IS the
            # baseline by construction).
            baseline = torch.zeros_like(xs)
            attrs = ig.attribute(
                xs, baselines=baseline, n_steps=32, internal_batch_size=batch
            )  # (B, channels, window)
            attrs_accum += np.abs(attrs.detach().float().cpu().numpy()).sum(axis=0)
            n_seen += xs.shape[0]
        mean_abs = attrs_accum / max(n_seen, 1)  # (channels, window)

        # Per-channel mean |IG| (channel-level shap-summary view).
        per_channel = mean_abs.mean(axis=1)  # collapse window dim
        # Direction: correlate the channel's *signed* IG against its input,
        # over the sampled batch. Positive corr → "+", negative → "-",
        # else "mixed". Same rule as XGB SHAP direction.
        directions = self._channel_directions(index, sample_idx, ig, batch, device)

        shap_summary = pl.DataFrame(
            {
                "feature_name": list(CHANNELS),
                "mean_abs_shap": per_channel.astype(np.float64),
                "direction": directions,
            }
        ).sort("mean_abs_shap", descending=True)

        # Long format: (feature, timestep, mean_abs_attribution).
        timesteps = np.arange(WINDOW, dtype=np.int64)
        # Convention from the contract: timestep 0 = day t-1 (most recent).
        # The window's last column IS t-1, so reverse the order so column 0
        # of the long table is the most recent day.
        mean_abs_recent_first = mean_abs[:, ::-1]
        timestep_long = pl.DataFrame(
            {
                "feature_name": np.repeat(list(CHANNELS), WINDOW),
                "timestep": np.tile(timesteps, len(CHANNELS)),
                "mean_abs_attribution": mean_abs_recent_first.reshape(-1).astype(np.float64),
            }
        ).sort(["feature_name", "timestep"])

        return shap_summary, timestep_long

    @torch.no_grad()
    def _channel_directions(
        self,
        index: WindowIndex,
        sample_idx: np.ndarray,
        ig: Any,
        batch: int,
        device: torch.device,
    ) -> list[str]:
        """For each channel, Pearson r between (input value) and (signed IG)
        across the sampled (window, timestep) elements. Same +/-/mixed rule
        as :meth:`quant.models.XGBDiscovery.shap_summary`.
        """
        ds = WindowDataset(index)
        # Accumulate sums for the streaming Pearson computation per channel.
        # Pearson r = (E[xy] - E[x]E[y]) / (std(x) * std(y))
        n_channels = len(CHANNELS)
        sx = np.zeros(n_channels)
        sy = np.zeros(n_channels)
        sxx = np.zeros(n_channels)
        syy = np.zeros(n_channels)
        sxy = np.zeros(n_channels)
        count = np.zeros(n_channels)

        for start in range(0, len(sample_idx), batch):
            chunk_idx = sample_idx[start : start + batch]
            xs = torch.stack([ds[int(j)][0] for j in chunk_idx]).to(device)
            baseline = torch.zeros_like(xs)
            attrs = ig.attribute(xs, baselines=baseline, n_steps=32, internal_batch_size=batch)
            xs_np = xs.detach().float().cpu().numpy()  # (B, C, W)
            at_np = attrs.detach().float().cpu().numpy()
            for c in range(n_channels):
                x = xs_np[:, c, :].ravel()
                y = at_np[:, c, :].ravel()
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                sx[c] += x.sum()
                sy[c] += y.sum()
                sxx[c] += (x * x).sum()
                syy[c] += (y * y).sum()
                sxy[c] += (x * y).sum()
                count[c] += x.size

        directions: list[str] = []
        for c in range(n_channels):
            n = count[c]
            if n < 100:
                directions.append("mixed")
                continue
            mean_x = sx[c] / n
            mean_y = sy[c] / n
            var_x = sxx[c] / n - mean_x * mean_x
            var_y = syy[c] / n - mean_y * mean_y
            cov_xy = sxy[c] / n - mean_x * mean_y
            if var_x <= 0 or var_y <= 0:
                directions.append("mixed")
                continue
            r = cov_xy / np.sqrt(var_x * var_y)
            if not np.isfinite(r):
                directions.append("mixed")
            elif r >= 0.3:
                directions.append("+")
            elif r <= -0.3:
                directions.append("-")
            else:
                directions.append("mixed")
        return directions
