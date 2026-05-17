"""1D CNN model for breakout_seq_v1 — DL Angle 2.

Architecture: lightweight 1D conv stack on (B, 6_channels, 60_timesteps)
OHLCV windows, mean-pooled → 2-layer MLP head → binary logit. Designed
to be CPU-trainable for smoke tests but really meant for GPU at full
scale (1M+ training rows).

Saliency for interpretability: gradient w.r.t. input gives per-(channel,
timestep) attribution; sum-over-channels gives a 60-day timeline of
"which days mattered for this prediction." Useful for surfacing in
prompts/manifests so Claude knows what the model is keying on.

Per-window z-normalization is done in this model's forward (not in the
data pipeline) so the predict-time embedding pipeline can pass raw
OHLCV windows directly.

Parameter count: ~200K, kept small to:
  - Train on CPU in <2 hrs on the smoke-test sample
  - Avoid overfitting on the ~1M train rows after 60-d windowing
  - Match the encoder-symbol-axis finding that the predictive signal is
    likely a low-dimensional manifold (don't need a large model for it)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.data.windows import CHANNELS, WINDOW


N_CHANNELS = len(CHANNELS)  # 6: open, high, low, close, close_adj, volume
SEQ_LEN = WINDOW  # 60


class BreakoutSeqCNN(nn.Module):
    """1D CNN classifier on 60-day OHLCV windows.

    Forward:
      x: (B, n_channels, seq_len) float32
      → per-window z-norm (mean/std across the seq_len axis)
      → Conv1d(C=6 → 32, k=7) + BN + GELU + MaxPool(2)  # → (B, 32, 30)
      → Conv1d(32 → 64, k=5) + BN + GELU + MaxPool(2)   # → (B, 64, 15)
      → Conv1d(64 → 128, k=3) + BN + GELU + GAP         # → (B, 128)
      → Linear(128 → 64) + GELU + Dropout(0.3)
      → Linear(64 → 1)                                  # logit
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        seq_len: int = SEQ_LEN,
        c1: int = 32, c2: int = 64, c3: int = 128,
        head_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        # Per-window z-norm: subtract per-channel mean, divide by per-channel std
        # (across seq_len). Numerically stable: clamp std to >= 1e-6.

        self.conv1 = nn.Conv1d(n_channels, c1, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(c1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(c3)
        # GAP via mean over time

        self.fc1 = nn.Linear(c3, head_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(head_hidden, 1)

    def _z_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window per-channel z-norm. x: (B, C, T)."""
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B,) logits."""
        x = self._z_norm(x)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = F.gelu(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = F.gelu(self.bn3(self.conv3(h)))
        # Global avg pool over time
        h = h.mean(dim=2)  # (B, c3)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        logit = self.fc_out(h).squeeze(-1)  # (B,)
        return logit

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forward + sigmoid → probabilities."""
        return torch.sigmoid(self.forward(x))

    def saliency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute |grad of output logit w.r.t. input|, summed over channels.

        Returns (B, T) — per-timestep attribution. Useful for surfacing
        "which days in the 60-day window mattered" alongside emitted signals.

        x: (B, C, T). Caller should detach + clone if x is part of a graph.
        """
        x = x.clone().detach().requires_grad_(True)
        logits = self.forward(x)
        logits.sum().backward()
        # x.grad shape: (B, C, T) → sum-abs over channels → (B, T)
        return x.grad.abs().sum(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
