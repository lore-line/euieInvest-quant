"""Checkpoint save/load with bit-identical resume.

A checkpoint captures everything the training loop needs to pick up
where it left off:

- model state_dict
- optimizer state_dict
- LR scheduler state_dict (optional)
- GradScaler state_dict (optional, for mixed precision)
- numpy + torch + python RNG states (for deterministic resume)
- per-run extras (best_metric, best_epoch, anything custom)

Cadence is "the sooner of per-epoch end OR every N minutes". The N-min
guardrail matters for long-epoch jobs (e.g. Track F's masked-bar
pretraining, where one epoch over 2.4M windows may take 30+ min).

Atomic writes via write-temp-then-rename — a reader (or a torn process)
never sees a half-written checkpoint. Retries on PermissionError because
on Windows + Docker-volume-mount + cloud-sync (Nextcloud / OneDrive /
Dropbox), the sync engine can hold a transient handle on the target,
making `os.replace` fail intermittently. The retry-then-unlink-then-
rename fallback survives that.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


_RENAME_RETRIES = 5
_RENAME_RETRY_DELAY_S = 1.0


def _atomic_replace(tmp: Path, target: Path) -> None:
    """``tmp.replace(target)`` with PermissionError-tolerant retries.

    Pure ``os.replace`` is atomic on POSIX and *almost* atomic on
    Windows — but cloud-sync engines holding a transient file handle on
    the target make it fail with WinError 5 / PermissionError. Retry a
    few times with short sleeps; if that doesn't clear, unlink the
    target and rename. The latter has a sub-millisecond window where
    the target doesn't exist — a reader hitting that window sees no
    file rather than a torn file, which is the failure mode we want.
    """
    last_exc: Exception | None = None
    for attempt in range(_RENAME_RETRIES):
        try:
            tmp.replace(target)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(_RENAME_RETRY_DELAY_S * (attempt + 1))
    # Last-resort fallback: unlink target then rename. Brief unprotected window.
    try:
        if target.exists():
            target.unlink()
        os.rename(tmp, target)
        return
    except OSError:
        pass
    assert last_exc is not None
    raise last_exc


__all__ = ["CheckpointManager"]


@dataclass
class CheckpointManager:
    """Per-run checkpoint coordinator.

    Parameters
    ----------
    dir:
        Directory the checkpoint lives in. Conventionally the run
        output dir (``runs/<date>-<pipeline_step>/``).
    min_interval_s:
        Minimum wall-clock seconds between checkpoint writes. The
        next checkpoint will fire at the SOONER of: (a) ``min_interval_s``
        elapsed since the last write, or (b) explicit ``save_now=True``
        from the caller (e.g. epoch boundary, graceful-shutdown signal).
        Default 1800 (30 min) per PR #1 issuecomment-4436101547.
    filename:
        Checkpoint filename. ``latest.pt`` writes are atomic
        (rename-from-tmp); ``best.pt`` is a separate file the caller
        writes via ``save_best()`` when a new best metric appears.
    """

    dir: Path
    min_interval_s: float = 1800.0
    filename: str = "latest.pt"

    _last_save_ts: float = field(default=0.0, init=False, repr=False)
    _last_save_epoch: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ----- save -----

    def should_save(self, now: float | None = None) -> bool:
        """``True`` if ``min_interval_s`` has elapsed since the last save."""
        if now is None:
            now = time.monotonic()
        return (now - self._last_save_ts) >= self.min_interval_s

    def save(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        extras: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> Path:
        """Atomic write of a full checkpoint to ``self.dir / filename``."""
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "extras": extras or {},
            "checkpoint_format_version": 1,
        }
        target = self.dir / (filename or self.filename)
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, tmp)
        _atomic_replace(tmp, target)
        self._last_save_ts = time.monotonic()
        self._last_save_epoch = epoch
        return target

    def save_best(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Write ``best.pt`` alongside the rolling ``latest.pt``."""
        return self.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            extras=extras,
            filename="best.pt",
        )

    # ----- load / resume -----

    def latest_path(self) -> Path | None:
        p = self.dir / self.filename
        return p if p.exists() else None

    def best_path(self) -> Path | None:
        p = self.dir / "best.pt"
        return p if p.exists() else None

    @staticmethod
    def load(
        path: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        map_location: str | torch.device | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint into the given handles; return the full payload."""
        # weights_only=False so we can deserialize optimizer/scheduler state
        # dicts (torch >=2.6 defaults to True, which would reject those).
        # The file is internal to this repo; we trust its contents.
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if restore_rng:
            if payload.get("torch_rng_state") is not None:
                torch.set_rng_state(payload["torch_rng_state"].cpu())
            if (
                payload.get("torch_cuda_rng_state_all") is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
            if payload.get("numpy_rng_state") is not None:
                np.random.set_state(payload["numpy_rng_state"])
            if payload.get("python_rng_state") is not None:
                random.setstate(payload["python_rng_state"])
        return payload
