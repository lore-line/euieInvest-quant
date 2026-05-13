"""status.json — per-run progress beacon.

Each long-running training job writes a ``status.json`` into its run
directory and updates it on every checkpoint and on lifecycle
transitions. A different agent session (or a human glancing at the
filesystem) can read it without attaching to the training process.

Schema (PR #1 issuecomment-4436101547, §Operational hygiene):

    {
      "run_id":               str,       # e.g. "2026-05-13-001"
      "pipeline_step":        str,       # e.g. "step3f_foundation_pretrain"
      "state":                str,       # "training" | "paused" | "done" | "failed"
      "epoch_current":        int,
      "epoch_total":          int,
      "started_at":           str,       # ISO 8601 UTC
      "last_checkpoint_at":   str | null,
      "last_checkpoint_epoch": int | null,
      "eta_estimate_s":       float | null,
      "pid":                  int,
      "host":                 str,
      "error":                str | null
    }

All writes are atomic (write-temp-then-rename). The graceful-interrupt
handler flips ``state`` to ``"paused"`` (Ctrl-C during training) or
``"failed"`` (uncaught exception) before exit so a future agent
session knows the difference between a clean pause and a crash.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


__all__ = ["RunStatus", "install_graceful_interrupt"]


def _utc_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision (`...Z` suffix).

    Millisecond resolution distinguishes two near-simultaneous
    checkpoint writes — useful when reconstructing crash timelines from
    status.json + log timestamps.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


@dataclass
class RunStatus:
    """Writer for ``status.json``. Construct once per training process."""

    dir: Path
    run_id: str
    pipeline_step: str
    epoch_total: int = 0
    filename: str = "status.json"

    _started_at: str = field(default_factory=_utc_iso, init=False, repr=False)
    _start_monotonic: float = field(default_factory=time.monotonic, init=False, repr=False)
    _epoch_durations: list[float] = field(default_factory=list, init=False, repr=False)
    _last_epoch_start: float = field(default=0.0, init=False, repr=False)
    _last_checkpoint_epoch: int | None = field(default=None, init=False, repr=False)
    _last_checkpoint_at: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last_epoch_start = time.monotonic()

    # ----- checkpoint tracking -----

    def record_checkpoint(self, epoch: int) -> None:
        """Record that a new checkpoint was just saved at ``epoch``.

        Persists across subsequent ``update()`` calls — the status.json
        keeps showing the latest checkpoint info even on non-checkpoint
        progress ticks, until a newer ``record_checkpoint`` overwrites
        it.
        """
        self._last_checkpoint_epoch = epoch
        self._last_checkpoint_at = _utc_iso()

    # ----- write -----

    def update(
        self,
        *,
        state: str,
        epoch_current: int = 0,
        error: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Atomic write of the current status. The most recent checkpoint
        (set via :meth:`record_checkpoint`) is preserved across calls."""
        eta = self._eta_estimate(epoch_current)
        doc: dict[str, Any] = {
            "run_id": self.run_id,
            "pipeline_step": self.pipeline_step,
            "state": state,
            "epoch_current": epoch_current,
            "epoch_total": self.epoch_total,
            "started_at": self._started_at,
            "last_checkpoint_at": self._last_checkpoint_at,
            "last_checkpoint_epoch": self._last_checkpoint_epoch,
            "eta_estimate_s": eta,
            "pid": os.getpid(),
            "host": platform.node(),
            "error": error,
        }
        if extras:
            doc.update(extras)
        target = self.dir / self.filename
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        tmp.replace(target)

    def mark_epoch_complete(self) -> None:
        """Record an epoch's duration for ETA estimation. Call at epoch end."""
        now = time.monotonic()
        self._epoch_durations.append(now - self._last_epoch_start)
        self._last_epoch_start = now

    def _eta_estimate(self, epoch_current: int) -> float | None:
        """Naive ETA: mean recent epoch duration × remaining epochs.

        Uses the last 3 epoch durations (or fewer if we don't have 3 yet)
        to dampen noise from a slow first epoch (jit/cache warmup).
        """
        if not self._epoch_durations or self.epoch_total <= 0:
            return None
        recent = self._epoch_durations[-3:]
        avg = sum(recent) / len(recent)
        remaining = max(self.epoch_total - epoch_current, 0)
        return round(avg * remaining, 1)

    # ----- read -----

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        """Read a status.json from disk. Useful for cross-session checks."""
        return json.loads(Path(path).read_text())


def install_graceful_interrupt(
    on_interrupt: callable,  # type: ignore[valid-type]
) -> None:
    """Install a SIGINT (Ctrl-C) handler that calls ``on_interrupt`` once,
    then restores the default behavior (so a second Ctrl-C still kills hard).

    ``on_interrupt`` is responsible for flushing a final checkpoint and
    flipping the status.json ``state`` to ``"paused"``. The training
    loop should check a flag the handler sets, save, and exit cleanly.
    """
    fired = {"count": 0}

    def _handler(signum: int, frame: Any) -> None:
        fired["count"] += 1
        if fired["count"] >= 2:
            # Restore default and re-raise so a second Ctrl-C terminates.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)
            return
        try:
            on_interrupt()
        except Exception:  # noqa: BLE001 — never let the handler itself raise
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)

    signal.signal(signal.SIGINT, _handler)
