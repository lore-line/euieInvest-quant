"""Long-running training infrastructure for Phase A.

CLAUDE.md §12 (post-2026-05-13 Phase A pivot) and PR #1
issuecomment-4436101547: every training job in this sprint must
support clean pause/resume, write a status.json, and survive
agent-session detach.

Public surface:
- ``CheckpointManager`` — atomic save + load of (model, optimizer,
  scheduler, scaler, rng-states) with periodic-or-per-epoch cadence
- ``RunStatus`` — atomic status.json writer + reader for cross-session
  progress checks
- ``install_graceful_interrupt`` — SIGINT handler that flips a flag the
  training loop checks at next iteration, then saves a final checkpoint
  before exit
"""
from __future__ import annotations

from quant.train.checkpoint import CheckpointManager
from quant.train.status import RunStatus, install_graceful_interrupt

__all__ = ["CheckpointManager", "RunStatus", "install_graceful_interrupt"]
