"""Tests for ``quant.train.CheckpointManager`` and ``RunStatus``.

Smoke-only — sufficient to catch torch-version regressions in the
state-dict save/load contract before launching a 24h Track F run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from quant.train import CheckpointManager, RunStatus


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_save_load_roundtrip_restores_weights_exactly(tmp_path: Path) -> None:
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    # Take one optimizer step so optimizer state is non-trivial.
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    sched.step()

    ckpt = CheckpointManager(dir=tmp_path)
    saved = ckpt.save(epoch=7, model=model, optimizer=opt, scheduler=sched, extras={"foo": "bar"})
    assert saved.exists()

    # New model + optimizer; load checkpoint into them.
    model2 = _TinyModel()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1)
    payload = CheckpointManager.load(saved, model=model2, optimizer=opt2, scheduler=sched2)

    for k in model.state_dict():
        torch.testing.assert_close(model.state_dict()[k], model2.state_dict()[k])
    assert payload["epoch"] == 7
    assert payload["extras"] == {"foo": "bar"}


def test_save_is_atomic_no_partial_file(tmp_path: Path) -> None:
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = CheckpointManager(dir=tmp_path)
    saved = ckpt.save(epoch=1, model=model, optimizer=opt)
    # After save, no .tmp file should remain.
    assert saved.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_should_save_respects_min_interval(tmp_path: Path) -> None:
    """should_save() fires only when min_interval_s has elapsed since the
    last save. Use case: mid-epoch periodic save. Epoch-end save is
    unconditional via direct save() — should_save() is the throttle for
    in-the-batch-loop checks.
    """
    ckpt = CheckpointManager(dir=tmp_path, min_interval_s=10.0)
    # Simulate "save at time 5".
    ckpt._last_save_ts = 5.0
    # 9s elapsed → not yet.
    assert not ckpt.should_save(now=14.0)
    # 10s elapsed → fire.
    assert ckpt.should_save(now=15.0)
    # 20s elapsed → still fires until next save.
    assert ckpt.should_save(now=25.0)


def test_rng_state_restoration_makes_resume_deterministic(tmp_path: Path) -> None:
    """A fresh randn() after save+load must match the original sequence."""
    torch.manual_seed(123)
    _ = torch.randn(5)  # consume initial draws

    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = CheckpointManager(dir=tmp_path)
    saved = ckpt.save(epoch=1, model=model, optimizer=opt)

    expected = torch.randn(7)  # the "next" draws after save

    # Restore and compare.
    model2 = _TinyModel()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    CheckpointManager.load(saved, model=model2, optimizer=opt2)
    actual = torch.randn(7)
    torch.testing.assert_close(actual, expected)


def test_run_status_writes_valid_json(tmp_path: Path) -> None:
    status = RunStatus(
        dir=tmp_path,
        run_id="2026-05-13-001",
        pipeline_step="step3f_foundation_pretrain",
        epoch_total=50,
    )
    status.record_checkpoint(epoch=2)
    status.update(state="training", epoch_current=3)
    doc = RunStatus.read(tmp_path / "status.json")
    assert doc["run_id"] == "2026-05-13-001"
    assert doc["pipeline_step"] == "step3f_foundation_pretrain"
    assert doc["state"] == "training"
    assert doc["epoch_current"] == 3
    assert doc["epoch_total"] == 50
    assert doc["last_checkpoint_epoch"] == 2
    assert doc["last_checkpoint_at"] is not None
    assert doc["pid"] > 0
    assert doc["error"] is None


def test_run_status_checkpoint_persists_across_updates(tmp_path: Path) -> None:
    """Once record_checkpoint() fires, subsequent updates keep the value
    until a newer record_checkpoint() overwrites it."""
    status = RunStatus(dir=tmp_path, run_id="r", pipeline_step="x", epoch_total=10)
    status.record_checkpoint(epoch=4)
    status.update(state="training", epoch_current=5)
    doc1 = RunStatus.read(tmp_path / "status.json")
    assert doc1["last_checkpoint_epoch"] == 4

    # Non-checkpoint update: still says epoch 4.
    status.update(state="training", epoch_current=6)
    doc2 = RunStatus.read(tmp_path / "status.json")
    assert doc2["last_checkpoint_epoch"] == 4
    assert doc2["last_checkpoint_at"] == doc1["last_checkpoint_at"]

    # New checkpoint: bumps both. Timestamp ordering is monotonic (>= prev),
    # but we don't assert strict inequality — two adjacent record_checkpoint
    # calls can land in the same millisecond on a fast machine.
    status.record_checkpoint(epoch=8)
    status.update(state="training", epoch_current=9)
    doc3 = RunStatus.read(tmp_path / "status.json")
    assert doc3["last_checkpoint_epoch"] == 8
    assert doc3["last_checkpoint_at"] >= doc1["last_checkpoint_at"]


def test_run_status_eta_uses_recent_epoch_durations(tmp_path: Path) -> None:
    status = RunStatus(
        dir=tmp_path,
        run_id="r",
        pipeline_step="step3a_xgb_rule_extraction",
        epoch_total=10,
    )
    # Simulate three epoch durations of 1s each.
    status._epoch_durations = [10.0, 10.0, 10.0]
    eta = status._eta_estimate(epoch_current=4)
    # 6 remaining × 10s avg = 60s.
    assert eta == 60.0


def test_run_status_atomic_no_partial_file(tmp_path: Path) -> None:
    status = RunStatus(
        dir=tmp_path,
        run_id="r",
        pipeline_step="x",
    )
    status.update(state="training", epoch_current=1)
    assert (tmp_path / "status.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_replace_retries_on_permission_error(tmp_path: Path, monkeypatch) -> None:
    """The PermissionError-tolerant replace must succeed when transient
    PermissionErrors are interleaved with eventual success.

    Cloud-sync engines (Nextcloud / OneDrive / Dropbox) on Windows hold
    transient file handles on hot-modified files; the bare ``os.replace``
    fails with PermissionError but a retry after a brief sleep
    usually succeeds. This regression-tests that pattern.
    """
    from quant.train.checkpoint import _atomic_replace

    tmp = tmp_path / "checkpoint.pt.tmp"
    target = tmp_path / "checkpoint.pt"
    tmp.write_bytes(b"new-content")
    target.write_bytes(b"old-content")

    real_replace = Path.replace
    call_count = {"n": 0}

    def flaky_replace(self, target_path):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise PermissionError(13, "Permission denied")
        return real_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    _atomic_replace(tmp, target)
    assert target.read_bytes() == b"new-content"
    assert not tmp.exists()
    assert call_count["n"] >= 3, "expected at least one retry before success"


def test_atomic_replace_falls_back_to_unlink_then_rename(tmp_path: Path, monkeypatch) -> None:
    """If every retry of os.replace fails, the final fallback unlinks
    the target then renames. Sub-millisecond window where target
    doesn't exist; that's accepted over total failure.
    """
    from quant.train.checkpoint import _atomic_replace

    tmp = tmp_path / "checkpoint.pt.tmp"
    target = tmp_path / "checkpoint.pt"
    tmp.write_bytes(b"new-content")
    target.write_bytes(b"old-content")

    def always_fail(self, target_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", always_fail)
    _atomic_replace(tmp, target)
    assert target.read_bytes() == b"new-content"
    assert not tmp.exists()
