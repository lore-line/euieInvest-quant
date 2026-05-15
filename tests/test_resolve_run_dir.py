"""Tests for ``quant.tracks.resolve_run_dir``.

Pins the resume-friendly run-dir resolution behavior introduced after
the Track F-v2 2026-05-15 UTC midnight restart bug — a container that
trained 38 epochs on day D restarted on day D+1 and generated a fresh
``runs/D+1-<pipeline_step>/`` instead of resuming the existing
``runs/D-<pipeline_step>/latest.pt``, losing all training state.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quant.tracks import resolve_run_dir


def test_explicit_out_dir_used_verbatim(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    explicit = tmp_path / "runs" / "2024-01-15-step3f_foundation_pretrain"
    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3f_foundation_pretrain",
        out_dir_arg=explicit,
        repo_root=tmp_path,
    )
    assert run_dir == explicit
    assert run_date_str == "2024-01-15"
    assert run_dir.exists()  # mkdir(exist_ok=True) was called


def test_relative_out_dir_resolves_against_repo_root(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3h_prototype_learning",
        out_dir_arg=Path("runs/2025-07-04-step3h_prototype_learning"),
        repo_root=tmp_path,
    )
    assert run_dir == tmp_path / "runs/2025-07-04-step3h_prototype_learning"
    assert run_date_str == "2025-07-04"


def test_no_existing_dir_creates_fresh_today_dated(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3f_foundation_pretrain",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    today = date.today().isoformat()
    assert run_date_str == today
    assert run_dir.name == f"{today}-step3f_foundation_pretrain"
    assert run_dir.exists()


def test_existing_dir_with_checkpoint_reused(tmp_path: Path) -> None:
    """The core resume-friendly behavior. The fix for the v2 midnight bug."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    existing = runs_root / "2026-05-14-step3f_foundation_pretrain_v2_temporal"
    existing.mkdir()
    (existing / "latest.pt").write_bytes(b"fake-checkpoint")

    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3f_foundation_pretrain_v2_temporal",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    assert run_dir == existing
    assert run_date_str == "2026-05-14"


def test_existing_dir_without_checkpoint_ignored(tmp_path: Path) -> None:
    """A run dir with manifest but no latest.pt isn't a resume candidate —
    that run already completed. New run should get a fresh today-dated dir."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    completed = runs_root / "2024-06-15-step3f_foundation_pretrain"
    completed.mkdir()
    (completed / "manifest.json").write_bytes(b'{"epochs_trained": 50}')
    (completed / "encoder.pt").write_bytes(b"fake")
    # NO latest.pt — training completed, checkpoint was cleaned up

    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3f_foundation_pretrain",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    today = date.today().isoformat()
    assert run_date_str == today
    assert run_dir != completed


def test_multiple_resumable_dirs_picks_most_recent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    older = runs_root / "2024-01-01-step3h_prototype_learning"
    newer = runs_root / "2024-12-31-step3h_prototype_learning"
    older.mkdir()
    newer.mkdir()
    (older / "latest.pt").write_bytes(b"a")
    (newer / "latest.pt").write_bytes(b"b")

    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3h_prototype_learning",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    assert run_dir == newer
    assert run_date_str == "2024-12-31"


def test_pipeline_step_exact_match(tmp_path: Path) -> None:
    """``step3f_foundation_pretrain`` and ``step3f_foundation_pretrain_v2_temporal``
    must NOT cross-match — different pipeline_steps live in different dirs."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    v1 = runs_root / "2024-01-01-step3f_foundation_pretrain"
    v1.mkdir()
    (v1 / "latest.pt").write_bytes(b"v1")

    # Looking for the v2 variant — must NOT pick up the v1 dir
    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3f_foundation_pretrain_v2_temporal",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    today = date.today().isoformat()
    assert run_date_str == today, "v2 should NOT inherit v1's run dir"
    assert run_dir != v1


def test_custom_resume_filename(tmp_path: Path) -> None:
    """For non-GPU tracks, the resume marker might be a different file."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    existing = runs_root / "2024-08-08-step4_walkforward_validation"
    existing.mkdir()
    (existing / "rule-validation.parquet").write_bytes(b"fake")

    # With default marker (latest.pt) → no match, fresh dir
    rd1, _ = resolve_run_dir(
        pipeline_step="step4_walkforward_validation",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    today = date.today().isoformat()
    assert rd1.name == f"{today}-step4_walkforward_validation"

    # With custom marker → matches the existing
    rd2, rds2 = resolve_run_dir(
        pipeline_step="step4_walkforward_validation",
        out_dir_arg=None,
        repo_root=tmp_path,
        resume_checkpoint_filename="rule-validation.parquet",
    )
    assert rd2 == existing
    assert rds2 == "2024-08-08"


def test_runs_root_missing_creates_fresh(tmp_path: Path) -> None:
    """If the repo's runs/ dir doesn't exist yet, create one and the new run."""
    # tmp_path/runs doesn't exist
    run_dir, run_date_str = resolve_run_dir(
        pipeline_step="step3f_foundation_pretrain",
        out_dir_arg=None,
        repo_root=tmp_path,
    )
    today = date.today().isoformat()
    assert run_date_str == today
    assert run_dir.exists()
    assert (tmp_path / "runs").exists()
