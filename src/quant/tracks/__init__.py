"""Phase A discovery tracks (CLAUDE.md §12 + PR #1 issuecomment-4436101547).

Each track is a runnable pipeline that produces its own outputs in
``euieInvest-reports/runs/<date>-<pipeline_step>/``. Tracks are
mutually independent except where noted in the brief.

All 12 tracks are implemented; Tracks 7-12 are gated on Track F's
encoder landing.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path


_STEP_PREFIX_RE = re.compile(r"^(step\d+[a-z]?)")


def short_step(pipeline_step: str) -> str:
    """Extract the short step ID from a full pipeline_step name.

    Examples:
        "step2_supervised_discovery"    → "step2"
        "step2b_dl_discovery_cnn"       → "step2b"
        "step3a_xgb_rule_extraction"    → "step3a"
        "step3f_foundation_pretrain"    → "step3f"

    Used in ``run_id`` to disambiguate runs on the same date — see
    PR #1 issuecomment-4436523651 (run_id collision finding).
    """
    m = _STEP_PREFIX_RE.match(pipeline_step)
    if not m:
        return pipeline_step
    return m.group(1)


def make_run_id(run_date_str: str, pipeline_step: str, sequence: int = 1) -> str:
    """Build a run_id per the post-2026-05-13 convention:
    ``YYYY-MM-DD-NNN-<step_short>``."""
    return f"{run_date_str}-{sequence:03d}-{short_step(pipeline_step)}"


_DATE_PIPELINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")


def resolve_run_dir(
    pipeline_step: str,
    out_dir_arg: Path | None,
    repo_root: Path,
    resume_checkpoint_filename: str = "latest.pt",
) -> tuple[Path, str]:
    """Resolve the run directory + run_date_str for a training script.

    Resolution order:

    1. **Explicit ``--out-dir``** — caller passed it via CLI; use verbatim.
       run_date_str extracted from the leading ``YYYY-MM-DD`` prefix in the
       dir name.

    2. **Resume-friendly auto-detect** — look for the most recent existing
       ``runs/YYYY-MM-DD-<pipeline_step>/`` that contains
       ``<resume_checkpoint_filename>``. If found, reuse it. This survives
       UTC date rollover during resumes — the original bug was that a
       container restarted after midnight would generate a NEW
       date-prefixed dir and fail to find the previous day's checkpoint.

    3. **Fresh run** — create ``runs/<today>-<pipeline_step>/`` and use it.

    Args:
        pipeline_step: the canonical pipeline_step string (e.g.
            ``"step3f_foundation_pretrain_v2_temporal"``)
        out_dir_arg: ``args.out_dir`` from the script's argparse (or None).
        repo_root: the repo root (typically ``Path(__file__).resolve().parents[3]``).
        resume_checkpoint_filename: filename whose existence indicates a
            resumable run (default ``latest.pt`` for the GPU training tracks;
            could be ``rules.parquet`` etc. for one-shot CPU tracks where
            "resume" really means "run dir already complete").

    Returns:
        (run_dir, run_date_str)

    See PR #1 issuecomment-* (Track F-v2 2026-05-15 03:01 UTC midnight
    restart bug — script generated runs/2026-05-15-... and didn't find
    latest.pt in runs/2026-05-14-..., started from epoch 0 instead of
    resuming from epoch 38).
    """
    if out_dir_arg is not None:
        run_dir = out_dir_arg if out_dir_arg.is_absolute() else (repo_root / out_dir_arg)
        run_date_str = run_dir.name[:10]
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, run_date_str

    runs_root = repo_root / "runs"
    if runs_root.exists():
        candidates: list[tuple[str, Path]] = []
        target_suffix = f"-{pipeline_step}"
        for d in runs_root.iterdir():
            if not d.is_dir():
                continue
            m = _DATE_PIPELINE_RE.match(d.name)
            if not m:
                continue
            if m.group(2) != pipeline_step:
                continue
            ckpt = d / resume_checkpoint_filename
            if ckpt.exists():
                candidates.append((m.group(1), d))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            run_date_str, run_dir = candidates[0]
            # Don't recreate; it already exists. mkdir(exist_ok=True) is safe.
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir, run_date_str

    # Fresh run dir at today's date.
    run_date_str = date.today().isoformat()
    run_dir = runs_root / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_date_str


def verify_encoder_sha(
    encoder_path: Path,
    expected_sha: str | None = None,
) -> str:
    """Compute SHA256 of an encoder.pt and optionally enforce a pinned value.

    Defense against silent encoder swaps — a downstream track that was
    authorized against encoder ``X`` should fail loudly (not silently
    consume a different encoder) if ``encoder.pt`` at the same path has
    been overwritten by a re-run of ``foundation_pretrain.py`` with
    different settings.

    Usage pattern in a downstream track::

        encoder_path = ...  # resolved from --encoder-path or auto-detect
        encoder_sha = verify_encoder_sha(
            encoder_path,
            expected_sha=args.expected_encoder_sha,  # CLI flag, may be None
        )
        # ... record encoder_sha in this track's own manifest.json for traceability

    Args:
        encoder_path: Path to the encoder ``.pt`` file. Must already be resolved
            to absolute or relative-to-cwd; this function does not resolve paths.
        expected_sha: Optional pinned hex SHA256. Tolerates both
            ``"sha256:abc..."`` and bare ``"abc..."`` forms (case-insensitive).
            When ``None``, no enforcement — just returns the actual SHA so the
            caller can record it for posterity.

    Returns:
        The actual SHA256 of ``encoder_path`` as a lowercase hex string (64 chars).

    Raises:
        ValueError: if ``expected_sha`` is set and doesn't match the actual SHA.
    """
    sha = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    if expected_sha is not None:
        expected_clean = expected_sha.removeprefix("sha256:").strip().lower()
        if sha != expected_clean:
            raise ValueError(
                f"Encoder SHA mismatch for {encoder_path}:\n"
                f"  expected: sha256:{expected_clean}\n"
                f"  actual:   sha256:{sha}\n"
                f"The encoder file may have been swapped or corrupted since this "
                f"run was authorized. If this is intentional (e.g. you re-ran "
                f"foundation_pretrain and want the new encoder), update the "
                f"--expected-encoder-sha argument."
            )
    return sha

