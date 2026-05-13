"""Phase A discovery tracks (CLAUDE.md §12 + PR #1 issuecomment-4436101547).

Each track is a runnable pipeline that produces its own outputs in
``euieInvest-reports/runs/<date>-<pipeline_step>/``. Tracks are
mutually independent except where noted in the brief.

All 12 tracks are implemented; Tracks 7-12 are gated on Track F's
encoder landing.
"""
from __future__ import annotations

import re


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

