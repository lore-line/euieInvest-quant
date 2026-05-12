"""Canonical peer-groups hash matching api-contract.md §5.2."""

from __future__ import annotations

import hashlib
import json


def peer_groups_hash(groups: dict[str, list[str]]) -> str:
    # Default separators on purpose: the consumer's contract test uses
    # json.dumps(..., sort_keys=True) with no separators kwarg, so the hash
    # must match Python's default ", " / ": " spacing exactly.
    canonical = {k: sorted(v) for k, v in groups.items()}
    body = json.dumps(canonical, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
