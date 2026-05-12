"""Unsupervised clustering of winning rows.

KMeans k in {3, 5, 8}; select best by silhouette. See CLAUDE.md §5 step 3.
"""
from __future__ import annotations

from typing import Any

import polars as pl

__all__ = ["cluster_winners"]


def cluster_winners(
    features: pl.DataFrame, ks: tuple[int, ...] = (3, 5, 8)
) -> dict[str, Any]:
    """Cluster winner-feature rows and return silhouettes + best-k assignments.

    Returns
    -------
    dict
        ``{"silhouettes": {k: score}, "best_k": int, "assignments": pl.Series}``
    """
    raise NotImplementedError(
        "src/quant/clusters/winners.py: cluster_winners — KMeans for "
        "k in (3,5,8) with silhouette selection; see CLAUDE.md §5 step 3."
    )
