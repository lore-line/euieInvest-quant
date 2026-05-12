"""Discovery pipeline entrypoint — 5-step orchestrator.

Phase 1 scaffold: each numbered step raises NotImplementedError. Fill in
the feature engineering modules under ``src/quant/features/`` first; see
CLAUDE.md §5 for the full methodology.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Winner-fingerprint discovery pipeline (CLAUDE.md §5)"
    )
    p.add_argument("--train-end", type=date.fromisoformat, default=date(2023, 12, 31))
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    return p.parse_args(argv)


def step1_build_features(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step1_build_features — fill in the feature "
        "modules under src/quant/features/ (price/volume/volatility/momentum/"
        "relative/gaps/behavioral). See CLAUDE.md §7."
    )


def step2_supervised_discovery(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step2_supervised_discovery — train XGBDiscovery "
        "with scale_pos_weight from train-set imbalance, emit SHAP summary. "
        "See CLAUDE.md §5 step 2 and §9."
    )


def step3_cluster_winners(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step3_cluster_winners — KMeans on winner-only "
        "rows for k in (3,5,8); select by silhouette. See CLAUDE.md §5 step 3."
    )


def step4_counterfactuals(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step4_counterfactuals — closest non-winners "
        "per winner cluster; report feature deltas. See CLAUDE.md §5 step 4."
    )


def step5_tier3_comparison(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "scripts/discover.py:step5_tier3_comparison — overlap, recall, and "
        "missed-winners vs the anomaly_flags baseline. See CLAUDE.md §5 step 5."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step1_build_features(args)
    step2_supervised_discovery(args)
    step3_cluster_winners(args)
    step4_counterfactuals(args)
    step5_tier3_comparison(args)


if __name__ == "__main__":
    main()
