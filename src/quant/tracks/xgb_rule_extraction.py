"""Track 1 — XGB rule extraction.

Walks every root-to-leaf path of the 400-tree XGB Step 2 model;
aggregates paths into deduplicated conjunctive rules; evaluates each
rule on the holdout window; filters by lift / coverage / precision;
ranks by ``lift × coverage``.

Output (per docs/reports-repo-layout.md):

  runs/<date>-step3a_xgb_rule_extraction/
    manifest.json
    rules.parquet — (rule_id, conditions_json, n_conditions,
                      coverage_n, coverage_pct, precision, lift,
                      example_symbol_dates_json)

The brief's filter (PR #1 issuecomment-4436101547 §Track 1):
  lift ≥ 1.5  AND  coverage ≥ 0.5% of holdout  AND  precision ≥ 35%

Run via:

  pwsh scripts\\ops\\quant-start.ps1 -Track step3a_xgb_rule_extraction

or directly:

  python -m quant.tracks.xgb_rule_extraction \\
      --model runs/2026-05-12/model.json \\
      --features data/features/features.parquet

CPU only; ~30s on the cleaned 2.4M-row dataset (the polars feature
load is the long pole, not the tree walk or the rule evaluation).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from quant.train import RunStatus, install_graceful_interrupt
from quant.tracks import make_run_id

__all__ = ["Condition", "Rule", "extract_paths", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Columns excluded from the feature matrix — mirror scripts/discover.py.
# Must match exactly so the recovered feature order lines up with the
# XGB booster's internal f0..fN naming.
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"symbol", "date", "open", "high", "low", "close", "close_adj", "volume", "is_winner"}
)


def _replay_feature_selection(labeled: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Apply the same bool→int8 + str→one-hot transforms scripts/discover.py
    uses before XGB training, then return the feature column list in the
    order XGB saw them. Pairs with index → name for the booster's f0..fN
    remapping.

    Caller must pass a frame already filtered to non-null is_winner so
    the unique string-feature levels match training-time.
    """
    feature_cols = [c for c in labeled.columns if c not in _NON_FEATURE_COLS]
    bool_cols = [c for c in feature_cols if labeled[c].dtype == pl.Boolean]
    if bool_cols:
        labeled = labeled.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])
    str_cols = [c for c in feature_cols if labeled[c].dtype == pl.Utf8]
    if str_cols:
        labeled = labeled.to_dummies(columns=str_cols)
    feature_cols = [c for c in labeled.columns if c not in _NON_FEATURE_COLS]
    return labeled, feature_cols

# ----- rule schema -----


@dataclass(frozen=True)
class Condition:
    """One non-leaf split in a tree path.

    ``op`` is the operator that holds on this branch of the split:
    ``"<"`` for the "yes" branch (xgb's default split goes left when
    feature < threshold), ``">="`` for the "no" branch.
    """

    feature: str
    op: str  # "<" or ">="
    threshold: float

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:g}"


@dataclass(frozen=True)
class Rule:
    """A conjunction of conditions. Order is canonicalized (sorted by
    feature, then op, then threshold) AND redundant inequalities on the
    same feature are collapsed to the tightest bound — so two trees
    that split the same feature multiple times along a path produce
    the same Rule.

    Example: a path with conditions ``[f1 >= 0.5, f1 >= 0.8, f2 < 3]``
    canonicalizes to ``[f1 >= 0.8, f2 < 3]`` (the 0.5 bound is implied
    by the 0.8 bound and contributes no information).
    """

    conditions: tuple[Condition, ...]

    @classmethod
    def from_path(cls, conditions: tuple[Condition, ...]) -> "Rule":
        # Per (feature, op): keep MAX threshold for ">=" (most restrictive)
        #                    keep MIN threshold for "<"  (most restrictive)
        tightest: dict[tuple[str, str], float] = {}
        for c in conditions:
            key = (c.feature, c.op)
            cur = tightest.get(key)
            if cur is None:
                tightest[key] = c.threshold
            elif c.op == ">=":
                tightest[key] = max(cur, c.threshold)
            elif c.op == "<":
                tightest[key] = min(cur, c.threshold)
            else:
                raise ValueError(f"unsupported op {c.op!r}")
        reduced = tuple(
            Condition(feature=feat, op=op, threshold=thr)
            for (feat, op), thr in tightest.items()
        )
        canon = tuple(sorted(reduced, key=lambda c: (c.feature, c.op, c.threshold)))
        return cls(canon)

    def conditions_json(self) -> list[dict[str, Any]]:
        return [dataclasses.asdict(c) for c in self.conditions]


# ----- tree walking -----


def _round_threshold(value: float, sig_figs: int = 4) -> float:
    """Round to N significant figures so near-identical splits across
    trees dedupe to the same Condition. 4 sig figs is the same precision
    XGBoost uses internally for histogram bins."""
    if value == 0 or not math.isfinite(value):
        return value
    digits = sig_figs - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, digits)


def _walk(node: dict[str, Any], prefix: tuple[Condition, ...]) -> list[tuple[Condition, ...]]:
    """Recursive DFS — yield one tuple of conditions per leaf reached."""
    if "leaf" in node:
        return [prefix]
    feature = node["split"]
    threshold = _round_threshold(float(node["split_condition"]))
    children_by_id = {int(c["nodeid"]): c for c in node["children"]}
    yes_node = children_by_id[int(node["yes"])]
    no_node = children_by_id[int(node["no"])]
    paths: list[tuple[Condition, ...]] = []
    paths.extend(_walk(yes_node, prefix + (Condition(feature, "<", threshold),)))
    paths.extend(_walk(no_node, prefix + (Condition(feature, ">=", threshold),)))
    return paths


def extract_paths(booster: xgb.Booster) -> list[tuple[Condition, ...]]:
    """All root-to-leaf paths across every tree in ``booster``.

    Order is not meaningful; downstream dedup canonicalizes condition
    ordering per-path.
    """
    dumps = booster.get_dump(dump_format="json")
    all_paths: list[tuple[Condition, ...]] = []
    for tree_json in dumps:
        tree = json.loads(tree_json)
        all_paths.extend(_walk(tree, prefix=()))
    return all_paths


# ----- evaluation -----


def _build_condition_masks(
    features: pl.DataFrame, conditions: list[Condition]
) -> dict[Condition, np.ndarray]:
    """Precompute a boolean mask per unique condition.

    Caching here is the difference between "30s rule eval" and
    "30min rule eval" — many trees split on the same (feature,
    threshold) pair after rounding, and conditions are reused across
    rules.
    """
    by_feature: dict[str, list[Condition]] = defaultdict(list)
    for c in conditions:
        by_feature[c.feature].append(c)
    masks: dict[Condition, np.ndarray] = {}
    n = features.height
    for feat, conds in by_feature.items():
        col = features[feat]
        # Numeric features arrive as f64/f32/i64; cast bool-or-int to f64
        # so the comparison semantics are uniform.
        if col.dtype == pl.Boolean:
            arr = col.cast(pl.Int8).to_numpy().astype(np.float64, copy=False)
        else:
            arr = col.cast(pl.Float64, strict=False).to_numpy()
        finite_arr = arr.copy()
        nan_mask = ~np.isfinite(finite_arr)
        for cond in conds:
            if cond.op == "<":
                m = finite_arr < cond.threshold
            elif cond.op == ">=":
                m = finite_arr >= cond.threshold
            else:
                raise ValueError(f"unsupported op {cond.op!r}")
            # NaN never matches either branch — xgb's "missing" routing
            # is per-split, but for rule semantics here we treat null as
            # "condition not satisfied" on both sides. Conservative.
            m[nan_mask] = False
            masks[cond] = m
    return masks


def _evaluate_rules(
    rules: list[Rule],
    features: pl.DataFrame,
    is_winner: np.ndarray,
    examples_max: int = 10,
) -> list[dict[str, Any]]:
    """Per-rule: coverage, precision, lift, example rows."""
    # Gather all unique conditions referenced by any rule.
    all_conds: set[Condition] = set()
    for r in rules:
        all_conds.update(r.conditions)
    masks = _build_condition_masks(features, list(all_conds))

    n_rows = features.height
    n_winners = int(is_winner.sum())
    base_rate = n_winners / n_rows if n_rows else 0.0

    symbols = features["symbol"].to_numpy()
    dates = features["date"].to_numpy().astype("datetime64[D]").astype(str)

    out: list[dict[str, Any]] = []
    for rid, rule in enumerate(rules):
        rule_mask = np.ones(n_rows, dtype=bool)
        for cond in rule.conditions:
            rule_mask &= masks[cond]
        coverage_n = int(rule_mask.sum())
        if coverage_n == 0:
            continue
        positives = int(np.logical_and(rule_mask, is_winner).sum())
        precision = positives / coverage_n
        lift = precision / base_rate if base_rate > 0 else float("nan")

        # First N matching rows as examples (deterministic — order matches
        # the features frame's sort by (symbol, date)).
        idx = np.flatnonzero(rule_mask)[:examples_max]
        examples = [
            {"symbol": str(symbols[i]), "date": dates[i]} for i in idx
        ]

        out.append(
            {
                "rule_id": rid,
                "conditions_json": json.dumps(rule.conditions_json()),
                "n_conditions": len(rule.conditions),
                "coverage_n": coverage_n,
                "coverage_pct": round(100.0 * coverage_n / n_rows, 6),
                "precision": round(precision, 6),
                "lift": round(lift, 6),
                "example_symbol_dates_json": json.dumps(examples),
            }
        )
    return out


# ----- entrypoint -----


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _read_source_manifest(model_path: Path) -> dict[str, Any]:
    """Pick up the source run's manifest so we can carry split dates +
    universe size into the rules-run manifest."""
    candidate = model_path.parent / "manifest.json"
    if candidate.exists():
        return json.loads(candidate.read_text())
    return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase A Track 1 — XGB rule extraction"
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("runs/2026-05-12/model.json"),
        help="Path to the saved xgboost Booster json (default: %(default)s)",
    )
    p.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/features.parquet"),
        help="Path to the labeled features parquet (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the run output dir. Default: runs/<today>-step3a_xgb_rule_extraction/",
    )
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2024, 12, 31))
    p.add_argument(
        "--min-lift", type=float, default=1.5,
        help="Drop rules with lift below this on the holdout (default 1.5)",
    )
    p.add_argument(
        "--min-coverage-pct", type=float, default=0.5,
        help="Drop rules covering <X%% of the holdout (default 0.5)",
    )
    p.add_argument(
        "--min-precision", type=float, default=0.35,
        help="Drop rules with precision below this on the holdout (default 0.35)",
    )
    p.add_argument(
        "--max-rules", type=int, default=0,
        help="If >0, keep only the top N rules by (lift × coverage). 0 = no cap.",
    )
    p.add_argument(
        "--synthesis-top-n", type=int, default=200,
        help="Top-N rules the synthesis stage foregrounds (full set still in "
        "rules.parquet). Recorded in manifest.synthesis_top_n_by_lift_coverage. "
        "Default 200 per PR #1 issuecomment-4436499617.",
    )
    p.add_argument(
        "--resume", default=None,
        help="No-op for this CPU track. Accepted for ops-shortcut compatibility.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    pipeline_step = "step3a_xgb_rule_extraction"
    run_date_str = date.today().isoformat()
    if args.out_dir is not None:
        run_dir = args.out_dir if args.out_dir.is_absolute() else (_REPO_ROOT / args.out_dir)
        run_date_str = run_dir.name[:10]
    else:
        run_dir = _REPO_ROOT / "runs" / f"{run_date_str}-{pipeline_step}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(
        dir=run_dir,
        run_id=make_run_id(run_date_str, pipeline_step),
        pipeline_step=pipeline_step,
        epoch_total=1,  # rule extraction is a one-pass job
    )
    stop_flag = {"stop": False}
    install_graceful_interrupt(lambda: stop_flag.__setitem__("stop", True))
    status.update(state="training", epoch_current=0)

    try:
        print(f"track 1 (xgb rule extraction) — run dir: {run_dir.relative_to(_REPO_ROOT)}")
        model_path = (
            args.model if args.model.is_absolute() else (_REPO_ROOT / args.model)
        )
        features_path = (
            args.features
            if args.features.is_absolute()
            else (_REPO_ROOT / args.features)
        )

        if not model_path.exists():
            raise FileNotFoundError(f"XGB model not found at {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"features parquet not found at {features_path}")

        print(f"  loading model: {model_path.relative_to(_REPO_ROOT)}")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        source_manifest = _read_source_manifest(model_path)

        # Replay the feature-selection logic so we can map XGB's internal
        # f0..fN back to the original feature names. XGB was trained on
        # numpy arrays → it has no recorded feature_names of its own.
        labeled = pl.read_parquet(features_path).filter(
            pl.col("is_winner").is_not_null()
        )
        labeled, feature_cols = _replay_feature_selection(labeled)
        # Sanity check against the booster's tree dump.
        n_features = booster.num_features()
        if n_features != len(feature_cols):
            raise RuntimeError(
                f"booster expects {n_features} features but replay-derived "
                f"{len(feature_cols)} from the parquet — feature-selection "
                f"logic in scripts/discover.py and this module are out of sync."
            )
        # Stamp the names onto the booster so get_dump emits the real
        # feature strings instead of f0..fN.
        booster.feature_names = feature_cols

        print(f"  walking trees: {booster.num_boosted_rounds()} ...")
        all_paths = extract_paths(booster)
        print(f"    {len(all_paths):,} root-to-leaf paths")
        if stop_flag["stop"]:
            raise KeyboardInterrupt

        # Dedup paths → unique rules (canonical condition ordering).
        rule_set: dict[Rule, int] = {}
        for path in all_paths:
            r = Rule.from_path(path)
            rule_set.setdefault(r, len(rule_set))
        rules = list(rule_set.keys())
        print(f"  unique rules after dedup: {len(rules):,}")

        # Match the splits used by the source XGB run: holdout = date > val_end.
        holdout = labeled.filter(pl.col("date") > args.val_end)

        print(
            f"  holdout: {holdout.height:,} rows  "
            f"({holdout['date'].min()}→{holdout['date'].max()})"
        )
        is_winner = holdout["is_winner"].cast(pl.Boolean).to_numpy()

        print("  evaluating rules ...")
        eval_t0 = time.perf_counter()
        rule_records = _evaluate_rules(rules, holdout, is_winner)
        print(
            f"    evaluated {len(rule_records):,} rules with non-zero coverage "
            f"in {time.perf_counter() - eval_t0:.1f}s"
        )

        # Filter.
        kept = [
            r for r in rule_records
            if r["lift"] >= args.min_lift
            and r["coverage_pct"] >= args.min_coverage_pct
            and r["precision"] >= args.min_precision
        ]
        # Sort by lift × coverage (the brief's ranking).
        kept.sort(key=lambda r: r["lift"] * r["coverage_pct"], reverse=True)
        if args.max_rules > 0:
            kept = kept[: args.max_rules]
        # Reassign rule_id post-sort so the parquet's rule_id is rank order.
        for new_id, r in enumerate(kept):
            r["rule_id"] = new_id
        print(
            f"  rules kept after filter (lift≥{args.min_lift}, coverage≥{args.min_coverage_pct}%, "
            f"precision≥{args.min_precision}): {len(kept):,}"
        )
        if kept:
            top = kept[0]
            print(
                f"  top rule:  lift={top['lift']:.2f}  "
                f"precision={top['precision']*100:.2f}%  "
                f"coverage={top['coverage_pct']:.2f}%  "
                f"({top['n_conditions']} conditions)"
            )

        # Write rules.parquet.
        rules_df = pl.DataFrame(
            kept,
            schema={
                "rule_id": pl.Int64,
                "conditions_json": pl.Utf8,
                "n_conditions": pl.Int64,
                "coverage_n": pl.Int64,
                "coverage_pct": pl.Float64,
                "precision": pl.Float64,
                "lift": pl.Float64,
                "example_symbol_dates_json": pl.Utf8,
            },
        )
        rules_path = run_dir / "rules.parquet"
        rules_df.write_parquet(rules_path)
        print(f"  wrote {rules_path.relative_to(_REPO_ROOT)}")

        # Manifest.
        wall_clock_s = round(time.perf_counter() - t0, 3)
        manifest = {
            "run_id": make_run_id(run_date_str, pipeline_step),
            "pipeline_step": pipeline_step,
            "source_model_run_id": source_manifest.get("run_id"),
            "source_model_sha": f"sha256:{model_sha}",
            "train_end": source_manifest.get("train_end"),
            "val_end": args.val_end.isoformat(),
            "holdout_end": str(holdout["date"].max()),
            "feature_count": source_manifest.get("feature_count"),
            "universe_size": source_manifest.get("universe_size"),
            "holdout_n_rows": int(holdout.height),
            "holdout_base_rate": round(float(is_winner.sum() / holdout.height), 6),
            "n_paths_walked": len(all_paths),
            "n_rules_unique": len(rules),
            "n_rules_kept": len(kept),
            "filter_min_lift": args.min_lift,
            "filter_min_coverage_pct": args.min_coverage_pct,
            "filter_min_precision": args.min_precision,
            "synthesis_top_n_by_lift_coverage": int(args.synthesis_top_n),
            "runtime_device": "cpu",
            "train_wall_clock_s": wall_clock_s,
            "git_commit_of_quant_repo": _git_head_sha(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"  wrote {(run_dir / 'manifest.json').relative_to(_REPO_ROOT)}")
        print()
        print(
            f"=== TRACK 1 RESULT: {len(kept)} rules kept "
            f"(out of {len(rules)} unique) — {wall_clock_s:.1f}s ==="
        )
        status.record_checkpoint(epoch=1)
        status.update(state="done", epoch_current=1)
        return 0
    except KeyboardInterrupt:
        status.update(state="paused", epoch_current=0)
        print("[track1] interrupted; partial state in status.json")
        return 130
    except Exception as exc:
        status.update(state="failed", epoch_current=0, error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
