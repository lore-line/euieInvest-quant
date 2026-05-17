"""Joint validation for sustained_winner_discovery_v1 — Workstream C step 5.

For each walk-forward-surviving rule in each per-g spec, simulate the
paired ENTRY × EXIT trade:

  ENTRY: rule fires on (symbol, date) in 2024-01-01 → 2026-03-30
  EXIT:  fixed +20 trading days later
  realized_gain = forward_endpoint_pct (already in the labeled frame)

Compute the 7 metrics from the server-team brief
(PR #1 issuecomment-4469094953 → 2026-05-17 03:22Z):

  1. median_realized_gain
  2. mean_realized_gain
  3. mean_realized_loss      (mean over fires where endpoint < 0)
  4. ev_per_trade            = win_rate × mean_gain + (1-win_rate) × mean_loss
  5. win_rate                (endpoint > 0)
  6. mean_hold_trading_days  (= 20 for this spec — invariant)
  7. signal_volume_per_day   (n_fires / n_unique_dates)
  +
  annualized_return_estimate = (1 + ev_per_trade)^(252/20) - 1

Per-spec Pareto-pick: the highest g where the SPEC-aggregate clears
all 4 gates:
  - ev_per_trade > 0
  - median_realized_gain >= g   (the full sustained-winner outcome)
  - mean_hold_trading_days >= 20 (auto-satisfied — invariant)
  - win_rate >= 0.50

Output per spec dir `runs/{date}-sustained_winner_v1_g{NN}/`:
  - `joint_validation.parquet` — one row per (rule_id) with all 7 metrics
  - `joint_validation_spec_summary.json` — aggregate gates + verdict

Sweep-level summary: `runs/{date}-sustained_winner_joint_validate_summary.json`
plus the picked g and the rules emit-list under that g.

Performance: ~10s per spec (sub-1 min total sweep).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant.tracks.sustained_winner_label import (
    SPECS,
    SustainedWinnerSpec,
    compute_sustained_winner_label,
    sweep_specs,
)
from quant.tracks.sustained_winner_walkforward import (
    SURVIVOR_MAX_LIFT_DECAY,
    SURVIVOR_MIN_LIFT,
    WINDOWS,
    _label_features_once,
    _rules_from_parquet,
)
from quant.tracks.xgb_rule_extraction import (
    Condition,
    Rule,
    _build_condition_masks,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "sustained_winner_joint_validate_v1"

# Joint-validation gates per server-team Phase B brief (re-applied here
# for the sustained_winner_discovery sweep).
GATE_MIN_EV = 0.0
GATE_MIN_WIN_RATE = 0.50
GATE_MIN_MEAN_HOLD_TD = 20  # auto-satisfied; invariant of the spec horizon
# Median realized gain gate is per-spec: must be >= spec.touch_threshold_pct
# (i.e. the FULL sustained-winner outcome — half the time, the rule fire
# must actually deliver the +g% endpoint).

TRADE_WINDOW_START = WINDOWS[0][0]  # 2024-01-01
TRADE_WINDOW_END = WINDOWS[-1][1]   # 2026-03-30


# -------------------- per-rule trade evaluation --------------------

def _evaluate_rules_for_trades(
    rules: list[Rule],
    labelable: pl.DataFrame,
) -> dict[int, dict]:
    """For each rule, return its trade-level metrics across the full
    walk-forward window.

    Returns dict keyed by rule_id (positional index in `rules`) with:
      n_fires, n_unique_dates, mean_endpoint, median_endpoint,
      mean_gain, mean_loss, win_rate, ev_per_trade

    Wins are endpoint_pct > 0; losses are endpoint_pct <= 0 (treats
    exactly-zero endpoints as losses, conservative).
    """
    if labelable.height == 0 or not rules:
        return {}

    # Pre-build condition masks ONCE for all unique conditions.
    all_conds: set[Condition] = set()
    for r in rules:
        all_conds.update(r.conditions)
    masks = _build_condition_masks(labelable, list(all_conds))

    endpoint = labelable["forward_endpoint_pct"].to_numpy()
    dates = labelable["date"].to_numpy()

    out: dict[int, dict] = {}
    for rid, rule in enumerate(rules):
        rule_mask = np.ones(labelable.height, dtype=bool)
        for cond in rule.conditions:
            rule_mask &= masks[cond]
        n_fires = int(rule_mask.sum())
        if n_fires == 0:
            out[rid] = {
                "n_fires": 0,
                "n_unique_dates": 0,
                "mean_endpoint": float("nan"),
                "median_endpoint": float("nan"),
                "mean_gain": float("nan"),
                "mean_loss": float("nan"),
                "win_rate": float("nan"),
                "ev_per_trade": float("nan"),
            }
            continue
        fired_endpoint = endpoint[rule_mask]
        fired_dates = dates[rule_mask]
        wins = fired_endpoint > 0
        n_wins = int(wins.sum())
        n_losses = n_fires - n_wins
        win_rate = n_wins / n_fires
        mean_gain = float(fired_endpoint[wins].mean()) if n_wins else 0.0
        mean_loss = float(fired_endpoint[~wins].mean()) if n_losses else 0.0
        # EV per trade: convex combination weighted by win rate
        ev = win_rate * mean_gain + (1.0 - win_rate) * mean_loss
        out[rid] = {
            "n_fires": n_fires,
            "n_unique_dates": int(len(np.unique(fired_dates))),
            "mean_endpoint": float(fired_endpoint.mean()),
            "median_endpoint": float(np.median(fired_endpoint)),
            "mean_gain": mean_gain,
            "mean_loss": mean_loss,
            "win_rate": float(win_rate),
            "ev_per_trade": float(ev),
        }
    return out


def _joint_validate_one_spec(
    labeled_with_returns: pl.DataFrame,
    spec: SustainedWinnerSpec,
    spec_dir: Path,
) -> dict:
    """Run joint validation for ONE spec: load surviving rules from
    walk_forward_aggregate.parquet, score trade economics across the
    full walk-forward window, evaluate spec-aggregate gates.
    """
    rules_path = spec_dir / "rules.parquet"
    wf_agg_path = spec_dir / "walk_forward_aggregate.parquet"
    if not rules_path.exists() or not wf_agg_path.exists():
        return {"spec": spec.name, "skipped": True, "reason": "missing rules.parquet or walkforward aggregate"}

    rules, _train_lifts = _rules_from_parquet(rules_path)
    wf_agg = pl.read_parquet(wf_agg_path)
    survivor_ids = set(
        wf_agg.filter(pl.col("is_walk_forward_survivor")).get_column("rule_id").to_list()
    )
    if not survivor_ids:
        return {"spec": spec.name, "skipped": True, "reason": "no walk-forward survivors"}

    # Derive the spec's label only for filtering rows that are LABELABLE
    # (i.e. have a 20-td forward window); we use forward_endpoint_pct
    # directly for the trade calc — not the boolean label.
    labelable = labeled_with_returns.filter(
        (pl.col("date") >= TRADE_WINDOW_START)
        & (pl.col("date") <= TRADE_WINDOW_END)
        & pl.col("close_adj").is_not_null()
        & pl.col("close_adj").ge(spec.min_entry_price_usd)
        & pl.col("forward_endpoint_pct").is_not_null()
        & pl.col("forward_max_pct").is_not_null()
    )
    n_labelable = labelable.height
    n_unique_dates_universe = int(labelable.select(pl.col("date").n_unique()).item())

    # Sub-select to surviving rules only (huge perf win — typically 25-85%
    # of rules survive, but masking over a smaller rule-set is cheaper).
    surviving_rules = [rules[i] for i in sorted(survivor_ids)]
    surviving_ids_ordered = sorted(survivor_ids)

    t = time.perf_counter()
    per_rule = _evaluate_rules_for_trades(surviving_rules, labelable)
    eval_s = time.perf_counter() - t

    # Build per-rule output rows. Map positional idx → original rule_id.
    rows = []
    for pos_idx, original_rid in enumerate(surviving_ids_ordered):
        m = per_rule.get(pos_idx, {})
        if m.get("n_fires", 0) == 0:
            continue
        signal_volume_per_day = m["n_fires"] / max(1, m["n_unique_dates"])
        rows.append({
            "rule_id": int(original_rid),
            "n_fires": m["n_fires"],
            "n_unique_dates": m["n_unique_dates"],
            "signal_volume_per_day": signal_volume_per_day,
            "mean_endpoint_pct": m["mean_endpoint"],
            "median_endpoint_pct": m["median_endpoint"],
            "mean_gain_pct": m["mean_gain"],
            "mean_loss_pct": m["mean_loss"],
            "win_rate": m["win_rate"],
            "ev_per_trade_pct": m["ev_per_trade"],
            "mean_hold_trading_days": 20,
            "annualized_return_est": float(
                (1.0 + m["ev_per_trade"] / 100.0) ** (252.0 / 20.0) - 1.0
            ),
        })
    rule_df = pl.DataFrame(rows)
    jv_path = spec_dir / "joint_validation.parquet"
    rule_df.write_parquet(jv_path)

    # Per-rule gate evaluation
    if rule_df.height > 0:
        rule_df = rule_df.with_columns(
            passes_ev=pl.col("ev_per_trade_pct") > GATE_MIN_EV,
            passes_median_gain=pl.col("median_endpoint_pct") >= spec.touch_threshold_pct,
            passes_win_rate=pl.col("win_rate") >= GATE_MIN_WIN_RATE,
            passes_mean_hold=pl.col("mean_hold_trading_days") >= GATE_MIN_MEAN_HOLD_TD,
        ).with_columns(
            passes_all=pl.col("passes_ev")
            & pl.col("passes_median_gain")
            & pl.col("passes_win_rate")
            & pl.col("passes_mean_hold"),
        )
        n_pass_ev = int(rule_df.filter(pl.col("passes_ev")).height)
        n_pass_median = int(rule_df.filter(pl.col("passes_median_gain")).height)
        n_pass_wr = int(rule_df.filter(pl.col("passes_win_rate")).height)
        n_pass_all = int(rule_df.filter(pl.col("passes_all")).height)
    else:
        n_pass_ev = n_pass_median = n_pass_wr = n_pass_all = 0

    # Spec-aggregate metrics: weight each rule by its n_fires
    if rule_df.height > 0 and rule_df["n_fires"].sum() > 0:
        total_fires = int(rule_df["n_fires"].sum())
        # EV-per-trade aggregate across all fires
        # (weighted mean of per-rule EV by n_fires)
        weighted_ev = float(
            (rule_df["ev_per_trade_pct"] * rule_df["n_fires"]).sum() / total_fires
        )
        weighted_win_rate = float(
            (rule_df["win_rate"] * rule_df["n_fires"]).sum() / total_fires
        )
        weighted_median_gain = float(rule_df["median_endpoint_pct"].median())
    else:
        total_fires = 0
        weighted_ev = weighted_win_rate = weighted_median_gain = float("nan")

    spec_passes_gates = (
        weighted_ev > GATE_MIN_EV
        and weighted_median_gain >= spec.touch_threshold_pct
        and weighted_win_rate >= GATE_MIN_WIN_RATE
    )

    summary = {
        "spec": spec.name,
        "touch_threshold_pct": spec.touch_threshold_pct,
        "endpoint_threshold_pct": spec.endpoint_threshold_pct,
        "n_labelable_rows": int(n_labelable),
        "n_unique_dates_universe": int(n_unique_dates_universe),
        "n_surviving_rules_input": int(len(surviving_rules)),
        "n_rules_with_fires": int(rule_df.height),
        "n_rules_pass_ev_gate": n_pass_ev,
        "n_rules_pass_median_gain_gate": n_pass_median,
        "n_rules_pass_win_rate_gate": n_pass_wr,
        "n_rules_pass_all_gates": n_pass_all,
        "total_fires": total_fires,
        "spec_aggregate_ev_pct": weighted_ev,
        "spec_aggregate_win_rate": weighted_win_rate,
        "spec_aggregate_median_gain_pct": weighted_median_gain,
        "spec_passes_aggregate_gates": spec_passes_gates,
        "joint_validation_parquet": str(jv_path),
        "eval_wall_clock_s": round(eval_s, 1),
    }
    with open(spec_dir / "joint_validation_spec_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# -------------------- CLI --------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--date-prefix", type=str, default="2026-05-17")
    p.add_argument("--specs", type=str, default=None,
                   help="Comma-separated subset of spec names (e.g. 'g20,g15'). Default: all 20.")
    p.add_argument("--horizon-days", type=int, default=20)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features_path = _resolve(args.features)
    runs_dir = _resolve(args.runs_dir)

    spec_dirs = sorted(
        runs_dir.glob(f"{args.date_prefix}-sustained_winner_v1_g*"),
        key=lambda d: int(d.name.split("_g")[-1]),
        reverse=True,
    )
    if args.specs:
        wanted = {s.strip() for s in args.specs.split(",")}
        spec_dirs = [d for d in spec_dirs if d.name.split("_v1_")[-1] in wanted]
    if not spec_dirs:
        print(f"no spec dirs matching {args.date_prefix}-sustained_winner_v1_g* in {runs_dir}")
        return 1

    print(f"joint_validate v1 — {len(spec_dirs)} specs queued")
    print(f"features:  {features_path}")
    print(f"horizon:   {args.horizon_days}td")
    print(f"window:    {TRADE_WINDOW_START} → {TRADE_WINDOW_END}")
    print(f"gates:     EV>0  win_rate>={GATE_MIN_WIN_RATE}  median_gain>=g  mean_hold>={GATE_MIN_MEAN_HOLD_TD}td")
    print()

    t0 = time.perf_counter()
    features = pl.read_parquet(features_path)
    print(f"loaded {features.height:,} rows × {len(features.columns)} cols in {time.perf_counter()-t0:.1f}s")
    t = time.perf_counter()
    labeled_with_returns = _label_features_once(features, args.horizon_days)
    print(f"computed forward returns + applied dummies in {time.perf_counter()-t:.1f}s")
    print()

    sweep_lookup = {s.name: s for s in sweep_specs()} | {s.name: s for s in SPECS.values()}

    print(f"{'g':>4}  {'rules in':>9}  {'with fires':>10}  {'agg EV%':>8}  {'agg win%':>8}  {'med gain%':>9}  {'n pass all':>11}  {'gate':>5}")
    print(f"{'-'*4}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*5}")
    summary_rows: list[dict] = []
    for spec_dir in spec_dirs:
        spec_name = spec_dir.name.split("_v1_")[-1]
        spec = sweep_lookup.get(spec_name)
        if spec is None:
            print(f"  skip {spec_dir.name}: unknown spec '{spec_name}'")
            continue
        s = _joint_validate_one_spec(labeled_with_returns, spec, spec_dir)
        summary_rows.append(s)
        if s.get("skipped"):
            print(f"  {spec.name}: SKIPPED — {s.get('reason')}")
            continue
        gate = "PASS" if s["spec_passes_aggregate_gates"] else "fail"
        print(
            f"{spec.name:>4}  {s['n_surviving_rules_input']:>9,}  "
            f"{s['n_rules_with_fires']:>10,}  "
            f"{s['spec_aggregate_ev_pct']:>8.2f}  "
            f"{100*s['spec_aggregate_win_rate']:>7.1f}%  "
            f"{s['spec_aggregate_median_gain_pct']:>8.2f}%  "
            f"{s['n_rules_pass_all_gates']:>11,}  "
            f"{gate:>5}"
        )

    # Pareto pick: highest g where spec_passes_aggregate_gates AND
    # n_rules_pass_all_gates > 0
    passing = [
        s for s in summary_rows
        if not s.get("skipped")
        and s.get("spec_passes_aggregate_gates")
        and s.get("n_rules_pass_all_gates", 0) > 0
    ]
    pareto_pick = None
    if passing:
        # Highest g (touch_threshold_pct) is the most demanding / highest-EV
        pareto_pick = max(passing, key=lambda s: s["touch_threshold_pct"])

    summary_path = runs_dir / f"{args.date_prefix}-sustained_winner_joint_validate_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "pipeline_step": PIPELINE_STEP,
            "trade_window": [TRADE_WINDOW_START.isoformat(), TRADE_WINDOW_END.isoformat()],
            "gates": {
                "min_ev_pct": GATE_MIN_EV,
                "min_win_rate": GATE_MIN_WIN_RATE,
                "min_mean_hold_trading_days": GATE_MIN_MEAN_HOLD_TD,
                "median_gain_threshold": "per-spec touch_threshold_pct",
            },
            "specs": summary_rows,
            "pareto_pick": {
                "spec_name": pareto_pick["spec"] if pareto_pick else None,
                "touch_threshold_pct": pareto_pick["touch_threshold_pct"] if pareto_pick else None,
                "n_rules_pass_all_gates": pareto_pick["n_rules_pass_all_gates"] if pareto_pick else 0,
                "spec_aggregate_ev_pct": pareto_pick["spec_aggregate_ev_pct"] if pareto_pick else None,
            },
            "total_wall_clock_s": round(time.perf_counter() - t0, 1),
        }, f, indent=2)

    print()
    print(f"summary: {summary_path}")
    if pareto_pick:
        print(f"\nPARETO PICK: spec={pareto_pick['spec']}  "
              f"g={pareto_pick['touch_threshold_pct']}%  "
              f"agg_EV={pareto_pick['spec_aggregate_ev_pct']:.2f}%  "
              f"{pareto_pick['n_rules_pass_all_gates']:,} rules pass all gates")
    else:
        print("\nNO PARETO PICK: no spec clears all aggregate gates "
              "(EV>0 AND median_gain>=g AND win_rate>=0.50)")
    print(f"total wall clock: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
