"""Joint validation for breakout_seq_v1 — simulates paired ENTRY × EXIT
trades using the hard-cap-60d exit mechanic per server-team commission
(PR #1 issuecomment-4469607665).

For each (symbol, date) in the validation window where the trained CNN
fires (score >= threshold):
  - Enter at close[entry] (proxy for next-day open)
  - Exit at FIRST of: (a) close >= entry * 1.20, OR (b) day-60 close
  - Record realized_gain_pct + hold_trading_days

Then apply server-team gates:
  - mean realized hold >= 30 trading days
  - median realized gain >= +15% (relaxed from +20% due to hard-cap)
  - win rate >= 55%
  - EV per trade > 0

Sweeps the decision threshold τ (0.50, 0.60, ..., 0.85). The Pareto pick
is the highest τ where all 4 gates clear AND the n_fires is meaningful
(>= 50 trades over the 2-year walk-forward window).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant.tracks.breakout_seq_label import (
    SPEC_DEFAULT,
    BreakoutSeqSpec,
    compute_realized_returns_60td,
)
from quant.tracks.sustained_winner_walkforward import (
    TRADE_WINDOW_END,
    TRADE_WINDOW_START,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "breakout_seq_joint_validate_v1"

# Per server-team commissioned gates
GATE_MIN_EV = 0.0
GATE_MIN_WIN_RATE = 0.55
GATE_MIN_MEDIAN_GAIN_PCT = 15.0
GATE_MIN_MEAN_HOLD_TD = 30

DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MIN_FIRES_TO_REPORT = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to runs/{date}-breakout_seq_v1_g20/ from breakout_seq_train.",
    )
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument(
        "--thresholds", type=str, default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
    )
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    run_dir = _resolve(args.run_dir)
    val_pred_path = run_dir / "val_predictions.parquet"
    if not val_pred_path.exists():
        print(f"ERROR: {val_pred_path} not found")
        return 1
    features_path = _resolve(args.features)

    val_df = pl.read_parquet(val_pred_path).with_columns(
        pl.col("date").str.to_date().alias("date")
    )
    print(f"loaded {val_df.height:,} val predictions")

    # Compute realized returns using the hard-cap-60d exit mechanic
    print("computing realized returns (hard-cap-60d exit) ...")
    t = time.perf_counter()
    features = pl.read_parquet(features_path)
    rr = compute_realized_returns_60td(features, SPEC_DEFAULT).select(
        ["symbol", "date", "bsq_realized_gain_pct", "bsq_hold_trading_days",
         "bsq_exit_reason"]
    )
    print(f"  realized returns: {rr.height:,} rows ({time.perf_counter()-t:.1f}s)")

    # Join score predictions with realized returns
    joined = val_df.join(rr, on=["symbol", "date"], how="left").filter(
        pl.col("bsq_realized_gain_pct").is_not_null()
        & (pl.col("date") >= TRADE_WINDOW_START)
        & (pl.col("date") <= TRADE_WINDOW_END)
    )
    print(f"  joined + filtered to trade window: {joined.height:,} rows")
    if joined.height == 0:
        print("ERROR: no rows after join — check val_predictions covers trade window")
        return 1

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    n_unique_dates = int(joined.select(pl.col("date").n_unique()).item())

    print(f"\n  threshold | n_fires | win_rate | med_gain% | mean_gain% | mean_loss% | EV%    | mean_hold | gates")
    print(f"  ----------+---------+----------+-----------+------------+------------+--------+-----------+--------")
    rows = []
    for tau in thresholds:
        fires = joined.filter(pl.col("score") >= tau)
        n_fires = fires.height
        if n_fires == 0:
            print(f"  {tau:>9.2f} |       0 | (no fires)")
            rows.append({"threshold": tau, "n_fires": 0})
            continue
        gains = fires["bsq_realized_gain_pct"].to_numpy()
        holds = fires["bsq_hold_trading_days"].to_numpy()
        wins_mask = gains > 0
        n_wins = int(wins_mask.sum())
        win_rate = n_wins / n_fires
        mean_gain = float(gains[wins_mask].mean()) if n_wins else 0.0
        n_losses = n_fires - n_wins
        mean_loss = float(gains[~wins_mask].mean()) if n_losses else 0.0
        ev = win_rate * mean_gain + (1 - win_rate) * mean_loss
        median_gain = float(np.median(gains))
        mean_hold = float(holds.mean())

        passes_ev = ev > GATE_MIN_EV
        passes_win = win_rate >= GATE_MIN_WIN_RATE
        passes_median = median_gain >= GATE_MIN_MEDIAN_GAIN_PCT
        passes_hold = mean_hold >= GATE_MIN_MEAN_HOLD_TD
        passes_all = passes_ev and passes_win and passes_median and passes_hold
        gate_str = "PASS" if (passes_all and n_fires >= MIN_FIRES_TO_REPORT) else "fail"

        rows.append({
            "threshold": tau,
            "n_fires": n_fires,
            "n_unique_dates": int(fires.select(pl.col("date").n_unique()).item()),
            "fires_per_day": n_fires / max(1, n_unique_dates),
            "win_rate": win_rate,
            "median_gain_pct": median_gain,
            "mean_gain_pct": mean_gain,
            "mean_loss_pct": mean_loss,
            "mean_endpoint_pct": float(gains.mean()),
            "ev_per_trade_pct": ev,
            "mean_hold_trading_days": mean_hold,
            "passes_ev": passes_ev,
            "passes_win_rate": passes_win,
            "passes_median_gain": passes_median,
            "passes_mean_hold": passes_hold,
            "passes_all_gates": bool(passes_all and n_fires >= MIN_FIRES_TO_REPORT),
        })
        print(f"  {tau:>9.2f} | {n_fires:>7,} | {100*win_rate:>7.1f}% | {median_gain:>8.2f}% | {mean_gain:>9.2f}% | "
              f"{mean_loss:>9.2f}% | {ev:>5.2f}% | {mean_hold:>9.1f} | {gate_str}")

    pl.DataFrame(rows).write_parquet(run_dir / "joint_validation.parquet")

    # Pareto pick: highest τ where all gates pass
    passing = [r for r in rows if r.get("passes_all_gates")]
    pareto = max(passing, key=lambda r: r["threshold"]) if passing else None

    summary = {
        "pipeline_step": PIPELINE_STEP,
        "trade_window": [TRADE_WINDOW_START.isoformat(), TRADE_WINDOW_END.isoformat()],
        "gates": {
            "min_ev_pct": GATE_MIN_EV,
            "min_win_rate": GATE_MIN_WIN_RATE,
            "min_median_gain_pct": GATE_MIN_MEDIAN_GAIN_PCT,
            "min_mean_hold_trading_days": GATE_MIN_MEAN_HOLD_TD,
            "min_fires_to_report": MIN_FIRES_TO_REPORT,
        },
        "per_threshold": rows,
        "pareto_pick": (
            {"threshold": pareto["threshold"], "ev_per_trade_pct": pareto["ev_per_trade_pct"],
             "win_rate": pareto["win_rate"], "median_gain_pct": pareto["median_gain_pct"],
             "n_fires": pareto["n_fires"]}
            if pareto else None
        ),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (run_dir / "joint_validation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nsummary: {run_dir / 'joint_validation_summary.json'}")
    if pareto:
        print(f"\nPARETO PICK: threshold={pareto['threshold']}, "
              f"EV={pareto['ev_per_trade_pct']:.2f}%, "
              f"win={100*pareto['win_rate']:.1f}%, "
              f"med_gain={pareto['median_gain_pct']:.2f}%, "
              f"n_fires={pareto['n_fires']:,}")
    else:
        print("\nNO PARETO PICK: no threshold clears all 4 gates with >=50 fires")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
