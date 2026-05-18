"""Equity momentum walkforward validator — Stream 2a/2b v1.

Per PR #1 issuecomment-4473364135 + 4473464239. Walkforward template
for the equity momentum cohort defined in `equity_momentum_label.py`.

Different from Phase B v3 walkforward in two ways:
  1. **No rule extraction** — the "rule" IS the Donchian breakout filter
     (single deterministic signal, not a tree-extracted conjunction).
     Each entry signal becomes one trade.
  2. **Realized return at fixed horizon** — exit at close[entry+H]
     where H = spec.horizon_trading_days. No early stops/targets at
     this validation layer (those live in the platform's adaptive-exit
     engine; my role here is signal-cohort validation, not execution).

Pipeline:
  1. Load features.parquet
  2. Apply compute_equity_momentum_label for the spec
  3. Filter to walkforward window (2024-01-01 → 2026-03-30 default)
  4. For each entry signal: compute realized exit-at-horizon return
  5. Split chronologically into 5 windows; per-window stats
  6. Apply friction (flat profile, mid-cap-class) + survivorship haircut
  7. Write artifacts to runs/{date}-equity_momentum_{spec}_walkforward/

Outputs:
  - signals.parquet                — per-trade (symbol, entry_date,
                                     entry_price, exit_date, exit_price,
                                     gross_pnl_pct, realized_horizon_days)
  - universe_filtered.parquet      — canonical R1000-equivalent list per
                                     server-team handoff (PR #1
                                     issuecomment-4473473735)
  - walk_forward.parquet           — per-window aggregate
  - friction_breakdown.json        — flat profile, 15% liquid-haircut
  - manifest.json                  — config + summary
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant.tracks.equity_momentum_label import (
    SPECS,
    EquityMomentumSpec,
    compute_equity_momentum_label,
    label_statistics,
)
from quant.tracks.phase_b_v3_friction_extension import _equity_curve_sharpe
from quant.tracks.sustained_winner_walkforward import WINDOWS

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "equity_momentum_walkforward_v1"

# Liquid mid-cap-class flat-friction profile (server-team doctrine §X)
LIQUID_FEE_PCT_PER_LEG = 0.0      # 0% commission retail US equities
LIQUID_SPREAD_PCT_RT = 0.0005     # 0.05% bid-ask round-trip (liquid mid-cap typical)
LIQUID_SLIPPAGE_PCT_RT = 0.0010   # 0.10% round-trip slippage (liquid mid-cap typical)
LIQUID_TOTAL_FRICTION_RT = (
    2 * LIQUID_FEE_PCT_PER_LEG + LIQUID_SPREAD_PCT_RT + LIQUID_SLIPPAGE_PCT_RT
)  # = 0.15% RT total

# Survivorship-bias haircut per server-team table (mid-cap-or-larger liquid)
LIQUID_SURVIVORSHIP_HAIRCUT = 0.15

# Stream 2 paper-gate threshold
GATE_MIN_NET_SHARPE = 0.8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument(
        "--spec", type=str, default="fast", choices=["fast", "slow"],
        help="Which equity momentum variant to validate: 'fast' (60d/60d/+15%) "
             "or 'slow' (252d/180d/+25%).",
    )
    p.add_argument(
        "--vol-confirm-mult", type=float, default=0.0,
        help="Volume-confirmation multiplier: entry volume must be ≥ N × trailing-30d-avg. "
             "0.0 = disabled (default). 1.5 = filter per server-team finding "
             "(PR #1 issuecomment-4473629237).",
    )
    p.add_argument(
        "--position-size-usd", type=float, default=1000.0,
        help="Per-trade position size for paper-sim equity curve. Default $1000.",
    )
    p.add_argument(
        "--sleeve-usd", type=float, default=10_000.0,
        help="Starting sleeve for equity-curve Sharpe calc. Default $10K matches "
             "paper_sleeve_simulate convention.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _build_per_trade_table(
    labeled: pl.DataFrame, spec: EquityMomentumSpec,
) -> pl.DataFrame:
    """Project labeled features to one row per (symbol, entry_date) signal.

    Computes the realized exit-at-horizon return for each entry.
    """
    # Filter to entry signals only (where the breakout-+-liquidity condition
    # is True at this row's date)
    entry_signals = labeled.filter(
        (pl.col("close_adj") >= spec.min_entry_price_usd)
        & (pl.col("avg_dollar_volume_30d") >= spec.min_avg_volume_30d_dollar)
        & (pl.col("close_adj") > pl.col("prior_n_day_high"))
        & pl.col(spec.label_column()).is_not_null()
    ).sort(["symbol", "date"])

    # Exit-at-horizon close + date are already computed in label module
    # (must be done there, not here — shift(-H) on filtered sparse series
    # would skip calendar days).
    H = spec.horizon_trading_days
    entry_signals = entry_signals.with_columns(
        exit_close_adj=pl.col(f"exit_close_adj_{H}td"),
        exit_date=pl.col(f"exit_date_{H}td"),
    ).filter(
        pl.col("exit_close_adj").is_not_null()
        & pl.col("exit_date").is_not_null()
    )

    entry_signals = entry_signals.with_columns(
        gross_pnl_pct=(pl.col("exit_close_adj") / pl.col("close_adj") - 1.0) * 100.0,
    )

    # Project to clean per-trade schema
    return entry_signals.select([
        pl.col("symbol"),
        pl.col("date").alias("entry_date"),
        pl.col("close_adj").alias("entry_price"),
        pl.col("exit_date"),
        pl.col("exit_close_adj").alias("exit_price"),
        pl.col("gross_pnl_pct"),
        pl.col(spec.forward_max_column()).alias("forward_max_pct"),
        pl.col(spec.label_column()).alias("is_winner_label"),
        pl.col("avg_dollar_volume_30d"),
        pl.col("atr_pct_14").alias("entry_atr_pct_14"),
    ]).with_columns(
        realized_horizon_days=pl.lit(H),
    )


def _per_window_stats(
    trades: pl.DataFrame, sleeve_usd: float, position_size_usd: float,
) -> pl.DataFrame:
    """Per chronological window: signal count, winner rate, mean/median return, sharpe."""
    rows = []
    for w_idx, (w_start, w_end) in enumerate(WINDOWS):
        bucket = trades.filter(
            (pl.col("entry_date") >= w_start) & (pl.col("entry_date") <= w_end)
        )
        if bucket.height == 0:
            continue
        gains = bucket["gross_pnl_pct"].to_numpy()
        # Build per-trade USD pnl + equity curve for this window
        per_trade_usd = (gains / 100.0) * position_size_usd
        sharpe, max_dd = _equity_curve_sharpe(
            bucket.sort("exit_date").select(
                (pl.col("gross_pnl_pct") / 100.0 * position_size_usd).alias("usd")
            )["usd"].to_numpy(),
            sleeve_usd,
        )
        rows.append({
            "window_idx": w_idx,
            "window_start": w_start,
            "window_end": w_end,
            "n_trades": int(bucket.height),
            "win_rate": float((gains > 0).mean()),
            "mean_gross_pnl_pct": float(gains.mean()),
            "median_gross_pnl_pct": float(np.median(gains)),
            "total_gross_pnl_usd": float(per_trade_usd.sum()),
            "gross_sharpe": float(sharpe),
            "max_drawdown_pct": float(max_dd),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "window_idx": pl.Int64, "window_start": pl.Date, "window_end": pl.Date,
            "n_trades": pl.Int64, "win_rate": pl.Float64,
            "mean_gross_pnl_pct": pl.Float64, "median_gross_pnl_pct": pl.Float64,
            "total_gross_pnl_usd": pl.Float64, "gross_sharpe": pl.Float64,
            "max_drawdown_pct": pl.Float64,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    base_spec = SPECS[args.spec]
    # If vol-confirm is requested, build a new spec with that mult, modified
    # name suffix so out_dir + label column don't collide with the baseline run.
    if args.vol_confirm_mult > 0.0:
        from dataclasses import replace
        spec = replace(
            base_spec,
            name=f"{base_spec.name}_vc{int(args.vol_confirm_mult * 10)}",
            vol_confirm_mult=args.vol_confirm_mult,
        )
    else:
        spec = base_spec

    features_path = _resolve(args.features)
    today = date.today().isoformat()
    out_dir = (
        _resolve(args.out_dir)
        if args.out_dir is not None
        else _REPO_ROOT / "runs" / f"{today}-equity_momentum_{spec.name}_walkforward"
    )
    if not out_dir.parent.exists():
        alt = Path("/workspace/runs")
        if alt.exists():
            out_dir = alt / f"{today}-equity_momentum_{spec.name}_walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{PIPELINE_STEP} — spec {spec.name}")
    print(f"  features:    {features_path}")
    print(f"  out_dir:     {out_dir}")
    print(f"  spec:        breakout_days={spec.entry_breakout_days} horizon={spec.horizon_trading_days}td target=+{100*spec.target_pct:.0f}%")
    print(f"  liquidity:   price >= ${spec.min_entry_price_usd:.0f}, ADV $ >= ${spec.min_avg_volume_30d_dollar/1e6:.0f}M")
    print()

    # Load + label
    t = time.perf_counter()
    features = pl.read_parquet(features_path)
    print(f"  loaded {features.height:,} rows × {features.width} cols ({time.perf_counter()-t:.1f}s)")
    t = time.perf_counter()
    labeled = compute_equity_momentum_label(features, spec)
    print(f"  labeled in {time.perf_counter()-t:.1f}s")
    stats = label_statistics(labeled, spec)
    print(f"  cohort stats: {stats['n_entry_signals_universe']:,} entry signals, "
          f"{stats['n_labelable_rows']:,} labelable, "
          f"{stats['n_winners']:,} winners ({100*stats['winner_rate']:.1f}%)")

    # Per-trade table (every entry signal becomes one trade)
    trades = _build_per_trade_table(labeled, spec)
    print(f"  per-trade table: {trades.height:,} trades (entries with complete forward window)")

    # Filter to walkforward window (2024-01-01 → 2026-03-30)
    in_window = trades.filter(
        (pl.col("entry_date") >= WINDOWS[0][0])
        & (pl.col("entry_date") <= WINDOWS[-1][1])
    )
    print(f"  in walkforward window {WINDOWS[0][0]}..{WINDOWS[-1][1]}: {in_window.height:,} trades")

    # Write signals.parquet
    in_window.write_parquet(out_dir / "signals.parquet")
    print(f"  wrote signals.parquet")

    # Universe filter snapshot — canonical R1000-equivalent list for server-team consumption
    universe = (
        labeled.filter(
            (pl.col("close_adj") >= spec.min_entry_price_usd)
            & (pl.col("avg_dollar_volume_30d") >= spec.min_avg_volume_30d_dollar)
        )
        .group_by("symbol")
        .agg([
            pl.col("date").min().alias("first_seen_date"),
            pl.col("date").max().alias("last_seen_date"),
            pl.col("avg_dollar_volume_30d").mean().alias("mean_avg_dollar_volume_30d"),
            pl.col("close_adj").mean().alias("mean_close_adj"),
            pl.col("date").count().alias("n_qualifying_days"),
        ])
        .with_columns(passes_filter=pl.lit(True))
        .sort("symbol")
    )
    universe.write_parquet(out_dir / "universe_filtered.parquet")
    print(f"  wrote universe_filtered.parquet ({universe.height:,} symbols)")

    # Per-window stats
    win_stats = _per_window_stats(in_window, args.sleeve_usd, args.position_size_usd)
    win_stats.write_parquet(out_dir / "walk_forward.parquet")
    print(f"  wrote walk_forward.parquet ({win_stats.height} windows)")
    if win_stats.height:
        print(f"  per-window:")
        for r in win_stats.iter_rows(named=True):
            print(f"    {r['window_start']}..{r['window_end']}: "
                  f"n={r['n_trades']:>4,}  win={100*r['win_rate']:>4.1f}%  "
                  f"mean_pnl={r['mean_gross_pnl_pct']:>+5.1f}%  sharpe={r['gross_sharpe']:>+5.2f}")

    # Aggregate friction (flat liquid mid-cap profile) + 15% survivorship haircut
    gross_pnl_pct_arr = in_window["gross_pnl_pct"].to_numpy()
    fee_cost_pct = 2 * LIQUID_FEE_PCT_PER_LEG * 100
    spread_cost_pct = LIQUID_SPREAD_PCT_RT * 100
    slippage_cost_pct = LIQUID_SLIPPAGE_PCT_RT * 100
    total_friction_pct = fee_cost_pct + spread_cost_pct + slippage_cost_pct  # 0.15% RT

    net_pnl_pct_arr = gross_pnl_pct_arr - total_friction_pct

    # Equity-curve sharpe on chronological per-trade pnl
    in_window_sorted = in_window.sort("exit_date")
    gross_per_trade_usd = in_window_sorted["gross_pnl_pct"].to_numpy() / 100.0 * args.position_size_usd
    net_per_trade_usd = (in_window_sorted["gross_pnl_pct"].to_numpy() - total_friction_pct) / 100.0 * args.position_size_usd

    gross_sharpe, gross_max_dd = _equity_curve_sharpe(gross_per_trade_usd, args.sleeve_usd)
    net_sharpe, net_max_dd = _equity_curve_sharpe(net_per_trade_usd, args.sleeve_usd)

    # Survivorship haircut at Sharpe level per server-team convention
    haircut_sharpe = net_sharpe * (1.0 - LIQUID_SURVIVORSHIP_HAIRCUT)

    gross_total_usd = float(gross_per_trade_usd.sum())
    net_total_usd = float(net_per_trade_usd.sum())
    haircut_total_usd = net_total_usd * (1.0 - LIQUID_SURVIVORSHIP_HAIRCUT)

    friction_summary = {
        "pipeline_step": PIPELINE_STEP,
        "spec": {"name": spec.name, "entry_breakout_days": spec.entry_breakout_days,
                 "horizon_trading_days": spec.horizon_trading_days,
                 "target_pct": spec.target_pct,
                 "min_entry_price_usd": spec.min_entry_price_usd,
                 "min_avg_volume_30d_dollar": spec.min_avg_volume_30d_dollar},
        "config": {
            "profile": "flat_liquid",
            "fee_pct_per_leg": LIQUID_FEE_PCT_PER_LEG,
            "spread_pct_rt": LIQUID_SPREAD_PCT_RT,
            "slippage_pct_rt": LIQUID_SLIPPAGE_PCT_RT,
            "total_friction_pct_round_trip": total_friction_pct,
            "survivorship_haircut_pct": LIQUID_SURVIVORSHIP_HAIRCUT,
        },
        "n_trades": int(in_window.height),
        "mean_hold_days": int(spec.horizon_trading_days),
        "gross": {
            "total_pnl_usd": gross_total_usd,
            "mean_pnl_pct": float(gross_pnl_pct_arr.mean()),
            "median_pnl_pct": float(np.median(gross_pnl_pct_arr)),
            "win_rate": float((gross_pnl_pct_arr > 0).mean()),
            "sharpe_equity_curve": float(gross_sharpe),
            "max_drawdown_pct": float(gross_max_dd),
        },
        "net": {
            "total_pnl_usd": net_total_usd,
            "mean_pnl_pct": float(net_pnl_pct_arr.mean()),
            "median_pnl_pct": float(np.median(net_pnl_pct_arr)),
            "win_rate": float((net_pnl_pct_arr > 0).mean()),
            "sharpe_equity_curve": float(net_sharpe),
            "max_drawdown_pct": float(net_max_dd),
        },
        "survivorship_haircut": {
            "haircut_pct": LIQUID_SURVIVORSHIP_HAIRCUT,
            "net_pnl_usd_after_haircut": haircut_total_usd,
            "net_sharpe_after_haircut": float(haircut_sharpe),
            "max_drawdown_pct_after_haircut": float(net_max_dd),
            "passes_gate_after_haircut": bool(haircut_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "gate": {
            "min_net_sharpe": GATE_MIN_NET_SHARPE,
            "passes_net_sharpe_gate": bool(net_sharpe >= GATE_MIN_NET_SHARPE),
            "passes_net_sharpe_gate_after_haircut": bool(haircut_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "friction_breakdown.json").write_text(json.dumps(friction_summary, indent=2))

    # Run-level manifest
    manifest = {
        "pipeline_step": PIPELINE_STEP,
        "spec_name": spec.name,
        "cohort_stats": stats,
        "walkforward_windows": [
            (s.isoformat(), e.isoformat()) for s, e in WINDOWS
        ],
        "n_universe_symbols": int(universe.height),
        "n_trades_total": int(in_window.height),
        "friction_profile": "flat_liquid_mid_cap",
        "survivorship_haircut_pct": LIQUID_SURVIVORSHIP_HAIRCUT,
        "results": {
            "gross_total_pnl_usd": gross_total_usd,
            "net_total_pnl_usd": net_total_usd,
            "haircut_total_pnl_usd": haircut_total_usd,
            "gross_sharpe": float(gross_sharpe),
            "net_sharpe": float(net_sharpe),
            "haircut_sharpe": float(haircut_sharpe),
            "passes_gate": bool(haircut_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"=== EQUITY MOMENTUM WALKFORWARD ({spec.name}) ===")
    print(f"  gross: ${gross_total_usd:,.0f}  mean +{gross_pnl_pct_arr.mean():.2f}%  sharpe {gross_sharpe:.3f}")
    print(f"  net:   ${net_total_usd:,.0f}  mean +{net_pnl_pct_arr.mean():.2f}%  sharpe {net_sharpe:.3f}")
    print(f"  haircut ({100*LIQUID_SURVIVORSHIP_HAIRCUT:.0f}%): ${haircut_total_usd:,.0f}  sharpe {haircut_sharpe:.3f}")
    gate_str = "PASS" if friction_summary["gate"]["passes_net_sharpe_gate_after_haircut"] else "FAIL"
    print(f"  Gate (haircut_sharpe >= {GATE_MIN_NET_SHARPE}): {gate_str}")
    print(f"  wall clock: {friction_summary['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
