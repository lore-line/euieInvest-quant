"""Phase B v3 friction-adjusted re-aggregation — Stream 2 v1 prep.

Per PR #1 issuecomment-4471981565 (server-team scope refocus + clarification):

  The friction-adjusted Sharpe is what we'll gate Stream 2 on going to
  paper trade (vs the gross-Sharpe of 1.88). Honest expectation: friction
  will eat 30-60% of gross on short-hold strategies. If net-Sharpe drops
  below 0.8, Stream 2 doesn't pass the lifecycle stage 2 → 3 gate.

Takes an existing sleeve-simulation `signals.parquet` (one row per trade,
produced by `paper_sleeve_simulate.py`), back-calculates pre-slippage
"gross" prices, then explicitly costs OUT three friction components:

  fee_cost_pct        = fee_pct_per_leg × 2 × 100  (round-trip taker)
  spread_cost_pct     = spread_pct_rt × 100  (bid-ask round-trip)
  slippage_cost_pct   = baked_in_slippage × 2 × 100  (existing 0.10%/leg)
                      + extra_slippage_pct_rt × 100  (configurable upgrade)

Then:
  net_pnl_pct  = gross_pnl_pct - fee_cost_pct - spread_cost_pct - slippage_cost_pct
  friction_pct_of_gross = (sum of friction) / max(|gross_pnl_pct|, ε) × 100

Aggregate metrics are recomputed on net_pnl_pct: net_sharpe, net_total_pnl,
net_win_rate, net_avg_holding, etc.

Default friction values (Kraken-Pro equivalent per server-team spec):
  - fee_pct_per_leg     = 0.0016 (0.16% taker)
  - spread_pct_rt       = 0.0005 (0.05% typical)
  - baked_in_slippage   = 0.001  (0.10% per leg, already in entered/exited prices)
  - extra_slippage      = 0.0    (set positive to upgrade beyond baked-in)

Output to the SAME run dir:
  - signals_with_friction.parquet — per-trade enrichment
  - friction_breakdown.json — aggregate net metrics + gate status
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "phase_b_v3_friction_extension_v1"

# Per server-team scope-refocus comment defaults (Kraken-Pro equivalent)
DEFAULT_FEE_PCT_PER_LEG = 0.0016        # 0.16% taker
DEFAULT_SPREAD_PCT_RT = 0.0005          # 0.05% bid-ask round-trip
DEFAULT_BAKED_IN_SLIPPAGE = 0.001       # 0.10% per leg — what paper_sleeve_simulate already applied
DEFAULT_EXTRA_SLIPPAGE_RT = 0.0         # upgrade beyond baked-in (set 0.0005-0.001 for stress test)

# Server-team gate: net-Sharpe >= 0.8 to pass Stream 2 lifecycle stage 2 → 3
GATE_MIN_NET_SHARPE = 0.8

# small_cap_atr profile per server-team friction formula
# (PR #1 issuecomment-4472873962):
#   fee_per_leg_usd       = 0 (Wealthsimple ws_equity is commission-free)
#   spread_per_leg_pct    = 0.0004 (0.04%)
#   slippage_per_leg_usd  = atr × 0.18 (stops) or atr × 0.08 (market/TP)
# Requires per-trade `entry_atr_pct_14` column on the signals frame
# (joined upstream by phase_b_v3_liquidity_filter or equivalent).
SMALL_CAP_ATR_FEE_PCT_PER_LEG = 0.0
SMALL_CAP_ATR_SPREAD_PCT_PER_LEG = 0.0004
SMALL_CAP_ATR_STOP_SLIP_FACTOR = 0.18
SMALL_CAP_ATR_MARKET_SLIP_FACTOR = 0.08


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--signals", type=Path, required=True,
        help="Path to signals.parquet produced by paper_sleeve_simulate.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Default: same dir as --signals.",
    )
    p.add_argument(
        "--fee-pct-per-leg", type=float, default=DEFAULT_FEE_PCT_PER_LEG,
        help=f"Per-leg taker fee. Default {DEFAULT_FEE_PCT_PER_LEG:.4f} (0.16%% Kraken-Pro taker).",
    )
    p.add_argument(
        "--spread-pct-rt", type=float, default=DEFAULT_SPREAD_PCT_RT,
        help=f"Bid-ask round-trip cost. Default {DEFAULT_SPREAD_PCT_RT:.4f} (0.05%%).",
    )
    p.add_argument(
        "--baked-in-slippage", type=float, default=DEFAULT_BAKED_IN_SLIPPAGE,
        help=f"Per-leg slippage already in signals.parquet entered/exited prices. "
             f"Default {DEFAULT_BAKED_IN_SLIPPAGE:.4f} (matches paper_sleeve_simulate default).",
    )
    p.add_argument(
        "--extra-slippage-pct-rt", type=float, default=DEFAULT_EXTRA_SLIPPAGE_RT,
        help=f"Additional round-trip slippage beyond baked-in. Default {DEFAULT_EXTRA_SLIPPAGE_RT:.4f}.",
    )
    p.add_argument(
        "--profile", type=str, default="flat", choices=["flat", "small_cap_atr"],
        help="Friction profile. 'flat' (default) uses the fee/spread/slip flag values "
             "(mid-cap-class). 'small_cap_atr' uses the server-team ATR-scaled formula "
             "(fee=0, spread=0.04%%/leg, slip=ATR×0.18 for stops / ×0.08 for TP). "
             "small_cap_atr requires `entry_atr_pct_14` column on signals.parquet.",
    )
    p.add_argument(
        "--survivorship-haircut-pct", type=float, default=0.25,
        help="Survivorship-bias haircut applied to net P&L per server-team convention "
             "(PR #1 issuecomment-4472966936). Default 0.25 = 25%% (quant-stream default, "
             "broad universe high-turnover ML). 0.0 for currency/single-symbol, 0.05-0.15 "
             "for liquid mid-cap-only, 0.20-0.30 for small-cap-heavy. Written to "
             "friction_breakdown.json; platform respects this value at ingest time. "
             "Also produces signals_haircut_adjusted.parquet with per-trade adjusted P&L.",
    )
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _equity_curve_sharpe(
    pnl_usd_chronological: np.ndarray, sleeve_usd: float = 10_000.0,
) -> tuple[float, float]:
    """Replicate `paper_sleeve_simulate._compute_equity_curve` + sharpe math.

    Equity curve = starting_sleeve + cumulative realized P&L (per exit event).
    Per-event returns = diff(curve) / lagged_curve.
    Sharpe = mean / std × sqrt(252).

    This matches the methodology that produced the original 1.88/1.51 numbers,
    so friction-adjusted Sharpe here is directly comparable to those.

    Returns (sharpe, max_drawdown_pct).
    """
    if len(pnl_usd_chronological) == 0:
        return 0.0, 0.0
    curve = np.concatenate([[sleeve_usd], sleeve_usd + np.cumsum(pnl_usd_chronological)])
    if len(curve) <= 1:
        return 0.0, 0.0
    returns = np.diff(curve) / np.where(curve[:-1] == 0, 1e-9, curve[:-1])
    sharpe = (
        float(np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252.0))
        if np.std(returns) > 0 else 0.0
    )
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    max_dd = float(dd.min())
    return sharpe, max_dd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    signals_path = _resolve(args.signals)
    if not signals_path.exists():
        print(f"ERROR: signals not found at {signals_path}")
        return 1
    out_dir = _resolve(args.out_dir) if args.out_dir else signals_path.parent

    print(f"phase_b_v3_friction_extension v1")
    print(f"  signals:                {signals_path}")
    print(f"  out_dir:                {out_dir}")
    print(f"  fee_pct_per_leg:        {args.fee_pct_per_leg:.4f} ({100*args.fee_pct_per_leg:.2f}%)")
    print(f"  spread_pct_rt:          {args.spread_pct_rt:.4f} ({100*args.spread_pct_rt:.2f}%)")
    print(f"  baked_in_slippage/leg:  {args.baked_in_slippage:.4f} ({100*args.baked_in_slippage:.2f}%)")
    print(f"  extra_slippage_pct_rt:  {args.extra_slippage_pct_rt:.4f} ({100*args.extra_slippage_pct_rt:.2f}%)")

    sig = pl.read_parquet(signals_path)
    print(f"  loaded {sig.height:,} trades")
    if sig.height == 0:
        print("ERROR: no trades to enrich")
        return 1

    # Validate required columns
    required = {"entered_price", "exited_price", "position_size_usd", "realized_pnl_usd",
                "entered_at", "exited_at"}
    missing = required - set(sig.columns)
    if missing:
        print(f"ERROR: signals.parquet missing required cols: {sorted(missing)}")
        return 1

    # Back-calculate pre-slippage prices.
    # paper_sleeve_simulate applies entry_price = fill_open * (1 + slippage)
    # and exit_price = last_close * (1 - slippage). So:
    #   gross_entry = entered_price / (1 + baked_in_slippage)
    #   gross_exit  = exited_price  / (1 - baked_in_slippage)
    enriched = sig.with_columns(
        gross_entry_price=pl.col("entered_price") / (1.0 + args.baked_in_slippage),
        gross_exit_price=pl.col("exited_price") / (1.0 - args.baked_in_slippage),
    ).with_columns(
        gross_pnl_pct=(pl.col("gross_exit_price") / pl.col("gross_entry_price") - 1.0) * 100.0,
    )

    if args.profile == "small_cap_atr":
        # Per-trade ATR-scaled friction. Requires entry_atr_pct_14 column
        # (joined upstream from features.parquet — see phase_b_v3_liquidity_filter).
        if "entry_atr_pct_14" not in enriched.columns:
            print(f"ERROR: --profile small_cap_atr requires 'entry_atr_pct_14' column on signals.parquet "
                  f"(join upstream from features.parquet via phase_b_v3_liquidity_filter or equivalent)")
            return 1

        # Fee: 0 for ws_equity
        # Spread: 0.04% per leg = 0.08% RT
        # Slippage: ATR × 0.18 per stop leg, × 0.08 per TP leg
        # exit_reason values from paper_sleeve_simulate are 'stop', 'target', 'time', 'end_of_period'
        # Treat 'stop' as stop-order slippage, everything else as market slippage.
        fee_cost_pct_expr = pl.lit(2.0 * SMALL_CAP_ATR_FEE_PCT_PER_LEG * 100.0)
        spread_cost_pct_expr = pl.lit(2.0 * SMALL_CAP_ATR_SPREAD_PCT_PER_LEG * 100.0)
        # Per-leg slippage: entry is always market (entry_atr × market_slip),
        # exit varies by reason (stop → stop_slip, else market_slip)
        # entry_atr_pct_14 is in decimal (0.05 = 5% ATR/price)
        slippage_cost_pct_expr = (
            pl.col("entry_atr_pct_14") * 100.0 * (
                pl.lit(SMALL_CAP_ATR_MARKET_SLIP_FACTOR)  # entry leg, always market
                + pl.when(pl.col("exit_reason") == "stop")
                .then(pl.lit(SMALL_CAP_ATR_STOP_SLIP_FACTOR))
                .otherwise(pl.lit(SMALL_CAP_ATR_MARKET_SLIP_FACTOR))  # exit leg
            )
        )
        enriched = enriched.with_columns(
            fee_cost_pct=fee_cost_pct_expr,
            spread_cost_pct=spread_cost_pct_expr,
            slippage_cost_pct=slippage_cost_pct_expr,
        )
        # Total per-trade friction varies; can't precompute a single constant
        enriched = enriched.with_columns(
            total_friction_pct_per_trade=pl.col("fee_cost_pct") + pl.col("spread_cost_pct") + pl.col("slippage_cost_pct"),
        ).with_columns(
            net_pnl_pct=pl.col("gross_pnl_pct") - pl.col("total_friction_pct_per_trade"),
            friction_pct_of_gross=pl.when(pl.col("gross_pnl_pct").abs() > 0.01)
            .then(pl.col("total_friction_pct_per_trade") / pl.col("gross_pnl_pct").abs() * 100.0)
            .otherwise(None),
        )
        # For reporting consistency, compute mean per-trade friction
        fee_cost_pct = float(enriched["fee_cost_pct"].mean())
        spread_cost_pct = float(enriched["spread_cost_pct"].mean())
        slippage_cost_pct = float(enriched["slippage_cost_pct"].mean())
        total_friction_pct = fee_cost_pct + spread_cost_pct + slippage_cost_pct
    else:
        # 'flat' profile (default, mid-cap-class)
        fee_cost_pct = 2.0 * args.fee_pct_per_leg * 100.0       # round-trip
        spread_cost_pct = args.spread_pct_rt * 100.0
        slippage_cost_pct = (2.0 * args.baked_in_slippage + args.extra_slippage_pct_rt) * 100.0
        total_friction_pct = fee_cost_pct + spread_cost_pct + slippage_cost_pct

        enriched = enriched.with_columns(
            fee_cost_pct=pl.lit(fee_cost_pct),
            spread_cost_pct=pl.lit(spread_cost_pct),
            slippage_cost_pct=pl.lit(slippage_cost_pct),
        ).with_columns(
            net_pnl_pct=pl.col("gross_pnl_pct") - pl.col("fee_cost_pct")
                        - pl.col("spread_cost_pct") - pl.col("slippage_cost_pct"),
        ).with_columns(
            friction_pct_of_gross=pl.when(pl.col("gross_pnl_pct").abs() > 0.01)
            .then(total_friction_pct / pl.col("gross_pnl_pct").abs() * 100.0)
            .otherwise(None),
        )

    # Also compute net USD pnl (recompute from net_pnl_pct × position_size_usd)
    # total_friction_usd uses per-trade friction sum (works for both flat + ATR profiles)
    enriched = enriched.with_columns(
        net_pnl_usd=pl.col("net_pnl_pct") / 100.0 * pl.col("position_size_usd"),
        total_friction_usd=(
            pl.col("fee_cost_pct") + pl.col("spread_cost_pct") + pl.col("slippage_cost_pct")
        ) / 100.0 * pl.col("position_size_usd"),
    )

    # Hold trading days (date diff). entered_at/exited_at are ISO datetime
    # strings (e.g. "2025-04-01T00:00:00+00:00") per paper_sleeve_simulate
    # output. Parse to datetime then take date diff.
    ent_dtype = enriched["entered_at"].dtype
    ext_dtype = enriched["exited_at"].dtype
    if ent_dtype == pl.String:
        # ISO 8601 with timezone, e.g. "2025-04-01T00:00:00+00:00"
        enriched = enriched.with_columns(
            entered_at_dt=pl.col("entered_at").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%:z", time_zone="UTC", strict=False,
            ),
            exited_at_dt=pl.col("exited_at").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%:z", time_zone="UTC", strict=False,
            ),
        )
    else:
        enriched = enriched.with_columns(
            entered_at_dt=pl.col("entered_at"),
            exited_at_dt=pl.col("exited_at"),
        )
    enriched = enriched.with_columns(
        hold_days=(pl.col("exited_at_dt") - pl.col("entered_at_dt")).dt.total_days(),
    ).drop(["entered_at_dt", "exited_at_dt"])

    out_path = out_dir / "signals_with_friction.parquet"
    enriched.write_parquet(out_path)
    print(f"  wrote {out_path}")

    # Aggregate
    n_trades = enriched.height
    gross_total_usd = float(enriched["realized_pnl_usd"].sum())
    net_total_usd = float(enriched["net_pnl_usd"].sum())
    total_friction_usd = float(enriched["total_friction_usd"].sum())

    gross_pnl_pct_arr = enriched["gross_pnl_pct"].to_numpy()
    net_pnl_pct_arr = enriched["net_pnl_pct"].to_numpy()
    hold_days_arr = enriched["hold_days"].to_numpy()

    gross_mean_pct = float(gross_pnl_pct_arr.mean())
    net_mean_pct = float(net_pnl_pct_arr.mean())
    gross_median_pct = float(np.median(gross_pnl_pct_arr))
    net_median_pct = float(np.median(net_pnl_pct_arr))

    gross_win_rate = float((gross_pnl_pct_arr > 0).mean())
    net_win_rate = float((net_pnl_pct_arr > 0).mean())
    mean_hold = float(hold_days_arr.mean()) if len(hold_days_arr) else 1.0

    # Equity-curve Sharpe matching paper_sleeve_simulate methodology — sort
    # by exit_at, cumulative pnl, per-event returns, annualized sqrt(252).
    # This is directly comparable to the original 1.88 / 1.51 numbers.
    enriched_sorted = enriched.sort("exited_at")
    gross_pnl_usd_chron = (
        enriched_sorted["realized_pnl_usd"].to_numpy()
        if "realized_pnl_usd" in enriched_sorted.columns else
        (enriched_sorted["gross_pnl_pct"].to_numpy() / 100.0
         * enriched_sorted["position_size_usd"].to_numpy())
    )
    net_pnl_usd_chron = enriched_sorted["net_pnl_usd"].to_numpy()
    SLEEVE_USD = 10_000.0  # matches paper_sleeve_simulate default
    gross_sharpe, gross_max_dd = _equity_curve_sharpe(gross_pnl_usd_chron, SLEEVE_USD)
    net_sharpe, net_max_dd = _equity_curve_sharpe(net_pnl_usd_chron, SLEEVE_USD)

    friction_pct_of_gross_total = (
        100.0 * total_friction_usd / abs(gross_total_usd) if abs(gross_total_usd) > 0.01 else None
    )

    # Apply survivorship-bias haircut per server-team convention
    # (PR #1 issuecomment-4472966936). The haircut is a derate on EXPECTED
    # forward performance, not a re-statement of historical realized P&L.
    # The standard convention from that comment table is:
    #   haircut_sharpe = raw_sharpe × (1 - haircut)
    #   haircut_pnl    = raw_pnl × (1 - haircut)
    # Note: scaling per-trade P&L doesn't change the Sharpe of that scaled
    # series (mean and std both scale). The Sharpe-level scaling is the
    # SEPARATE expected-forward adjustment; it's what the platform applies
    # at ingest. We compute both for transparency.
    haircut = max(0.0, min(1.0, args.survivorship_haircut_pct))
    haircut_adjusted = enriched.with_columns(
        haircut_pct=pl.lit(haircut * 100.0),
        net_pnl_pct_after_haircut=pl.col("net_pnl_pct") * (1.0 - haircut),
        net_pnl_usd_after_haircut=pl.col("net_pnl_usd") * (1.0 - haircut),
    )
    haircut_path = out_dir / "signals_haircut_adjusted.parquet"
    haircut_adjusted.write_parquet(haircut_path)
    print(f"  wrote {haircut_path}  (haircut={100*haircut:.1f}%)")

    haircut_total_usd = net_total_usd * (1.0 - haircut)
    haircut_sharpe = net_sharpe * (1.0 - haircut)  # platform-convention math
    haircut_max_dd = net_max_dd  # drawdown not haircut-adjusted (it's a path property)

    summary = {
        "pipeline_step": PIPELINE_STEP,
        "signals_path": str(signals_path),
        "config": {
            "profile": args.profile,
            "fee_pct_per_leg": args.fee_pct_per_leg if args.profile == "flat" else SMALL_CAP_ATR_FEE_PCT_PER_LEG,
            "spread_pct_rt": args.spread_pct_rt if args.profile == "flat" else (2.0 * SMALL_CAP_ATR_SPREAD_PCT_PER_LEG),
            "baked_in_slippage_per_leg": args.baked_in_slippage if args.profile == "flat" else None,
            "extra_slippage_pct_rt": args.extra_slippage_pct_rt if args.profile == "flat" else None,
            "atr_stop_slip_factor": SMALL_CAP_ATR_STOP_SLIP_FACTOR if args.profile == "small_cap_atr" else None,
            "atr_market_slip_factor": SMALL_CAP_ATR_MARKET_SLIP_FACTOR if args.profile == "small_cap_atr" else None,
            "fee_cost_pct_round_trip_mean": fee_cost_pct,
            "spread_cost_pct_round_trip_mean": spread_cost_pct,
            "slippage_cost_pct_round_trip_mean": slippage_cost_pct,
            "total_friction_pct_round_trip_mean": total_friction_pct,
        },
        "n_trades": int(n_trades),
        "mean_hold_days": mean_hold,
        "gross": {
            "total_pnl_usd": gross_total_usd,
            "mean_pnl_pct": gross_mean_pct,
            "median_pnl_pct": gross_median_pct,
            "win_rate": gross_win_rate,
            "sharpe_equity_curve": float(gross_sharpe),
            "max_drawdown_pct": float(gross_max_dd),
        },
        "net": {
            "total_pnl_usd": net_total_usd,
            "mean_pnl_pct": net_mean_pct,
            "median_pnl_pct": net_median_pct,
            "win_rate": net_win_rate,
            "sharpe_equity_curve": float(net_sharpe),
            "max_drawdown_pct": float(net_max_dd),
        },
        "friction": {
            "total_friction_usd": total_friction_usd,
            "friction_pct_of_gross_pnl": friction_pct_of_gross_total,
            "friction_drag_per_trade_pct": total_friction_pct,
        },
        "survivorship_haircut": {
            "haircut_pct": haircut,
            "net_pnl_usd_after_haircut": haircut_total_usd,
            "net_sharpe_after_haircut": float(haircut_sharpe),
            "max_drawdown_pct_after_haircut": float(haircut_max_dd),
            "passes_gate_after_haircut": bool(haircut_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "gate": {
            "min_net_sharpe": GATE_MIN_NET_SHARPE,
            "passes_net_sharpe_gate": bool(net_sharpe >= GATE_MIN_NET_SHARPE),
            "passes_net_sharpe_gate_after_haircut": bool(haircut_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    summary_path = out_dir / "friction_breakdown.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_path}")

    print()
    print(f"=== FRICTION-ADJUSTED RESULT ===")
    print(f"  n_trades:              {n_trades:,}")
    print(f"  mean_hold_days:        {mean_hold:.2f}")
    print()
    print(f"  GROSS:  total_pnl ${gross_total_usd:,.0f}  mean_pct {gross_mean_pct:+.2f}%  win {100*gross_win_rate:.1f}%  sharpe {gross_sharpe:.3f}  max_dd {100*gross_max_dd:.1f}%")
    print(f"  NET:    total_pnl ${net_total_usd:,.0f}  mean_pct {net_mean_pct:+.2f}%  win {100*net_win_rate:.1f}%  sharpe {net_sharpe:.3f}  max_dd {100*net_max_dd:.1f}%")
    print()
    print(f"  Friction breakdown (per trade, % of notional):")
    print(f"    fee_cost_pct:       {fee_cost_pct:.3f}%")
    print(f"    spread_cost_pct:    {spread_cost_pct:.3f}%")
    print(f"    slippage_cost_pct:  {slippage_cost_pct:.3f}%")
    print(f"    total:              {total_friction_pct:.3f}%")
    if friction_pct_of_gross_total is not None:
        print(f"  Friction as % of gross pnl: {friction_pct_of_gross_total:.1f}%")
    print()
    gate_str = "PASS" if summary["gate"]["passes_net_sharpe_gate"] else "FAIL"
    print(f"  Gate (net_sharpe >= {GATE_MIN_NET_SHARPE}): {gate_str} (actual {net_sharpe:.3f})")
    hc_gate_str = "PASS" if summary["gate"]["passes_net_sharpe_gate_after_haircut"] else "FAIL"
    print(f"  Survivorship haircut ({100*haircut:.1f}%): net_pnl ${haircut_total_usd:,.0f}  sharpe {haircut_sharpe:.3f}  {hc_gate_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
