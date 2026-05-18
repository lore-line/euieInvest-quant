"""Crypto momentum walkforward validator — Stream 2c v1.

Per PR #1 issuecomment-4473364135/4473464239/4473682523.

Differs from equity_momentum_walkforward:
  - Universe fixed at 8 symbols (no liquidity filter)
  - 4×6mo walkforward windows (vs 5×6mo for equity) per our agreement
    on 24-month-history-only Kraken constraint
  - Kraken Pro fee tiers as friction profile (not flat liquid mid-cap)
  - 0% survivorship haircut (single-asset class, no universe selection)

Kraken Pro fee profiles per server-team correction
(PR #1 issuecomment-4473682523):
  kraken_pro_starter:  $0+ tier, all-taker — 0.80% RT
  kraken_pro_active:   $50K+ 30d-rolling, maker-favored — 0.35% RT
  kraken_pro_pro:      $500K+ 30d-rolling, near-all-maker — 0.23% RT

Default is `kraken_pro_active` since that matches realistic operator
volume for momentum on 8-symbol universe.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from dataclasses import replace

import numpy as np
import polars as pl

from quant.tracks.crypto_momentum_label import (
    CRYPTO_UNIVERSE,
    CryptoMomentumSpec,
    SPEC_DEFAULT,
    compute_crypto_momentum_label,
    label_statistics,
)
from quant.tracks.phase_b_v3_friction_extension import _equity_curve_sharpe

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "crypto_momentum_walkforward_v1"

# Kraken Pro friction profiles (RT = round-trip)
KRAKEN_FRICTION_PROFILES = {
    "kraken_pro_starter": 0.0080,  # 0.80% RT — all-taker $0+ tier
    "kraken_pro_active":  0.0035,  # 0.35% RT — maker-favored $50K+ tier
    "kraken_pro_pro":     0.0023,  # 0.23% RT — near-all-maker $500K+ tier
}

# 4×6mo walkforward windows over Kraken's 24-month effective history
# (BTC-USD min date 2024-05-29, max 2026-05-17 → ~24mo)
WINDOWS_4X6MO: list[tuple[date, date]] = [
    (date(2024, 6, 1), date(2024, 11, 30)),  # Bull early
    (date(2024, 12, 1), date(2025, 5, 31)),  # Bull late / peak
    (date(2025, 6, 1), date(2025, 11, 30)),  # Corrective
    (date(2025, 12, 1), date(2026, 5, 17)),  # Recovery
]

GATE_MIN_NET_SHARPE = 0.8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ohlcv", type=Path, default=Path("/workspace/runs/crypto_ohlcv.parquet"),
        help="Path to 8-symbol crypto OHLCV parquet (fetched from /api/v1/ohlcv).",
    )
    p.add_argument(
        "--target-pct", type=float, default=0.30,
        help="Target return for label cohort. Default 0.30 (30%%). Server-team's "
             "edge finding was at 0.40 — worth running both.",
    )
    p.add_argument(
        "--horizon-trading-days", type=int, default=60,
        help="Forward window for label. Default 60.",
    )
    p.add_argument(
        "--entry-breakout-days", type=int, default=14,
        help="Donchian-high lookback for entry. Default 14.",
    )
    p.add_argument(
        "--friction-profile", type=str, default="kraken_pro_active",
        choices=list(KRAKEN_FRICTION_PROFILES.keys()),
        help="Kraken Pro fee tier. Default 'kraken_pro_active' (realistic operator volume).",
    )
    p.add_argument("--position-size-usd", type=float, default=1000.0)
    p.add_argument("--sleeve-usd", type=float, default=10_000.0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    ohlcv_path = _resolve(args.ohlcv)
    if not ohlcv_path.exists():
        print(f"ERROR: {ohlcv_path} not found. Fetch via:")
        print(f"  curl 'http://100.68.86.56:8443/api/v1/ohlcv?symbols=BTC-USD,...&format=parquet' -o {ohlcv_path}")
        return 1

    spec = CryptoMomentumSpec(
        name=f"g{int(args.target_pct * 100)}",
        entry_breakout_days=args.entry_breakout_days,
        horizon_trading_days=args.horizon_trading_days,
        target_pct=args.target_pct,
    )
    friction_rt_pct = KRAKEN_FRICTION_PROFILES[args.friction_profile] * 100

    today = date.today().isoformat()
    out_dir = (
        _resolve(args.out_dir) if args.out_dir is not None
        else _REPO_ROOT / "runs" / f"{today}-crypto_momentum_{spec.name}_{args.friction_profile}_walkforward"
    )
    if not out_dir.parent.exists():
        alt = Path("/workspace/runs")
        if alt.exists():
            out_dir = alt / f"{today}-crypto_momentum_{spec.name}_{args.friction_profile}_walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{PIPELINE_STEP} — spec {spec.name} + {args.friction_profile}")
    print(f"  ohlcv:        {ohlcv_path}")
    print(f"  universe:     {len(CRYPTO_UNIVERSE)} symbols")
    print(f"  spec:         breakout={spec.entry_breakout_days}d horizon={spec.horizon_trading_days}td target=+{100*spec.target_pct:.0f}%")
    print(f"  friction:     {args.friction_profile} = {friction_rt_pct:.3f}% RT")
    print(f"  windows:      {len(WINDOWS_4X6MO)} chronological splits")
    print()

    ohlcv = pl.read_parquet(ohlcv_path)
    print(f"  loaded {ohlcv.height:,} OHLCV rows ({ohlcv['symbol'].n_unique()} symbols, "
          f"{ohlcv['date'].min()} → {ohlcv['date'].max()})")

    # Filter to expected universe (defensive)
    ohlcv = ohlcv.filter(pl.col("symbol").is_in(CRYPTO_UNIVERSE))
    print(f"  after universe filter: {ohlcv.height:,} rows / {ohlcv['symbol'].n_unique()} symbols")

    # Ensure date is Date type
    if ohlcv["date"].dtype != pl.Date:
        ohlcv = ohlcv.with_columns(pl.col("date").cast(pl.Date, strict=False))

    # Apply label
    labeled = compute_crypto_momentum_label(ohlcv, spec)
    stats = label_statistics(labeled, spec)
    print(f"  cohort stats: {stats['n_labelable_rows']:,} labelable, "
          f"{stats['n_winners']:,} winners ({100*stats['winner_rate']:.1f}%)")

    # Per-trade table (every entry signal = one trade). Same null-aware
    # detection as label module — crypto's close_adj column is all-null.
    if "close_adj" in labeled.columns and labeled["close_adj"].drop_nulls().len() > 0:
        price_col = "close_adj"
    else:
        price_col = "close"
    H = spec.horizon_trading_days
    entry_signals = labeled.filter(
        (pl.col(price_col) > pl.col("prior_n_day_high"))
        & pl.col(spec.label_column()).is_not_null()
        & pl.col(f"exit_close_{H}td").is_not_null()
    ).with_columns(
        exit_close=pl.col(f"exit_close_{H}td"),
        exit_date=pl.col(f"exit_date_{H}td"),
    )
    trades = entry_signals.select([
        pl.col("symbol"),
        pl.col("date").alias("entry_date"),
        pl.col(price_col).alias("entry_price"),
        pl.col("exit_date"),
        pl.col("exit_close").alias("exit_price"),
        ((pl.col("exit_close") / pl.col(price_col) - 1.0) * 100.0).alias("gross_pnl_pct"),
        pl.col(spec.label_column()).alias("is_winner_label"),
    ])
    print(f"  per-trade table: {trades.height:,} trades")

    # Filter to walkforward window
    in_window = trades.filter(
        (pl.col("entry_date") >= WINDOWS_4X6MO[0][0])
        & (pl.col("entry_date") <= WINDOWS_4X6MO[-1][1])
    )
    print(f"  in walkforward window: {in_window.height:,} trades")

    # Per-window stats
    rows = []
    for w_idx, (w_start, w_end) in enumerate(WINDOWS_4X6MO):
        bucket = in_window.filter(
            (pl.col("entry_date") >= w_start) & (pl.col("entry_date") <= w_end)
        )
        if bucket.height == 0:
            continue
        gains = bucket["gross_pnl_pct"].to_numpy()
        per_trade_usd = (gains / 100.0) * args.position_size_usd
        sharpe, max_dd = _equity_curve_sharpe(
            bucket.sort("exit_date")["gross_pnl_pct"].to_numpy() / 100.0 * args.position_size_usd,
            args.sleeve_usd,
        )
        rows.append({
            "window_idx": w_idx, "window_start": w_start, "window_end": w_end,
            "n_trades": int(bucket.height), "win_rate": float((gains > 0).mean()),
            "mean_gross_pnl_pct": float(gains.mean()),
            "total_gross_pnl_usd": float(per_trade_usd.sum()),
            "gross_sharpe": float(sharpe), "max_drawdown_pct": float(max_dd),
            "per_symbol_trades": {
                row["symbol"]: int(row["n"])
                for row in bucket.group_by("symbol").len().rename({"len": "n"}).iter_rows(named=True)
            },
        })
    win_stats = pl.DataFrame([
        {k: v for k, v in r.items() if k != "per_symbol_trades"} for r in rows
    ]) if rows else pl.DataFrame()

    if win_stats.height > 0:
        win_stats.write_parquet(out_dir / "walk_forward.parquet")
        print(f"  per-window:")
        for r in rows:
            print(f"    {r['window_start']}..{r['window_end']}: "
                  f"n={r['n_trades']:>3,}  win={100*r['win_rate']:>4.1f}%  "
                  f"mean_pnl={r['mean_gross_pnl_pct']:>+5.1f}%  sharpe={r['gross_sharpe']:>+5.2f}")

    in_window.write_parquet(out_dir / "signals.parquet")

    # Aggregate friction (Kraken Pro tier-aware, 0% survivorship)
    if in_window.height > 0:
        gross_arr = in_window["gross_pnl_pct"].to_numpy()
        net_arr = gross_arr - friction_rt_pct
        in_window_sorted = in_window.sort("exit_date")
        gross_chron = in_window_sorted["gross_pnl_pct"].to_numpy() / 100.0 * args.position_size_usd
        net_chron = (in_window_sorted["gross_pnl_pct"].to_numpy() - friction_rt_pct) / 100.0 * args.position_size_usd
        gross_sharpe, gross_max_dd = _equity_curve_sharpe(gross_chron, args.sleeve_usd)
        net_sharpe, net_max_dd = _equity_curve_sharpe(net_chron, args.sleeve_usd)

        # 0% haircut for single-asset class crypto
        haircut = 0.0
        haircut_sharpe = net_sharpe * (1.0 - haircut)
        gross_total = float(gross_chron.sum())
        net_total = float(net_chron.sum())
        friction_pct_of_gross = (
            100.0 * (gross_total - net_total) / abs(gross_total) if abs(gross_total) > 0.01 else None
        )
    else:
        gross_arr = net_arr = np.array([])
        gross_total = net_total = gross_sharpe = net_sharpe = haircut_sharpe = 0.0
        gross_max_dd = net_max_dd = 0.0
        friction_pct_of_gross = None
        haircut = 0.0  # 0% for single-asset crypto

    friction_summary = {
        "pipeline_step": PIPELINE_STEP,
        "spec": {"name": spec.name, "entry_breakout_days": spec.entry_breakout_days,
                 "horizon_trading_days": spec.horizon_trading_days,
                 "target_pct": spec.target_pct},
        "universe": CRYPTO_UNIVERSE,
        "config": {
            "profile": args.friction_profile,
            "friction_pct_round_trip": friction_rt_pct,
            "survivorship_haircut_pct": 0.0,
        },
        "n_trades": int(in_window.height),
        "mean_hold_days": int(spec.horizon_trading_days),
        "gross": {
            "total_pnl_usd": gross_total,
            "mean_pnl_pct": float(gross_arr.mean()) if len(gross_arr) else 0.0,
            "median_pnl_pct": float(np.median(gross_arr)) if len(gross_arr) else 0.0,
            "win_rate": float((gross_arr > 0).mean()) if len(gross_arr) else 0.0,
            "sharpe_equity_curve": float(gross_sharpe),
            "max_drawdown_pct": float(gross_max_dd),
        },
        "net": {
            "total_pnl_usd": net_total,
            "mean_pnl_pct": float(net_arr.mean()) if len(net_arr) else 0.0,
            "median_pnl_pct": float(np.median(net_arr)) if len(net_arr) else 0.0,
            "win_rate": float((net_arr > 0).mean()) if len(net_arr) else 0.0,
            "sharpe_equity_curve": float(net_sharpe),
            "max_drawdown_pct": float(net_max_dd),
        },
        "survivorship_haircut": {
            "haircut_pct": haircut,
            "net_pnl_usd_after_haircut": net_total * (1.0 - haircut),
            "net_sharpe_after_haircut": float(haircut_sharpe),
            "max_drawdown_pct_after_haircut": float(net_max_dd),
            "passes_gate_after_haircut": bool(haircut_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "friction_pct_of_gross": friction_pct_of_gross,
        "gate": {
            "min_net_sharpe": GATE_MIN_NET_SHARPE,
            "passes_net_sharpe_gate": bool(net_sharpe >= GATE_MIN_NET_SHARPE),
        },
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "friction_breakdown.json").write_text(json.dumps(friction_summary, indent=2))

    manifest = {
        "pipeline_step": PIPELINE_STEP,
        "spec_name": spec.name,
        "friction_profile": args.friction_profile,
        "universe": CRYPTO_UNIVERSE,
        "cohort_stats": stats,
        "windows": [(s.isoformat(), e.isoformat()) for s, e in WINDOWS_4X6MO],
        "n_trades_total": int(in_window.height),
        "results": friction_summary["net"] | {"haircut_sharpe": float(haircut_sharpe)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"=== CRYPTO MOMENTUM WALKFORWARD ({spec.name} @ {args.friction_profile}) ===")
    print(f"  gross: ${gross_total:,.0f}  mean +{(gross_arr.mean() if len(gross_arr) else 0):.2f}%  sharpe {gross_sharpe:.3f}")
    print(f"  net:   ${net_total:,.0f}  mean +{(net_arr.mean() if len(net_arr) else 0):.2f}%  sharpe {net_sharpe:.3f}")
    print(f"  friction: {friction_rt_pct:.2f}% RT = {friction_pct_of_gross:.1f}% of gross" if friction_pct_of_gross else f"  friction: {friction_rt_pct:.2f}% RT")
    gate_str = "PASS" if friction_summary["gate"]["passes_net_sharpe_gate"] else "FAIL"
    print(f"  Gate (net_sharpe >= {GATE_MIN_NET_SHARPE}): {gate_str}")
    print(f"  wall clock: {friction_summary['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
