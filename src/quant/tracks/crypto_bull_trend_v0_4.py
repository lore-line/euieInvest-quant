"""P2 Day 1 v0.4 — crypto bull-trend strategy class (issue #21).

Per server-team multi-strategy framing (issue #20 inverse-gating result):
this strategy is positioned as the **steady_bull-regime alt** that pairs
with inverse-amplified DCA's bear-coverage. inverse_aggressive profile
sets steady_bull=0.0 (capital hole when market is in clean uptrend).
P2 bull-trend strategy fills that hole.

Strategy class candidate #1 from issue #21 spec: weekly Donchian breakout.
v0.4 implements as wider-daily Donchian (60d entry breakout, hold until
+50% target OR 120-day stop). True weekly-bar implementation is v0.5.

Evaluation methodology (v0.4):
  - Universe: 8-symbol crypto from sidecar (BTC + 7 alts)
  - Signal: close > rolling_max(close, 60d shifted by 1) — Donchian breakout
  - Position management: enter on breakout signal, exit at TP+50% or 120d
  - Friction: kraken_pro_active (0.35% round-trip)
  - Per-regime CAGR using P1 v0.4 labels (joined on date)
  - Baseline: "hold BTC during steady_bull-only days" comparator

Acceptance (v0.4):
  - Strategy CAGR-conditional-on-steady_bull-days >= +10% annualized
    (the bar from inverse-gating analysis — alt must clear DCA's
    marginal bear-regime contribution scaled to bull regime)
  - Strategy beats "hold BTC during steady_bull" baseline
  - Per-regime breakdown reported for multi-strategy allocator decision
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import requests


SIDECAR_BASE_URL = os.environ.get("EUIEINVEST_API_BASE_URL", "http://100.68.86.56:8443")
CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
                   "AVAX-USD", "DOT-USD", "LINK-USD", "ATOM-USD"]

# Strategy params
ENTRY_BREAKOUT_DAYS = 60     # Donchian high lookback
HORIZON_TRADING_DAYS = 120   # max hold
TARGET_PCT = 0.50            # TP at +50%
FRICTION_RT = 0.0035         # kraken_pro_active round-trip

REGIME_LABELS_PATH = Path("data/quant_publish/regime_labels_v1.parquet")
DATA_START = "2024-05-29"    # earliest BTC date in sidecar
DATA_END = "2026-05-17"


def fetch_crypto() -> pl.DataFrame:
    """Fetch 8-symbol crypto OHLCV from sidecar /api/v1/ohlcv."""
    url = f"{SIDECAR_BASE_URL.rstrip('/')}/api/v1/ohlcv"
    params = {"symbols": ",".join(CRYPTO_UNIVERSE),
              "since": DATA_START, "until": DATA_END}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pl.read_parquet(io.BytesIO(r.content))
    return df.sort(["symbol", "date"])


def compute_breakout_entries(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """Mark each row as breakout-entry-eligible.

    Donchian breakout: close > max(close over prior N days, excluding today).
    Returns OHLCV with added column `is_breakout` (Boolean).
    """
    df = ohlcv.sort(["symbol", "date"]).with_columns(
        prior_n_day_high=(
            pl.col("close").shift(1).rolling_max(window_size=ENTRY_BREAKOUT_DAYS)
            .over("symbol")
        ),
    )
    return df.with_columns(
        is_breakout=(pl.col("close") > pl.col("prior_n_day_high")),
    )


def simulate_trades(ohlcv_with_signal: pl.DataFrame) -> pl.DataFrame:
    """Generate trade records by walking each symbol chronologically.

    Rule: enter at first breakout signal AFTER prior trade exited.
    Exit when close >= entry × (1 + TARGET_PCT) OR after HORIZON_TRADING_DAYS bars.
    """
    trades = []
    for sym in CRYPTO_UNIVERSE:
        sub = ohlcv_with_signal.filter(pl.col("symbol") == sym).sort("date")
        dates = sub["date"].to_list()
        closes = sub["close"].to_list()
        breakouts = sub["is_breakout"].to_list()

        i = 0
        n = len(sub)
        while i < n:
            if not breakouts[i]:
                i += 1
                continue
            # Entry at day i
            entry_date = dates[i]
            entry_price = closes[i]
            tp = entry_price * (1 + TARGET_PCT)
            exit_idx = None
            exit_reason = None
            for j in range(i + 1, min(i + 1 + HORIZON_TRADING_DAYS, n)):
                if closes[j] >= tp:
                    exit_idx = j
                    exit_reason = "tp"
                    break
            if exit_idx is None:
                exit_idx = min(i + HORIZON_TRADING_DAYS, n - 1)
                exit_reason = "horizon" if i + HORIZON_TRADING_DAYS < n else "data_end"

            exit_date = dates[exit_idx]
            exit_price = closes[exit_idx]
            gross_pnl_pct = (exit_price / entry_price - 1.0) * 100.0
            net_pnl_pct = gross_pnl_pct - FRICTION_RT * 100.0
            hold_days = exit_idx - i
            trades.append({
                "symbol": sym,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl_pct": gross_pnl_pct,
                "net_pnl_pct": net_pnl_pct,
                "hold_days": hold_days,
                "exit_reason": exit_reason,
            })
            i = exit_idx + 1  # next entry after this exit
    return pl.DataFrame(trades) if trades else pl.DataFrame(schema={
        "symbol": pl.String, "entry_date": pl.Date, "exit_date": pl.Date,
        "entry_price": pl.Float64, "exit_price": pl.Float64,
        "gross_pnl_pct": pl.Float64, "net_pnl_pct": pl.Float64,
        "hold_days": pl.Int64, "exit_reason": pl.String,
    })


def annualize_cagr(returns_pct: list[float], hold_days: list[int]) -> float:
    """Geometric mean per-trade return, annualized by mean hold."""
    if not returns_pct:
        return 0.0
    mults = [1 + r / 100.0 for r in returns_pct]
    geo_mean = np.exp(np.mean([np.log(m) for m in mults])) - 1.0
    mean_hold = max(np.mean(hold_days), 1.0)
    annual_periods = 252.0 / mean_hold
    return float((1 + geo_mean) ** annual_periods - 1.0)


def per_regime_breakdown(trades: pl.DataFrame, regime_labels: pl.DataFrame) -> pl.DataFrame:
    """Join trade entries to regime label on entry_date, group + report."""
    # The trade is "in" the regime that was active on its entry day
    joined = trades.join(
        regime_labels.select(["date", "regime_label"]).rename({"date": "entry_date"}),
        on="entry_date", how="left",
    )
    rows = []
    for regime in joined["regime_label"].unique().drop_nulls().to_list():
        sub = joined.filter(pl.col("regime_label") == regime)
        net_returns = sub["net_pnl_pct"].to_list()
        holds = sub["hold_days"].to_list()
        rows.append({
            "regime": regime,
            "n_trades": sub.height,
            "win_rate_pct": float((sub["net_pnl_pct"] > 0).sum() / sub.height * 100) if sub.height else 0.0,
            "mean_net_pnl_pct": float(np.mean(net_returns)) if net_returns else 0.0,
            "median_net_pnl_pct": float(np.median(net_returns)) if net_returns else 0.0,
            "annualized_cagr": annualize_cagr(net_returns, holds),
            "mean_hold_days": float(np.mean(holds)) if holds else 0.0,
        })
    return pl.DataFrame(rows).sort("regime")


def hold_btc_baseline(ohlcv: pl.DataFrame, regime_labels: pl.DataFrame, regime: str) -> dict:
    """Buy BTC at first day of `regime`, hold until exit of regime; aggregate."""
    btc = ohlcv.filter(pl.col("symbol") == "BTC-USD").sort("date").select(["date", "close"])
    rl = regime_labels.select(["date", "regime_label"])
    joined = btc.join(rl, on="date", how="inner").sort("date")
    # Build streaks
    in_streak = False
    streak_entry_price = None
    streak_entry_date = None
    streak_returns = []
    prev_regime = None
    for row in joined.iter_rows(named=True):
        r = row["regime_label"]
        if r == regime and not in_streak:
            in_streak = True
            streak_entry_price = row["close"]
            streak_entry_date = row["date"]
        elif r != regime and in_streak:
            # exit streak — record return
            exit_price = row["close"]
            streak_returns.append((exit_price / streak_entry_price - 1.0) * 100.0)
            in_streak = False
        prev_regime = r
    # Open streak at end
    if in_streak:
        last_close = joined["close"].to_list()[-1]
        streak_returns.append((last_close / streak_entry_price - 1.0) * 100.0)

    if not streak_returns:
        return {"n_streaks": 0, "mean_net_pnl_pct": 0.0, "annualized_cagr": 0.0}

    # Annualize: total compound return / total days
    mults = [1 + r / 100.0 for r in streak_returns]
    total_mult = float(np.prod(mults))
    total_days = (joined.filter(pl.col("regime_label") == regime).height)
    annual = total_mult ** (252.0 / max(total_days, 1)) - 1.0
    return {
        "n_streaks": len(streak_returns),
        "total_days_in_regime": total_days,
        "mean_net_pnl_pct": float(np.mean(streak_returns)),
        "total_compound_pct": (total_mult - 1.0) * 100.0,
        "annualized_cagr": float(annual),
    }


def main() -> None:
    print("=== P2 v0.4 — crypto bull-trend (Donchian 60d entry, 120d hold, +50% TP) ===\n")

    print("[1/5] fetching crypto OHLCV from sidecar...")
    ohlcv = fetch_crypto()
    print(f"      {ohlcv.height} rows × {ohlcv['symbol'].n_unique()} symbols")
    print(f"      date range: {ohlcv['date'].min()} -> {ohlcv['date'].max()}")

    print("\n[2/5] loading P1 regime labels (v0.4)...")
    regime_labels = pl.read_parquet(REGIME_LABELS_PATH)
    print(f"      {regime_labels.height} day-labels")

    print(f"\n[3/5] computing breakout entries (N={ENTRY_BREAKOUT_DAYS})...")
    ohlcv_sig = compute_breakout_entries(ohlcv)
    n_breakouts = int(ohlcv_sig["is_breakout"].fill_null(False).sum())
    print(f"      {n_breakouts} raw breakout days across universe (incl. clustered)")

    print(f"\n[4/5] simulating trades (TP=+{TARGET_PCT*100:.0f}%, max hold={HORIZON_TRADING_DAYS}d)...")
    trades = simulate_trades(ohlcv_sig)
    print(f"      {trades.height} trades executed")
    if trades.height == 0:
        print("      NO TRADES — strategy never fired. Check breakout calibration.")
        return

    # Overall stats
    print("\n--- overall ---")
    nr = trades["net_pnl_pct"].to_list()
    hd = trades["hold_days"].to_list()
    print(f"      mean net pnl:  {np.mean(nr):+.2f}%")
    print(f"      median net:    {np.median(nr):+.2f}%")
    print(f"      win rate:      {(np.array(nr) > 0).mean()*100:.1f}%")
    print(f"      mean hold:     {np.mean(hd):.1f} days")
    print(f"      annualized CAGR (per-trade-compound): {annualize_cagr(nr, hd)*100:+.2f}%")
    print(f"      exit reasons:")
    for er, cnt in trades.group_by("exit_reason").len().sort("len", descending=True).iter_rows(named=False):
        print(f"        {er}: {cnt}")

    print("\n[5/5] per-regime breakdown (using P1 v0.4 labels)...")
    pr = per_regime_breakdown(trades, regime_labels)
    for row in pr.iter_rows(named=True):
        print(f"      {row['regime']:25s} n={row['n_trades']:3d} "
              f"win={row['win_rate_pct']:5.1f}% "
              f"mean_pnl={row['mean_net_pnl_pct']:+6.2f}% "
              f"cagr={row['annualized_cagr']*100:+7.2f}% "
              f"hold={row['mean_hold_days']:5.1f}d")

    print("\n--- comparator: hold-BTC during steady_bull-only days ---")
    for regime in ["steady_bull", "choppy_recovery", "sideways_range", "bear_trend"]:
        b = hold_btc_baseline(ohlcv, regime_labels, regime)
        print(f"      {regime:25s} streaks={b.get('n_streaks'):2d} "
              f"days_in_regime={b.get('total_days_in_regime', 0):4d} "
              f"compound={b.get('total_compound_pct', 0):+7.2f}% "
              f"annualized={b.get('annualized_cagr', 0)*100:+7.2f}%")

    print("\n=== v0.4 acceptance check ===")
    sb = pr.filter(pl.col("regime") == "steady_bull")
    if sb.height > 0 and sb["annualized_cagr"][0] >= 0.10:
        print(f"      [PASS] steady_bull CAGR {sb['annualized_cagr'][0]*100:.2f}% >= 10% bar")
    elif sb.height > 0:
        print(f"      [FAIL] steady_bull CAGR {sb['annualized_cagr'][0]*100:.2f}% < 10% bar")
    else:
        print(f"      [FAIL] no steady_bull trades (label coverage gap)")


if __name__ == "__main__":
    main()
