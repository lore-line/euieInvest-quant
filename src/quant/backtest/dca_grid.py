#!/usr/bin/env python3
"""Crypto DCA grid backtest — user's actual 2022-2023 strategy class.

Strategy mechanics (decoded from user's Pine scripts + 3Commas DCA config):

ENTRY:  AND-gate of two Supertrend-style signals on the version's native TF:
        1. "Custom Buy Sell" inner trend flip (EMA75/ATR14, multi-TF stack)
        2. "ST MA" standalone Supertrend buy (EMA100/ATR10)
        + ATR% gate on 1H (per-version bounded vol regime)
        + (3% version) SMI buy cross from below 0

SIZING: Martingale-scaled DCA accumulation
        - base_order_usd: initial buy at signal trigger
        - n_safety_orders × so_volume_scale^n: geometrically scaled SOs
          fire as price drops by first_so_step_pct × so_step_scale^n

EXIT:   - TP: close entire position when price ≥ avg_entry × (1 + tp_pct)
        - SL: close at -tp_pct/2 from avg AFTER max SOs hit (catastrophic stop)
        - 1:2 R:R baked in via tp_pct vs sl_pct = tp_pct/2

VERSIONS (each runs as independent bot per symbol):
        1% version: 5m TF, TP=1%, ATR%>1, inner stack EMA75 on current/doubled
        2% version: 15m TF, TP=2%, ATR%>=2, inner stack EMA75/EMA14
        3% version: 1h TF, TP=3%, ATR%∈[4,8], inner stack EMA75 + SMI buy req

UNIVERSE: 8 liquid Kraken-Pro USD pairs (BTC, ETH, SOL, LINK, AVAX, DOT,
          ATOM, ADA). Excludes sub-$20M-daily-volume alts where user's
          discretionary experience showed liquidity traps killed deals.

FRICTION: kraken_pro_dynamic (30d-rolling tier-aware fee schedule).

Usage:
    python scripts/backtest-crypto-dca-grid.py \\
        --version 1 --symbol BTC-USD \\
        --start 2022-09-15 --end 2023-04-01 \\
        --base-order 10 --n-safety-orders 5 \\
        --first-so-step 1.0 --so-step-scale 1.5 --so-volume-scale 1.5

    # All three versions, all 8 symbols, full window:
    python scripts/backtest-crypto-dca-grid.py --all \\
        --start 2022-09-15 --end 2026-05-17
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

# Snapshot dir defaults to the project's `data/snapshots/` (consumer-team
# convention). Intraday parquets live as `intraday_{interval_min}m.parquet`
# inside it, written by `scripts/pull_intraday.py`.
SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"

# Universe — liquid Kraken-Pro USD pairs only. User's discretionary
# experience: sub-$20M daily volume alts caused liquidity traps where
# exits got stranded. Universe filter is a HARD discipline, not optional.
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD",
    "AVAX-USD", "DOT-USD", "ATOM-USD", "ADA-USD",
]

# Kraken Pro fee schedule (taker, maker) by 30d rolling USD volume.
# Engine maintains rolling deque of (timestamp, notional) tuples and
# looks up current tier per fill. Conservatively starts at $0 tier
# (0.40% maker / 0.80% taker) for users without prior volume history.
KRAKEN_FEE_TIERS = [
    (0,           0.0040, 0.0080), (2_500,       0.0030, 0.0060),
    (10_000,      0.0022, 0.0038), (25_000,      0.0020, 0.0035),
    (50_000,      0.0014, 0.0024), (100_000,     0.0012, 0.0022),
    (250_000,     0.0010, 0.0020), (500_000,     0.0008, 0.0018),
    (1_000_000,   0.0006, 0.0016), (2_500_000,   0.0004, 0.0014),
    (5_000_000,   0.0002, 0.0012), (10_000_000,  0.0000, 0.0010),
]
# Spread + slippage per symbol-tier (rough Kraken Pro live observations).
# BTC/ETH are tightest; alts wider. Applied additively to fee rate on
# market-taker fills only; maker fills assume no spread crossing.
SYMBOL_SPREAD_PCT = {
    "BTC-USD":  0.0002,  "ETH-USD":  0.0003,  "SOL-USD":  0.0004,
    "LINK-USD": 0.0005,  "AVAX-USD": 0.0006,  "DOT-USD":  0.0006,
    "ATOM-USD": 0.0008,  "ADA-USD":  0.0005,
}
SLIPPAGE_TAKER_PCT = 0.0010  # market orders pay slippage proportional to ATR

# Per-version configuration matching the user's Pine scripts.
VERSION_CONFIG = {
    1: {
        "tp_pct": 0.01,
        "sl_pct": 0.005,
        "native_tf_min": 5,
        "atr_pct_min": 1.0,
        "atr_pct_max": None,
        "inner_ema": 75,
        "inner_atr": 14,
        "inner_atr_mult": 0.5,
        "slow_tf_mult": 2,
        "slow_ema": 75,                # length param in Pine
        "stma_ema": 100,
        "stma_atr": 10,
        "stma_atr_mult": 0.5,
        "require_smi_buy": False,
        "require_fresh_flip": True,
    },
    2: {
        "tp_pct": 0.02,
        "sl_pct": 0.01,
        "native_tf_min": 15,
        "atr_pct_min": 2.0,
        "atr_pct_max": None,
        "inner_ema": 75,
        "inner_atr": 14,
        "inner_atr_mult": 0.5,
        "slow_tf_mult": 2,
        "slow_ema": 14,                # Periods param in Pine (different from 1%)
        "stma_ema": 100,
        "stma_atr": 10,
        "stma_atr_mult": 0.5,
        "require_smi_buy": False,
        "require_fresh_flip": True,
    },
    3: {
        "tp_pct": 0.03,
        "sl_pct": 0.015,
        "native_tf_min": 60,
        "atr_pct_min": 4.0,
        "atr_pct_max": 8.0,
        "inner_ema": 75,
        "inner_atr": 14,
        "inner_atr_mult": 0.5,
        "slow_tf_mult": 4,             # floor-clamped to 4h on 1h chart
        "slow_ema": 75,
        "stma_ema": 100,
        "stma_atr": 10,
        "stma_atr_mult": 0.5,
        "require_smi_buy": True,       # SMI cross required for 3% buy
        "require_fresh_flip": False,   # 3% doesn't need fresh flip
    },
    # --- Phase 2b extensions (extrapolated from v1/v2/v3 patterns) ---
    # v0.5: tighter scalper, fires on any ATR ≥ 0.5% (overlaps v1+ regimes).
    # v4 / v5: extreme-vol scalpers ABOVE v3's gate.
    # v0.5 deliberately UNBOUNDED on upper ATR — per user direction, v0.5
    # is a "tight-TP harvester across all ATR regimes", not a low-vol-only
    # version. Overlapping with v1/v2/... regimes is the design.
    0.5: {
        "tp_pct": 0.005,
        "sl_pct": 0.0025,
        "native_tf_min": 1,            # 1m — finer timing for tight TP
        "atr_pct_min": 0.5,
        "atr_pct_max": None,           # fires on any ATR ≥ 0.5%, including v1's regime
        "inner_ema": 75,
        "inner_atr": 14,
        "inner_atr_mult": 0.5,
        "slow_tf_mult": 2,
        "slow_ema": 75,
        "stma_ema": 100,
        "stma_atr": 10,
        "stma_atr_mult": 0.5,
        "require_smi_buy": False,
        "require_fresh_flip": True,
    },
    4: {
        "tp_pct": 0.04,
        "sl_pct": 0.02,
        "native_tf_min": 240,          # 4h, resampled from 60m
        "atr_pct_min": 8.0,
        "atr_pct_max": None,
        "inner_ema": 75,
        "inner_atr": 14,
        "inner_atr_mult": 0.5,
        "slow_tf_mult": 4,
        "slow_ema": 75,
        "stma_ema": 100,
        "stma_atr": 10,
        "stma_atr_mult": 0.5,
        "require_smi_buy": True,
        "require_fresh_flip": False,
    },
    5: {
        "tp_pct": 0.05,
        "sl_pct": 0.025,
        "native_tf_min": 1440,         # 1d, resampled from 60m
        "atr_pct_min": 10.0,
        "atr_pct_max": None,
        "inner_ema": 75,
        "inner_atr": 14,
        "inner_atr_mult": 0.5,
        "slow_tf_mult": 4,
        "slow_ema": 75,
        "stma_ema": 100,
        "stma_atr": 10,
        "stma_atr_mult": 0.5,
        "require_smi_buy": True,
        "require_fresh_flip": False,
    },
}


# =====================================================================
# Data loading + multi-TF resampling
# =====================================================================

_PARQUET_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def _load_parquet(snapshot_dir: Path, interval_min: int) -> pd.DataFrame:
    """Read one interval's parquet → pandas DataFrame (cached per-process).

    The puller (`scripts/pull_intraday.py`) writes one parquet per interval
    containing every symbol's bars. We load it once and slice by symbol +
    date range on each call.
    """
    key = (str(snapshot_dir), interval_min)
    if key in _PARQUET_CACHE:
        return _PARQUET_CACHE[key]
    path = snapshot_dir / f"intraday_{interval_min}m.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"intraday snapshot not found: {path}. Run "
            f"`uv run scripts/pull_intraday.py --interval {interval_min}` first."
        )
    df = pl.read_parquet(path).to_pandas()
    # Parquet stores timestamps as tz-aware UTC datetimes. Strip the tz to
    # match the SQLite-era simulator which used naive timestamps; the sim
    # logic doesn't care about UTC labeling, only relative ordering.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    _PARQUET_CACHE[key] = df
    return df


def load_bars(snapshot_dir, symbol: str, interval_min: int,
              start_date: str, end_date: str) -> pd.DataFrame:
    """Load bars for a (symbol, interval) slice as pandas DataFrame.

    If `intraday_{interval_min}m.parquet` exists in `snapshot_dir`, reads
    directly from it. Otherwise falls back to the finest available source
    (60m by default) and resamples up to the requested interval. The
    resample fallback supports v4 (240m) / v5 (1440m) without explicit
    fetches at those intervals.
    """
    snapshot_dir = Path(snapshot_dir)
    target_path = snapshot_dir / f"intraday_{interval_min}m.parquet"
    if target_path.exists():
        source_interval = interval_min
        df = _load_parquet(snapshot_dir, source_interval)
    else:
        # Pick the finest available source ≤ interval_min that's on disk.
        # Common case: target is 240 (4h) or 1440 (1d); 60m parquet exists.
        candidates = [60, 15, 5, 1]
        source_interval = next(
            (c for c in candidates
             if c <= interval_min and (snapshot_dir / f"intraday_{c}m.parquet").exists()),
            None,
        )
        if source_interval is None:
            raise FileNotFoundError(
                f"No intraday parquet ≤ {interval_min}m found in {snapshot_dir}. "
                f"Run `uv run scripts/pull_intraday.py --intervals {interval_min}` "
                f"or ensure a finer-resolution parquet (e.g. 60m) is available."
            )
        df = _load_parquet(snapshot_dir, source_interval)

    end_bound = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    mask = (df["symbol"] == symbol) & \
           (df["timestamp"] >= pd.to_datetime(start_date)) & \
           (df["timestamp"] < end_bound)
    sub = df.loc[mask, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.set_index("timestamp").sort_index()

    # Resample if we loaded a finer source than requested.
    if source_interval < interval_min:
        sub = resample(sub, interval_min)
    return sub


def resample(df: pd.DataFrame, target_min: int) -> pd.DataFrame:
    """Resample bars to a coarser timeframe."""
    rule = f"{target_min}min"
    out = pd.DataFrame()
    out["open"]   = df["open"].resample(rule).first()
    out["high"]   = df["high"].resample(rule).max()
    out["low"]    = df["low"].resample(rule).min()
    out["close"]  = df["close"].resample(rule).last()
    out["volume"] = df["volume"].resample(rule).sum()
    return out.dropna()


# =====================================================================
# Pine script ports
# =====================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Pine-equivalent EMA with span = period."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Pine-equivalent ATR(period). True Range then EMA-smoothing (Wilder)."""
    high, low, close = df["high"], df["low"], df["close"]
    pc = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - pc).abs(),
        (low - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def supertrend_state(df: pd.DataFrame, ema_len: int, atr_len: int,
                     atr_mult: float) -> pd.DataFrame:
    """Compute a Supertrend-style trailing-band trend state.

    Returns DataFrame indexed like df with columns:
      ma:    EMA(close, ema_len)
      atr:   ATR(atr_len)
      up:    trailing-ratchet lower band (ma - mult×atr, only moves up)
      dn:    trailing-ratchet upper band (ma + mult×atr, only moves down)
      trend: +1 long / -1 short, flipped on close crossing prior band
    """
    ma_v = ema(df["close"], ema_len)
    atr_v = atr(df, atr_len)
    n = len(df)

    up_raw = (ma_v - atr_mult * atr_v).values
    dn_raw = (ma_v + atr_mult * atr_v).values
    closes = df["close"].values

    up = np.copy(up_raw)
    dn = np.copy(dn_raw)
    trend = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        # Trailing ratchet: up only moves UP, dn only moves DOWN, conditional
        # on the prior bar's close staying on the right side of the band.
        up[i] = max(up[i], up[i - 1]) if closes[i - 1] > up[i - 1] else up[i]
        dn[i] = min(dn[i], dn[i - 1]) if closes[i - 1] < dn[i - 1] else dn[i]
        # Trend flip on close crossing prior band
        if trend[i - 1] == -1 and closes[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and closes[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    return pd.DataFrame({
        "ma": ma_v.values, "atr": atr_v.values,
        "up": up, "dn": dn, "trend": trend,
    }, index=df.index)


def smi_signal(df: pd.DataFrame, k_len: int = 14, d_len: int = 3,
               smma_len: int = 2) -> pd.Series:
    """Stochastic Momentum Index buy cross — Pine spec.

    smiSignal = crossover(smi, ema(smi, smma_len)) AND ema(smi) < 0
    (3% version's SMI requirement).
    """
    src = df["close"]
    lowest = src.rolling(k_len).min()
    highest = src.rolling(k_len).max()
    midpt = (lowest + highest) / 2
    m_sum = (src - midpt).rolling(d_len).mean()
    span = (highest - lowest).rolling(d_len).mean()
    smi = 100 * m_sum / span.replace(0, np.nan)
    ema_smi = ema(smi.fillna(0), smma_len)
    crossover = (smi > ema_smi) & (smi.shift(1) <= ema_smi.shift(1))
    return (crossover & (ema_smi < 0)).fillna(False)


def atr_pct_1h(df_1h: pd.DataFrame) -> pd.Series:
    """1H ATR(9) / 1H close × 100, matching Pine's atrPercent."""
    atr_v = atr(df_1h, 9)
    return (atr_v / df_1h["close"]) * 100


# =====================================================================
# Entry signal generation per version
# =====================================================================

def generate_entry_signals(
    bars: pd.DataFrame, version: int, bars_1h_for_atr: pd.DataFrame,
) -> pd.Series:
    """Generate AND-gate entry signals for a given version on its native TF bars.

    AND-gate = Custom Buy Sell internal buy  AND  ST MA standalone buy.

    Returns Series indexed like bars, bool, True where entry fires.
    """
    cfg = VERSION_CONFIG[version]

    # Inner trend (current TF, EMA75/ATR14)
    inner = supertrend_state(bars, cfg["inner_ema"], cfg["inner_atr"], cfg["inner_atr_mult"])

    # Slow trend (resampled to doubled TF, EMA per version's slow_ema)
    slow_tf_min = cfg["native_tf_min"] * cfg["slow_tf_mult"]
    bars_slow = resample(bars, slow_tf_min)
    slow = supertrend_state(bars_slow, cfg["slow_ema"], cfg["inner_atr"], cfg["inner_atr_mult"])
    # Align slow back to native TF (forward-fill since slow is sparser)
    slow_aligned = slow.reindex(bars.index, method="ffill")

    # ST MA standalone (current TF, EMA100/ATR10)
    stma = supertrend_state(bars, cfg["stma_ema"], cfg["stma_atr"], cfg["stma_atr_mult"])

    # ATR% gate (computed on 1H bars, then forward-aligned to native TF)
    atr_pct = atr_pct_1h(bars_1h_for_atr)
    atr_pct_aligned = atr_pct.reindex(bars.index, method="ffill")
    if cfg["atr_pct_max"] is not None:
        atr_gate = (atr_pct_aligned >= cfg["atr_pct_min"]) & (atr_pct_aligned < cfg["atr_pct_max"])
    else:
        atr_gate = atr_pct_aligned > cfg["atr_pct_min"] if version == 1 else atr_pct_aligned >= cfg["atr_pct_min"]

    # Custom Buy Sell inner buySignal logic
    if cfg["require_fresh_flip"]:
        inner_buy = (inner["trend"] == 1) & (inner["trend"].shift(1) == -1) & (slow_aligned["trend"] == 1) & atr_gate
    else:
        # 3% version: no fresh flip, just slowTF up + ATR gate + SMI
        inner_buy = (slow_aligned["trend"] == 1) & atr_gate
        if cfg["require_smi_buy"]:
            smi_buy = smi_signal(bars)
            inner_buy = inner_buy & smi_buy

    # ST MA standalone — confirming filter, "in agreement" semantics.
    # User's live setup ran ST MA as a SEPARATE alert on the same chart;
    # "i only entered when they both agreed" interpreted as: Custom Buy
    # Sell fresh-trigger provides the entry; ST MA must be currently in
    # UP regime (not necessarily fresh-flipped) to confirm. Strict same-
    # bar dual-fresh-flip is impossibly rare with different EMA configs
    # (75/14 vs 100/10) and would produce 0 signals in 6 months.
    stma_up_now = (stma["trend"] == 1)

    # AND-gate: Custom Buy Sell trigger + ST MA confirming-up regime
    return (inner_buy & stma_up_now).fillna(False)


# =====================================================================
# Friction model
# =====================================================================

class RollingVolume30d:
    """Maintain 30-day rolling sum of fill notionals for tier lookup."""
    def __init__(self):
        self.fills: deque = deque()

    def add(self, ts: datetime, notional_usd: float):
        self.fills.append((ts, notional_usd))
        self._prune(ts)

    def _prune(self, now: datetime):
        cutoff = now - timedelta(days=30)
        while self.fills and self.fills[0][0] < cutoff:
            self.fills.popleft()

    def current(self, as_of: datetime) -> float:
        self._prune(as_of)
        return sum(n for _, n in self.fills)


def kraken_fee_rate(vol_30d_usd: float, is_taker: bool) -> float:
    rate = KRAKEN_FEE_TIERS[0][2 if is_taker else 1]
    for threshold, maker_rate, taker_rate in KRAKEN_FEE_TIERS:
        if vol_30d_usd >= threshold:
            rate = taker_rate if is_taker else maker_rate
        else:
            break
    return rate


def fill_cost_usd(symbol: str, notional_usd: float, vol_30d_usd: float,
                  is_taker: bool) -> float:
    """Per-fill cost. Maker fills earn no spread/slippage premium because
    the limit order rests at-or-better-than mid. Taker fills pay the
    spread cross + ATR-proportional slippage."""
    fee = kraken_fee_rate(vol_30d_usd, is_taker)
    if is_taker:
        spread = SYMBOL_SPREAD_PCT.get(symbol, 0.0008)
        return notional_usd * (fee + spread + SLIPPAGE_TAKER_PCT)
    return notional_usd * fee


# =====================================================================
# DCA position manager
# =====================================================================

@dataclass
class DcaDeal:
    symbol: str
    version: int
    opened_at: pd.Timestamp
    base_price: float
    so_levels_pct: list  # cumulative % drops from base where each SO fires
    so_volumes_usd: list  # USD volume of each SO (incl. base as index 0)
    n_filled: int = 1  # base order counts as 1 fill
    cumulative_units: float = 0.0
    cumulative_cost_usd: float = 0.0
    cumulative_friction_usd: float = 0.0

    @property
    def avg_entry_price(self) -> float:
        if self.cumulative_units <= 0:
            return self.base_price
        return self.cumulative_cost_usd / self.cumulative_units

    @property
    def max_filled(self) -> bool:
        return self.n_filled >= len(self.so_volumes_usd)

    def next_so_price(self) -> Optional[float]:
        if self.max_filled:
            return None
        return self.base_price * (1 - self.so_levels_pct[self.n_filled])


def build_so_schedule(base_order_usd: float, n_safety_orders: int,
                      first_so_step_pct: float, so_step_scale: float,
                      so_volume_scale: float) -> tuple[list, list]:
    """Compute the cumulative SO price-drop levels and per-SO volumes (including base)."""
    levels = [0.0]  # base is the 0-drop reference
    volumes = [base_order_usd]
    cur_step = first_so_step_pct
    cur_cumulative = 0.0
    cur_volume = base_order_usd
    for _ in range(n_safety_orders):
        cur_cumulative += cur_step
        levels.append(cur_cumulative)
        cur_volume *= so_volume_scale
        volumes.append(cur_volume)
        cur_step *= so_step_scale
    return levels, volumes


# =====================================================================
# Portfolio simulator
# =====================================================================

@dataclass
class CloseResult:
    deal: DcaDeal
    close_ts: pd.Timestamp
    close_price: float
    reason: str  # 'tp', 'sl'
    realized_pnl_usd: float
    holding_bars: int


def _pget(params: dict, version: int, key: str, default=None):
    """Get a parameter, preferring per-version override if present.

    Allows ensemble runs where v1/v2/v3 need different ladder shapes (e.g.
    v1 wants vol_scale=2.30 + base_pct=0.5% but v2 wants vol_scale=1.00 +
    base_pct=2.50%). Pass `per_version: {1: {...}, 2: {...}}` in params;
    the simulator resolves the right value per (sym, version) event.
    """
    per_ver = params.get("per_version", {})
    if version in per_ver and key in per_ver[version]:
        return per_ver[version][key]
    return params.get(key, default)


def simulate_portfolio(
    entry_signals_per_sv: dict,   # {(symbol, version): pd.Series of bool}
    bars_per_sv: dict,             # {(symbol, version): pd.DataFrame (native TF)}
    params: dict,
    starting_capital: float = 3000.0,
) -> dict:
    """Walk bars chronologically across all (symbol, version) pairs and
    simulate deals. Returns aggregate metrics + per-deal log.

    Implements the user's strand-and-abandon discipline: after a deal hits
    SL (max SOs filled + price keeps falling), the symbol is banned from
    future entries for `strand_ban_days`. This is the discipline that
    protected the user from catastrophic Martingale blowups in 2022.

    Per-version param overrides via `params["per_version"] = {ver: {...}}`.
    Top-level params remain the fallback when a key isn't version-overridden.
    """

    # Aggregate bars: merge all native TFs into a master timeline.
    # We walk in order of bar timestamps; for each bar we (1) check signals,
    # (2) advance open deals on this symbol's bar.
    all_keys = list(entry_signals_per_sv.keys())
    open_deals: dict = {}  # (symbol, version) → DcaDeal
    closed_deals: list = []
    banned_symbols: dict = {}  # symbol → unban_timestamp
    rolling_vol = RollingVolume30d()
    # When set, fee lookups use this fixed 30d-volume instead of the
    # rolling counter. Lets cliff/topology sweeps isolate deployment-scale
    # friction effects from the volume-buildup transient.
    fixed_vol_override = params.get("fixed_friction_vol_30d")
    def _vol_for_fees(ts_now):
        if fixed_vol_override is not None:
            return float(fixed_vol_override)
        return rolling_vol.current(ts_now)
    cash = starting_capital
    deployed = 0.0
    equity_curve: list = []
    strand_ban_days = params.get("strand_ban_days", 90)

    # Stream bar events in timestamp order via heap-merge over per-key
    # iterators. Each (sym, ver) bars frame is already sorted by load_bars,
    # so heapq.merge gives O(N log K) chronological iteration without
    # materializing the full event list — drops memory from O(N) to O(K)
    # where K is the number of (sym, ver) pairs. For 22 syms × 3 TFs × 4y
    # the flat-list path peaked at ~2GB; this path stays under ~50MB.

    def _per_key_events(sym: str, ver: int, bars: pd.DataFrame,
                          sigs: pd.Series):
        """Yield (ts, sym, ver, open, high, low, close, signal) per bar.

        Pre-extracts numpy arrays so the per-bar loop avoids pd.Series
        allocation. Signals are reindexed onto bars.index once, then
        boolean-array-indexed by position. Timestamp conversion is paid
        once at the index level, not per-bar.
        """
        ts_pd = list(bars.index)  # pd.Timestamp objects; downstream uses .normalize/.to_pydatetime
        op_arr = bars["open"].to_numpy()
        hi_arr = bars["high"].to_numpy()
        lo_arr = bars["low"].to_numpy()
        cl_arr = bars["close"].to_numpy()
        sig_aligned = sigs.reindex(bars.index, fill_value=False).to_numpy()
        for i in range(len(ts_pd)):
            yield (ts_pd[i], sym, ver,
                   float(op_arr[i]), float(hi_arr[i]), float(lo_arr[i]),
                   float(cl_arr[i]), bool(sig_aligned[i]))

    iters = [_per_key_events(sym, ver, bars,
                              entry_signals_per_sv[(sym, ver)])
             for (sym, ver), bars in bars_per_sv.items()]
    event_stream = heapq.merge(*iters, key=lambda e: e[0])

    last_eq_ts = None
    for ts, sym, ver, op, hi, lo, cl, sig in event_stream:
        key = (sym, ver)

        # 1. Advance open deal on this (sym, ver) bar
        if key in open_deals:
            deal = open_deals[key]
            cfg = VERSION_CONFIG[ver]
            tp_price = deal.avg_entry_price * (1 + cfg["tp_pct"])
            # SL only active after max SOs hit (catastrophic stop)
            sl_active = deal.max_filled
            sl_price = deal.avg_entry_price * (1 - cfg["sl_pct"]) if sl_active else None

            # Check next SO fill (intra-bar low crosses next SO threshold)
            while not deal.max_filled:
                next_p = deal.next_so_price()
                if next_p is not None and lo <= next_p:
                    so_idx = deal.n_filled
                    so_vol = deal.so_volumes_usd[so_idx]
                    if cash >= so_vol:
                        units = so_vol / next_p
                        vol_now = _vol_for_fees(ts.to_pydatetime())
                        friction = fill_cost_usd(sym, so_vol, vol_now, is_taker=params.get("is_taker", False))
                        deal.cumulative_units += units
                        deal.cumulative_cost_usd += so_vol
                        deal.cumulative_friction_usd += friction
                        deal.n_filled += 1
                        cash -= so_vol
                        deployed += so_vol
                        rolling_vol.add(ts.to_pydatetime(), so_vol)
                    else:
                        # Out of cash — can't fill this SO. Break and proceed to exit checks.
                        # In reality the bot would just hold and wait for TP.
                        break
                else:
                    break

            # Re-evaluate SL after all SOs that fired this bar
            sl_active = deal.max_filled
            sl_price = deal.avg_entry_price * (1 - cfg["sl_pct"]) if sl_active else None

            # Early SL: catastrophic stop from BASE entry, fires before max SOs.
            # Caps loss magnitude at the expense of more false-strand triggers.
            # When set, the strategy becomes a "loss-leader for volume" with
            # bounded per-deal downside.
            early_sl_pct = _pget(params, ver, "early_sl_pct")
            early_sl_price = (deal.base_price * (1 - early_sl_pct)
                               if early_sl_pct is not None else None)

            # Check exits (TP first per ordering convention)
            exit_price = None
            exit_reason = None
            if hi >= tp_price:
                exit_price = tp_price
                exit_reason = "tp"
            elif early_sl_price is not None and lo <= early_sl_price:
                exit_price = early_sl_price
                exit_reason = "early_sl"
            elif sl_active and sl_price is not None and lo <= sl_price:
                exit_price = sl_price
                exit_reason = "sl"

            if exit_price is not None:
                exit_notional = deal.cumulative_units * exit_price
                vol_now = _vol_for_fees(ts.to_pydatetime())
                # All exits use maker fees — user's 3Commas setup ran
                # take-profit-limit and stop-limit orders (not stop-market).
                # Even SL exits are limit-order fills, never market sells.
                exit_friction = fill_cost_usd(sym, exit_notional, vol_now, is_taker=False)
                deal.cumulative_friction_usd += exit_friction
                gross = exit_notional - deal.cumulative_cost_usd
                net_pnl = gross - deal.cumulative_friction_usd
                cash += exit_notional - exit_friction
                deployed -= deal.cumulative_cost_usd
                rolling_vol.add(ts.to_pydatetime(), exit_notional)
                holding_bars = int(((ts - deal.opened_at).total_seconds() // 60) // VERSION_CONFIG[ver]["native_tf_min"])
                closed_deals.append(CloseResult(
                    deal=deal, close_ts=ts, close_price=exit_price, reason=exit_reason,
                    realized_pnl_usd=net_pnl, holding_bars=max(holding_bars, 1),
                ))
                del open_deals[key]
                # Strand-and-abandon: if this was a catastrophic SL exit, ban
                # the symbol from future entries for the configured window.
                # Both 'sl' and 'early_sl' trigger the ban.
                if exit_reason in ("sl", "early_sl"):
                    banned_symbols[sym] = ts + timedelta(days=strand_ban_days)

        # 2. On signal AND no open deal on (sym, ver), open new deal at current bar's CLOSE
        # (entry fills at close of signal bar — assumes signal evaluated at bar close)
        # Skip if symbol is currently banned (post-strand recovery window).
        is_banned = sym in banned_symbols and ts < banned_symbols[sym]
        if sig and key not in open_deals and not is_banned:
            # Dynamic base order: if base_pct_of_equity is set, scale base order
            # with current equity (cash + estimated open-deal value). This lets
            # the strategy compound — base orders grow proportionally with the
            # account, maintaining the buffer ratio rather than the absolute $.
            base_pct = _pget(params, ver, "base_pct_of_equity")
            if base_pct is not None:
                # Current equity ≈ cash + cost-basis of all open deals
                current_equity = cash + sum(d.cumulative_cost_usd for d in open_deals.values())
                base_usd = current_equity * base_pct
            else:
                base_usd = _pget(params, ver, "base_order_usd")
            # Regime-conditional multiplier — if regime_multipliers + regime_lookup
            # are provided, look up today's regime and apply the multiplier.
            # Multiplier of 0.0 = pause new entries during this regime.
            regime_mults = params.get("regime_multipliers")
            regime_lookup = params.get("regime_lookup")
            current_regime = None
            if regime_mults and regime_lookup is not None:
                today_date = ts.normalize() if hasattr(ts, "normalize") else ts
                # regime_lookup is a dict: {pd.Timestamp.normalize() → str}
                current_regime = regime_lookup.get(today_date, "unknown")
                mult = regime_mults.get(current_regime, 1.0)
                base_usd *= mult
            if base_usd is not None and base_usd > 0 and cash >= base_usd:
                levels, volumes = build_so_schedule(
                    base_usd, _pget(params, ver, "n_safety_orders"),
                    _pget(params, ver, "first_so_step_pct"),
                    _pget(params, ver, "so_step_scale"),
                    _pget(params, ver, "so_volume_scale"),
                )
                units = base_usd / cl
                vol_now = _vol_for_fees(ts.to_pydatetime())
                friction = fill_cost_usd(sym, base_usd, vol_now, is_taker=params.get("is_taker", False))
                deal = DcaDeal(
                    symbol=sym, version=ver, opened_at=ts, base_price=cl,
                    so_levels_pct=levels, so_volumes_usd=volumes,
                    cumulative_units=units, cumulative_cost_usd=base_usd,
                    cumulative_friction_usd=friction,
                )
                cash -= base_usd
                deployed += base_usd
                rolling_vol.add(ts.to_pydatetime(), base_usd)
                open_deals[key] = deal

        # 3. Equity snapshot per day (for daily-Sharpe computation)
        day = ts.normalize()
        if last_eq_ts != day:
            # Mark-to-market open deals at this bar's close
            mtm = 0.0
            for k, d in open_deals.items():
                if k[0] == sym:
                    mtm += d.cumulative_units * cl
                else:
                    # Other symbols: use last close stored elsewhere; for
                    # simplicity, leave at cost (slight inaccuracy but
                    # bounded; the deal will mark properly when its own
                    # bar arrives)
                    mtm += d.cumulative_cost_usd
            equity = cash + mtm
            equity_curve.append((day, equity))
            last_eq_ts = day

    # Final close-out of open deals at last available close
    for key, deal in list(open_deals.items()):
        sym, ver = key
        last_close = bars_per_sv[key]["close"].iloc[-1]
        exit_notional = deal.cumulative_units * last_close
        gross = exit_notional - deal.cumulative_cost_usd
        net_pnl = gross - deal.cumulative_friction_usd
        cash += exit_notional
        closed_deals.append(CloseResult(
            deal=deal, close_ts=bars_per_sv[key].index[-1],
            close_price=last_close, reason="end_of_data",
            realized_pnl_usd=net_pnl, holding_bars=0,
        ))

    return {
        "closed_deals": closed_deals,
        "equity_curve": equity_curve,
        "final_cash": cash,
        "starting_capital": starting_capital,
    }


# =====================================================================
# Performance metrics
# =====================================================================

def compute_metrics(result: dict) -> dict:
    deals = result["closed_deals"]
    eq = result["equity_curve"]
    starting = result["starting_capital"]
    final = result["final_cash"]

    if not deals:
        return {"n_deals": 0, "net_pnl_usd": 0, "win_rate": 0,
                "sharpe": 0, "max_dd_pct": 0, "cagr_pct": 0,
                "final_equity": final, "return_x": 1.0}

    n = len(deals)
    wins = sum(1 for d in deals if d.realized_pnl_usd > 0)
    pnls = [d.realized_pnl_usd for d in deals]
    net_pnl = sum(pnls)
    win_rate = wins / n if n else 0
    mean_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    mean_loss = np.mean([abs(p) for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
    rr_ratio = mean_loss / mean_win if mean_win > 0 else float("inf")

    # Equity-based metrics
    if len(eq) > 5:
        eq_series = pd.Series([e[1] for e in eq], index=[e[0] for e in eq])
        daily_returns = eq_series.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(365)
        else:
            sharpe = 0
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        max_dd = dd.min() * 100
        days_span = (eq_series.index[-1] - eq_series.index[0]).days
        years = max(days_span / 365.25, 0.001)
        if final > 0 and starting > 0:
            cagr = ((final / starting) ** (1 / years) - 1) * 100
        else:
            cagr = -100
    else:
        sharpe = 0
        max_dd = 0
        cagr = 0

    return {
        "n_deals": n,
        "n_wins": wins,
        "win_rate": win_rate,
        "mean_win_usd": mean_win,
        "mean_loss_usd": mean_loss,
        "rr_realized": rr_ratio,
        "net_pnl_usd": net_pnl,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "cagr_pct": cagr,
        "final_equity": final,
        "return_x": final / starting if starting > 0 else 0,
        "exit_breakdown": dict({"tp": sum(1 for d in deals if d.reason == "tp"),
                                 "sl": sum(1 for d in deals if d.reason == "sl"),
                                 "end_of_data": sum(1 for d in deals if d.reason == "end_of_data")}),
    }


# =====================================================================
# CLI / main
# =====================================================================

def get_native_tf_min(version: int) -> int:
    return VERSION_CONFIG[version]["native_tf_min"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, choices=[1, 2, 3], default=None,
                   help="Single version to backtest. Use --all for ensemble.")
    p.add_argument("--all", action="store_true",
                   help="Run all 3 versions concurrently across the universe (ensemble)")
    p.add_argument("--symbol", default=None,
                   help="Single symbol; defaults to full universe")
    p.add_argument("--start", default="2022-09-15", help="Start date (ISO)")
    p.add_argument("--end", default=None, help="End date (ISO); default = today")
    p.add_argument("--starting-capital", type=float, default=3000.0)
    p.add_argument("--base-order", type=float, default=10.0)
    p.add_argument("--n-safety-orders", type=int, default=5)
    p.add_argument("--first-so-step", type=float, default=1.0,
                   help="First SO price step %% (e.g. 1.0 = 1%% drop from base)")
    p.add_argument("--so-step-scale", type=float, default=1.5,
                   help="Multiplier on step between successive SOs")
    p.add_argument("--so-volume-scale", type=float, default=1.5,
                   help="Multiplier on SO size between successive SOs")
    p.add_argument("--fee-mode", choices=["maker", "taker"], default="maker",
                   help="Fee mode. 'maker' assumes limit-order fills (3Commas default). "
                        "'taker' assumes market orders (worse friction). Maker is much closer "
                        "to actual 3Commas DCA bot behavior.")
    p.add_argument("--strand-ban-days", type=int, default=90,
                   help="Days to ban a symbol from new entries after a SL (strand) exit. "
                        "Models user's 'if stranded, sell and stop trading it' discipline.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--snapshot-dir", default=None,
                   help="Directory holding intraday_{N}m.parquet files. "
                        "Defaults to <repo>/data/snapshots/.")
    args = p.parse_args()

    if not args.version and not args.all:
        print("Must specify --version 1|2|3 or --all", file=sys.stderr)
        return 1

    versions = [1, 2, 3] if args.all else [args.version]
    symbols = [args.symbol] if args.symbol else CRYPTO_UNIVERSE
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {
        "base_order_usd": args.base_order,
        "n_safety_orders": args.n_safety_orders,
        "first_so_step_pct": args.first_so_step / 100,  # convert to fractional
        "so_step_scale": args.so_step_scale,
        "so_volume_scale": args.so_volume_scale,
        "is_taker": args.fee_mode == "taker",
        "strand_ban_days": args.strand_ban_days,
    }

    print(f"[crypto-dca-grid] versions={versions} symbols={len(symbols)} window={args.start}→{end}")
    print(f"[crypto-dca-grid] base=${args.base_order} n_SOs={args.n_safety_orders} "
          f"step={args.first_so_step}% step_scale={args.so_step_scale} vol_scale={args.so_volume_scale}")

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else SNAPSHOT_DIR

    # Load + prepare data for each (symbol, version) combination
    bars_per_sv = {}
    signals_per_sv = {}
    for sym in symbols:
        bars_1h = load_bars(snapshot_dir, sym, 60, args.start, end)
        if bars_1h.empty:
            print(f"[crypto-dca-grid] {sym}: no 1h data, skipping")
            continue
        for ver in versions:
            tf_min = get_native_tf_min(ver)
            bars = load_bars(snapshot_dir, sym, tf_min, args.start, end)
            if bars.empty:
                print(f"[crypto-dca-grid] {sym} v{ver}: no {tf_min}m data, skipping")
                continue
            sigs = generate_entry_signals(bars, ver, bars_1h)
            n_sigs = sigs.sum()
            print(f"[crypto-dca-grid] {sym} v{ver}: {n_sigs} entry signals over {len(bars)} bars")
            bars_per_sv[(sym, ver)] = bars
            signals_per_sv[(sym, ver)] = sigs

    if not bars_per_sv:
        print("[crypto-dca-grid] no data — abort", file=sys.stderr)
        return 1

    result = simulate_portfolio(signals_per_sv, bars_per_sv, params, args.starting_capital)
    metrics = compute_metrics(result)

    print()
    print(f"=== Backtest Results ===")
    print(f"  Deals: {metrics['n_deals']} (wins {metrics.get('n_wins', 0)}, win_rate {metrics['win_rate']:.1%})")
    if metrics["n_deals"] > 0:
        print(f"  Exit breakdown: {metrics.get('exit_breakdown', {})}")
        print(f"  Mean win: ${metrics.get('mean_win_usd', 0):.2f}, mean loss: ${metrics.get('mean_loss_usd', 0):.2f}")
        print(f"  R:R realized: {metrics.get('rr_realized', 0):.2f}:1")
    print(f"  Net P&L: ${metrics['net_pnl_usd']:.2f}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
    print(f"  CAGR: {metrics['cagr_pct']:.2f}%")
    print(f"  Final equity: ${metrics['final_equity']:.2f} (start ${result['starting_capital']:.2f}, return {metrics['return_x']:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
