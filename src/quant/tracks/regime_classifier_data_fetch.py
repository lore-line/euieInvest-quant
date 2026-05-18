"""Regime classifier — Day 2 data fetch.

Per PR #1 issuecomment-4475264011 (Day 2 plan). Two data sources:

  1. **yfinance** for macro/cross-asset features that aren't in my local
     snapshot: ^VIX, HYG, LQD, GLD, DX-Y.NYB (or UUP as proxy).
  2. **sidecar OHLCV endpoint** (`/api/v1/ohlcv`) for crypto symbols
     (BTC-USD primary; ETH/SOL/etc. for alt-basket correlation feature).
     Per PR #1 issuecomment-4475233656, sidecar `price_history` was
     backfilled with 5y of crypto.

Output: writes to `data/snapshots/macro_panel.parquet` (gitignored, run-
local) and `data/snapshots/crypto_panel.parquet`. Caller composes these
with the existing `data/snapshots/ohlcv.parquet` (which has SPY) into
the feature pipeline.

Note: stays out of `data/quant_publish/` — these snapshots are run-time
data, not publish artifacts. The PUBLISHED artifact is the trained
regime_labels parquet downstream.

Status: Day 2 scaffold per PR #1 issuecomment-4475233656 unblock.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl


SIDECAR_BASE_URL = os.environ.get(
    "EUIEINVEST_API_BASE_URL", "http://100.68.86.56:8443"
)

# yfinance symbols for non-crypto macros
MACRO_YFINANCE_SYMBOLS = {
    "^VIX": "vix",            # CBOE volatility index
    "HYG": "hyg",             # iShares high-yield bond ETF (credit spread numerator)
    "LQD": "lqd",             # iShares investment-grade bond ETF (denominator)
    "GLD": "gld",             # SPDR gold ETF (flight-to-quality proxy)
    "UUP": "uup",             # Invesco DXY-tracking ETF (DXY proxy — DX-Y.NYB is futures, less liquid)
    "XLU": "xlu",             # utilities sector — defensive rotation indicator
    "XLE": "xle",             # energy sector — inflation/commodity sensitivity
}

# Crypto symbols to fetch from sidecar
CRYPTO_SYMBOLS = [
    "BTC-USD",  # primary
    # alt basket (for crypto_alt_to_btc_corr feature)
    "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "ATOM-USD",
]


def fetch_macros_yfinance(
    start_date: date | str = "2021-05-19",
    end_date: date | str = "2026-05-17",
    symbols: dict[str, str] = MACRO_YFINANCE_SYMBOLS,
) -> pl.DataFrame:
    """Fetch macro symbols from yfinance into a long-format polars panel.

    Returns: [date, symbol, close] with one row per (date, symbol).

    Uses auto_adjust=False so we get raw close (not split-adjusted) — for
    ETFs there are occasional small distributions but they don't impact
    regime features (which look at slopes, ratios, percentiles).
    """
    import yfinance as yf

    frames: list[pl.DataFrame] = []
    for ticker, _short in symbols.items():
        df_pd = yf.download(
            ticker, start=str(start_date), end=str(end_date),
            progress=False, auto_adjust=False,
        )
        if df_pd is None or df_pd.empty:
            print(f"  [warn] no yfinance data for {ticker}")
            continue
        # yfinance returns multi-index columns when downloading a single ticker
        # in 1.3+; flatten to just the 'Close' series.
        if isinstance(df_pd.columns, type(df_pd.columns)) and len(df_pd.columns.names) > 1:
            close = df_pd[("Close", ticker)]
        else:
            close = df_pd["Close"]
        # Reset index → date column; convert to polars
        close = close.reset_index()
        close.columns = ["date", "close"]
        pl_df = pl.from_pandas(close).with_columns(
            symbol=pl.lit(ticker),
            date=pl.col("date").cast(pl.Date),
            close=pl.col("close").cast(pl.Float64),
        ).select(["date", "symbol", "close"])
        frames.append(pl_df)
        print(f"  {ticker}: {pl_df.height} rows")

    if not frames:
        return pl.DataFrame({"date": [], "symbol": [], "close": []})

    panel = pl.concat(frames, how="vertical").sort(["symbol", "date"])
    return panel


def fetch_crypto_sidecar(
    sidecar_base: str = SIDECAR_BASE_URL,
    start_date: date | str = "2021-05-19",
    end_date: date | str = "2026-05-17",
    symbols: list[str] = CRYPTO_SYMBOLS,
) -> pl.DataFrame:
    """Fetch crypto OHLCV from the sidecar `/api/v1/ohlcv` endpoint.

    Per PR #1 issuecomment-4475233656, sidecar `price_history` was backfilled
    with 5y of BTC + alt basket. Endpoint returns **parquet** (default per
    api-contract.md §5.3), so we read the bytes via polars.

    Returns: [date, symbol, open, close, close_adj, high, low, volume].
    """
    import io
    import requests

    symbols_csv = ",".join(symbols)
    url = f"{sidecar_base.rstrip('/')}/api/v1/ohlcv"
    # Contract uses `since`/`until`, NOT `from_date`/`to_date`.
    params = {
        "symbols": symbols_csv,
        "since": str(start_date),
        "until": str(end_date),
    }
    print(f"  GET {url}?symbols={symbols_csv}&since={start_date}&until={end_date}")

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        body = r.content
    except Exception as e:  # noqa: BLE001
        print(f"  [error] sidecar fetch failed: {e}")
        return pl.DataFrame({"date": [], "symbol": [], "close": []})

    if not body:
        print("  [warn] sidecar returned empty body")
        return pl.DataFrame()

    try:
        df = pl.read_parquet(io.BytesIO(body))
    except Exception as e:  # noqa: BLE001
        print(f"  [error] failed to parse parquet body: {e}")
        return pl.DataFrame()

    print(f"  crypto sidecar: {df.height} rows across {df['symbol'].n_unique()} symbols, columns: {df.columns}")
    return df.sort(["symbol", "date"])


def load_spy_from_snapshot(
    snapshot_path: str | Path = "data/snapshots/ohlcv.parquet",
) -> pl.DataFrame:
    """Load SPY OHLCV from the existing equity snapshot."""
    df = pl.read_parquet(str(snapshot_path)).filter(pl.col("symbol") == "SPY")
    return df.sort("date")


def main(
    out_dir: str | Path = "data/snapshots",
    start_date: str = "2021-05-19",
    end_date: str = "2026-05-17",
) -> None:
    """Day 2 ENTRYPOINT: fetch + save macro + crypto panels."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Fetching macros via yfinance ===")
    macros = fetch_macros_yfinance(start_date, end_date)
    macros_path = out / "regime_macro_panel.parquet"
    macros.write_parquet(macros_path)
    print(f"  wrote {macros_path} ({macros.height:,} rows)")

    print("\n=== Fetching crypto via sidecar ===")
    crypto = fetch_crypto_sidecar(start_date=start_date, end_date=end_date)
    crypto_path = out / "regime_crypto_panel.parquet"
    if crypto.height > 0:
        crypto.write_parquet(crypto_path)
        print(f"  wrote {crypto_path} ({crypto.height:,} rows)")
    else:
        print(f"  [warn] no crypto data fetched; sidecar offline?")

    print("\n=== SPY from local snapshot ===")
    spy = load_spy_from_snapshot()
    print(f"  SPY: {spy.height} rows, {spy['date'].min()} -> {spy['date'].max()}")

    print("\n=== Summary ===")
    print(f"  macros: {macros['symbol'].n_unique() if macros.height else 0} symbols")
    print(f"  crypto: {crypto['symbol'].n_unique() if crypto.height else 0} symbols")
    print(f"  SPY:    {spy.height} rows")


if __name__ == "__main__":
    main()
