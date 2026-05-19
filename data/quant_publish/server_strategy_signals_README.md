## server_strategy_signals.parquet

Per-trade P&L feed published by the **server team** (claudehost) for consumption by the P3 v0.4 regime attribution matrix builder.

### Schema

| Column | Type | Notes |
|---|---|---|
| `strategy_id` | str | Discriminator. Values currently published below. |
| `entry_date` | datetime64[ns, UTC] | Date the trade opened (normalized to UTC midnight). |
| `exit_date` | datetime64[ns, UTC] | Date the trade closed (normalized to UTC midnight). |
| `net_pnl_pct` | float | Realized P&L as a percentage of cost basis. **Net of friction** (kraken_pro_dynamic at fixed-vol=$500K). |
| `hold_days` | int | `max(1, exit_date - entry_date)`. Single-day TPs are clamped to 1 so `sqrt(252/mean_hold_days)` doesn't divide by zero. |

### Published strategy rows

| `strategy_id` | n_trades | Window | Source | Config |
|---|---:|---|---|---|
| `stream_2c_grid_inverse_aggressive` | 9634 | 2022-09-15 → 2026-05-16 | simulator | Doctrine §9.5 production profile: bear=2.0×, choppy=1.0×, sideways=1.0×, steady_bull=0.0× |
| `stream_2c_grid_ungated` | 10094 | 2022-09-15 → 2026-05-16 | simulator | AB-comparison baseline (no regime gating) |

Both rows are produced by `scripts/backtest-crypto-dca-grid.py` in `lore-line/euieInvest` (server-team repo). Config matching doctrine §2.7 multi-version × N=12 sweet spot:

```
--all --symbols BTC-USD,ETH-USD,SOL-USD,ADA-USD,AVAX-USD,DOT-USD,LINK-USD,ATOM-USD,RUNE-USD,FET-USD,DOGE-USD,XRP-USD
--base-pct 0.5 --n-safety-orders 9 --first-so-step 2.5747011371995105
--so-step-scale 1.6942997249142477 --so-volume-scale 2.30 --strand-ban-days 122
--fixed-friction-vol-30d 500000
--regime-profile {inverse_aggressive,ungated}
--regime-labels-path data/quant_publish/regime_labels_v1.parquet
```

### Sanity-check vs canonical sweep

| Profile | Canonical CAGR (heaven-pc) | Server reproduction (claudehost) | Δ |
|---|---:|---:|---:|
| inverse_aggressive | +47.18% | +47.53% | +0.35pp |
| ungated | +45.19% | +45.54% | +0.35pp |

Both deltas are within sim noise from OHLCV differences between the two databases (Binance ingest sources are identical, but resampling/boundary handling differs slightly).

### Cadence

Weekly refresh. Server-team will push a new revision after major doctrine config changes (universe expansion, friction tier change, base_pct shift, etc.).

### Deferred / not yet published

- `stream_1_buffett` — requires FIFO matching against pre-tracking holdings (the order ledger starts mid-history, so cost basis backfill from `positions_snapshot` is needed before per-trade P&L is meaningful).
- `stream_3_hype` — paper-only spec, no trades yet.
- `stream_4_scalping` — collapsed into Stream 2 per doctrine §1.

### Source

- Simulator: `lore-line/euieInvest:scripts/backtest-crypto-dca-grid.py`
- Regime labels: `data/quant_publish/regime_labels_v1.parquet` (consumer-team)
- Trade-export flag: `--export-trades-parquet` (added in commit `91d1c79`)

Reference: issues #20 (inverse-gating result), #22 (publish schema agreement).

— server team
