## server_strategy_signals.parquet + server_strategy_daily.parquet

Two complementary feeds published by the **server team** (claudehost):

- **`server_strategy_signals.parquet`** — per-trade P&L ledger for the P3 v0.4 per-trade attribution matrix
- **`server_strategy_daily.parquet`** — per-day portfolio state for the P3 v0.6 daily aggregation matrix (capital-utilization-aware cross-class comparison)

---

## server_strategy_signals.parquet (per-trade)

Per-trade P&L feed for the P3 v0.4 regime attribution matrix builder.

### Schema

| Column | Type | Notes |
|---|---|---|
| `strategy_id` | str | Discriminator. Values currently published below. |
| `entry_date` | datetime64[ns, UTC] | Date the trade opened (normalized to UTC midnight). |
| `exit_date` | datetime64[ns, UTC] | Date the trade closed (normalized to UTC midnight). |
| `net_pnl_pct` | float | Realized P&L as a percentage of cost basis. **Net of friction** (kraken_pro_dynamic at fixed-vol=$500K). |
| `hold_days` | int | `max(1, exit_date - entry_date)`. Single-day TPs are clamped to 1 so `sqrt(252/mean_hold_days)` doesn't divide by zero. |

### Published strategy rows

| `strategy_id` | n_trades | Window | Source | Sharpe class | Config |
|---|---:|---|---|---|---|
| `stream_2c_grid_inverse_aggressive` | 9634 | 2022-09-15 → 2026-05-16 | DCA-grid sim | tp_clustered | Doctrine §9.5: bear=2.0×, choppy=1.0×, sideways=1.0×, steady_bull=0.0× |
| `stream_2c_grid_ungated` | 10094 | 2022-09-15 → 2026-05-16 | DCA-grid sim | tp_clustered | AB-comparison baseline (no regime gating) |
| `stream_1b_momentum_donchian` | 194 | 2024-08-05 → 2026-05-15 | momentum backtest | realistic | 252d Donchian + SMA(50)>SMA(200) + vol-confirm 1.5× / stop 10% / target 40% / max-hold 120d / max-concurrent 20 |

DCA-grid rows are produced by `scripts/backtest-crypto-dca-grid.py`. Stream 1b row is from `scripts/backtest-momentum-portfolio.py --volume-confirm 1.5 --stop-pct 0.10 --target-pct 0.40 --max-hold-days 120 --max-concurrent 20 --vol-scaled-sizing`. Both in `lore-line/euieInvest` (server-team repo).

DCA-grid config matching doctrine §2.7 multi-version × N=12 sweet spot:

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

### Stream 1b summary stats (backtest, 2024-08 → 2026-05)

- 194 trades, 39.2% win rate (37.6% net of friction)
- Mean per-trade pnl: +4.99%; median -10.24% (typical momentum distribution — wide right tail from 40% targets)
- Mean hold: 57 days (matches doctrine §1b spec)
- CAGR: +37.81%; Sharpe (portfolio, not per-trade): 2.178; Max DD: -17.1%
- `sharpe_class = "realistic"` — distributed exits, directly comparable to consumer-side stream_2a/2b

### Deferred / not yet published

- `stream_1_buffett` — requires FIFO matching against pre-tracking holdings (the order ledger starts mid-history, so cost basis backfill from `positions_snapshot` is needed before per-trade P&L is meaningful).
- `stream_3_hype` — paper-only spec, no trades yet.
- `stream_4_scalping` — collapsed into Stream 2 per doctrine §1.

### Source

- Simulator: `lore-line/euieInvest:scripts/backtest-crypto-dca-grid.py`
- Regime labels: `data/quant_publish/regime_labels_v1.parquet` (consumer-team)
- Trade-export flag: `--export-trades-parquet` (added in commit `91d1c79`)

Reference: issues #20 (inverse-gating result), #22 (publish schema agreement).

---

## server_strategy_daily.parquet (per-day)

Per-day portfolio state for the P3 v0.6 daily aggregation matrix. Built from the simulator's equity curve (already populated per-bar) deduplicated to daily granularity, joined to closed_deals for per-day open-deal counts and active-capital%.

### Schema

| Column | Type | Notes |
|---|---|---|
| `strategy_id` | str | Same discriminator as in `server_strategy_signals.parquet`. |
| `date` | datetime64[ns, UTC] | Calendar date (normalized to UTC midnight). |
| `daily_return_pct` | float | `(equity_t / equity_{t-1} - 1) × 100`. First-day value is 0. |
| `open_deal_count` | int | Number of deals open at end-of-day (across all symbols × versions). |
| `active_capital_pct` | float | `sum(cost_basis_of_open_deals) / equity × 100`. Can exceed 100% as compounding grows open positions past starting capital. |

### Currently published

| `strategy_id` | n_days | Window | Source |
|---|---:|---|---|
| `stream_2c_grid_inverse_aggressive` | 1340 | 2022-09-15 → 2026-05-16 | DCA-grid sim, inverse_aggressive profile |
| `stream_2c_grid_ungated` | 1340 | 2022-09-15 → 2026-05-16 | DCA-grid sim, baseline (no gating) |

### What this unblocks

The per-trade `Sharpe` is **not comparable across strategy classes** (TP-clustered DCA vs distributed-exit momentum — see "CRITICAL CAVEAT" in P3 v0.4.1 matrix README). The per-day matrix is the right cross-class comparator:

- Per-day daily_return_pct is in the **same units** for every strategy (annualized portfolio return per regime, capital-utilization-aware via `active_capital_pct`)
- Consumer-team's v0.6 matrix builder reads this parquet to compute per-regime daily Sharpe → directly comparable across strategy classes
- Enables **wrapper × regime allocation decisions** without the Sharpe-inflation footgun

### Source

- Daily export: `--export-daily-parquet` flag in `scripts/backtest-crypto-dca-grid.py` (added in commit `939c271`)
- Same simulator + config as the trades feed; same kraken_pro_dynamic friction model

Reference: euieInvest-quant#22 v0.6 per-day aggregation contract.

— server team
