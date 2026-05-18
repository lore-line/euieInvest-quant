# quant_publish — stable artifacts for platform consumption

This directory holds parquet artifacts produced by the quant track that the
platform-side (euieInvest) repo consumes as stable, version-pinned inputs.

Unlike `data/snapshots/` and `data/features/` (gitignored, run-local), files
in `data/quant_publish/` ARE committed to git. Treat them as a publish surface:
overwriting one is an API-breaking change — bump the version suffix (`_v1` →
`_v2`) and keep the old file until consumers cut over.

## Current artifacts

### `equity_slow_universe_v1.parquet`

Russell-1000-ish liquid-equity universe, used as the scan universe for the
equity_slow (Stream 2b) momentum signal class.

**Origin:** `equity_momentum_walkforward.py` per-window universe filter, taken
from the 2026-05-18 slow_g25_vc15 walkforward run (the validated production
config). Filter rule:

  - `close_adj >= $10` (entry price gate)
  - `rolling_mean(volume × close_adj, 30) >= $10M` (ADV-dollar gate)

Applied at every candidate row across 2021-05-18 → 2026-05-18. A symbol is
included iff it has ≥30 qualifying days in the 5-year window.

**Schema:**

| Column | Type | Notes |
|---|---|---|
| `symbol` | String | Yahoo/canonical ticker |
| `first_seen_date` | Date | first date the symbol passed the filter |
| `last_seen_date` | Date | last date the symbol passed the filter |
| `mean_avg_dollar_volume_30d` | Float64 | mean across qualifying days |
| `mean_close_adj` | Float64 | mean across qualifying days |
| `n_qualifying_days` | UInt32 | count of qualifying days in 5y window |
| `passes_filter` | Boolean | True for all rows in this file (kept for forward-compat) |

**Size:** 1,491 symbols.

**Versioning:** when the filter spec changes (target_pct, min_price, ADV
threshold, lookback window, vol-confirm), bump `_v1` → `_v2`. Document the
delta below.

**Consumer:** platform-side `scripts/generate-momentum-signals.mjs` for the
equity_slow scan universe (replaces ad-hoc "anything in price_history with
252+ bars" auto-expansion). Per PR #1 issuecomment-4474233879 wire-up queue
item #1.

**Regeneration:**

```powershell
python -m quant.tracks.equity_momentum_walkforward `
    --spec slow `
    --vol-confirm-mult 1.5 `
    --out-dir D:/quant-runs/<date>-equity_momentum_slow_g25_vc15_walkforward
```

Then copy `<out-dir>/universe_filtered.parquet` into this directory under
the appropriate version suffix.
