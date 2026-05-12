# CLAUDE.md — euieInvest-quant project brief

Read this first. Every session.

---

## 1. Mission

**Discovery, not validation.** Find the pre-breakout fingerprint of equities
that run **+20% in 30 trading days**. Let the data tell us what winners
share — do NOT bias the feature set toward any prior doctrine.

A hand-crafted "Tier-3 anomaly" doctrine (volume breakout + trend +
persistence + cluster) **runs live** on the trading platform on
`claudehost`. The universe is **S&P 500 ∪ NDX 100 ∪ user watchlist ∪
discovery candidate pool** (~2,046 distinct symbols over 5y, ~2.43M
bars). An earlier project brief mentioned 641 historical doctrine
flags — that figure was a **backtest projection** that never wrote to
the `anomaly_flags` table. The live table is seeded 2026-05-12 with a
single open flag (SMCI) and accumulates at ~0–2 signals/week as the
doctrine warms up. **Pre-2026-05-12 has no baseline cohort to compare
against.** Step 5 (CLAUDE.md §5) must be designed around the
live-trickle reality, not the backtest count — see §5 notes below.

## 2. Hardware

- Host: `heaven-pc` (Win11)
- GPU: NVIDIA RTX 5090 (Blackwell, compute_cap 12.0 / sm_120)
- Driver: 596.36 supports CUDA 12.8+
- Runtime: Docker Desktop with NVIDIA Container Toolkit on the WSL2 backend
- Network: Tailscale tailnet member (peer: `claudehost`)

## 3. Architecture decisions

- **Offline-of-production.** This repo does not run alongside the trading
  platform. It pulls source data from euieInvest's read-only data API,
  runs analysis locally on heaven-pc, and emits batch JSON / Markdown
  reports. The trading platform on `claudehost` consumes outputs
  out-of-band (e.g. a shared report directory, or a future write-back
  endpoint).
- **Source-data API on claudehost; no model-serving API yet.** euieInvest
  exposes `/api/v1/*` parquet endpoints over Tailscale-bound HTTP — see
  `docs/api-contract.md` for the contract this repo consumes. A
  **model-serving / inference API on heaven-pc** is explicitly out of
  scope until the Phase 2 go/no-go gate (§14) is cleared. The two are
  separate: source-data serving is allowed in Phase 1; inference serving
  is not.
- **Polars, not pandas.** Single source of truth for tabular dtypes.
- **Parquet over Tailscale** as the data wire format. ~10× smaller than
  NDJSON for OHLCV; zero-copy through pyarrow into polars.
- **uv-managed**, Python 3.11 pinned via `.python-version` and `pyproject.toml`.

## 4. Source data shape

The local cache lives under `data/snapshots/`:

- `ohlcv.parquet` — `price_history` (symbol, date, close, high, low, volume)
- `peer_groups.json` — `{group_name: [symbol, ...]}`
- `anomaly_flags.parquet` — 22-col flag log (id, symbol, flag_date,
  fired_at, pivot_price, vol_mult, rsi, sma20, sma50, peer_group, tier,
  status, position_units, entry_price, entry_at, peak_price,
  trailing_stop, invalidated_at, exited_at, exit_reason, dismissed_until,
  notes)
- `cursor.json` — server cursor + fetch timestamp; produced by the pull
  script, consumed by future incremental syncs

These are refreshed by `scripts/pull-via-api.py`, which calls
euieInvest's `/api/v1/{ohlcv,peer-groups,anomaly-flags,snapshot-cursor,health,symbols}`
endpoints. The canonical wire contract is `docs/api-contract.md`.

`/api/v1/symbols` (contract §5.6, shipped 2026-05-12) returns per-symbol
static-ish metadata: `status`, `last_seen`, `shares_outstanding`,
`sector`, `listing_date`. Backs the `cap_bucket` and `market_regime`
features plus survivorship filtering. Two caveats baked into the spec:
**(a) `status` is uniformly `active` today** — the discovery pipeline
keeps every tracked symbol fresh, so truly-delisted names never enter
`price_history` (the §11 survivorship caveat still holds; `last_seen`
is the honest signal if drift starts). **(b) `shares_outstanding` is
current, not point-in-time**: `cap_bucket` for a 2021 row uses today's
shares × that day's close, the agreed-upon approximation from §5.6's
fallback clause. `sector` is null on ~7 ETFs/index trackers (SPY, QQQ,
etc.) — feature code must handle nullable-sector (skip-or-bucket-as-Unknown).

`price_history` covers 2021-05-12 → 2026-05-11 (5y, ~2.43M rows, ~2,046
symbols). `anomaly_flags` is the live trading-platform flag log
(seeded 2026-05-12 with the first real signal). Treated as a
**baseline cohort to beat** in Step 5 — not as a feature source. There
is no pre-2026-05-12 history; see §1.

**Data-shape gotchas (load-bearing — confirmed with the trading-platform team 2026-05-12):**

- **Adjustment basis (consistent split-adjustment; dividends handled
  via a separate column).** All four stored columns — `close`, `high`,
  `low`, `volume` — are Yahoo's `chart()` values, **all split-adjusted
  end-to-end** (NOT mixed-basis, as an earlier draft of this brief
  said — that was a server-side reasoning error corrected on
  2026-05-12 and verified empirically against the NVDA 10:1 and
  TSLA 3:1 splits: no jump on either side of the split day). The
  new `open` column (server migration done; consumer un-blocked once
  [PR #10](https://github.com/lore-line/euieInvest-quant/pull/10) merges)
  is also split-adjusted, so per-bar features that mix close with
  high/low/open have a **consistent basis** — no split-day distortion.
  **What's NOT in `close`:** dividends. The new `close_adj` column
  (same migration as `open`) is Yahoo's native `adjclose` =
  split + dividend-adjusted. Use `close_adj` for total-return labels
  and any feature that cares about dividend-inclusive returns. The
  `(close, close_adj)` divergence is a clean ex-dividend-date detector
  with no other source of systematic divergence between the two.
- **`open` column server-side ready, consumer un-blocked on PR #10.**
  Once `open` lands in /ohlcv, two feature functions in
  `quant/features/gaps.py` (`gap_pct`, `body_range_ratio`) come
  off scaffold.
- **Immutable historicals.** The upstream ETL uses `INSERT OR IGNORE`;
  stored rows are never rewritten. Corporate actions do not trigger a
  re-fetch, so historical adjusted closes only drift if Yahoo's
  stored splits change. Treat as immutable; reproducible enough for
  backtesting without point-in-time snapshots.
- **Survivorship bias is present.** Delisted symbols stay in
  `price_history` but stop receiving Yahoo updates. No `delisted_at`
  column today. The server team is adding `/api/v1/symbols` with
  `status` + `last_seen` so we can filter / weight. Until then,
  treat win-rate estimates as systematically optimistic — see §11.
- **Update cadence: daily batch.** `fetch-prices.mjs` runs weekdays
  at 11:00 UTC (07:00 ET), writing EOD prices. NOT intraday.
  Schedule `scripts/pull-via-api.py` after that window. The
  `/snapshot-cursor` endpoint's `as_of` is server wall-clock at
  request time, not data-batch time.
- **Time zones.** `price_history.date` and `anomaly_flags.flag_date`
  are **exchange-local trading dates** (ET, NYSE/NASDAQ). The
  `*_at` timestamps in `anomaly_flags` are **UTC** with explicit `Z`.
  All listings USD; no non-USD symbols.
- **`anomaly_flags.status` enum (drift-corrected).** Actual DB values
  are `{active, invalidated, entered, exited, dismissed}`. The
  initial contract draft listed `open` instead of `active`. Code
  filtering by status must use `active`. See `docs/api-contract.md`
  §5.5 for the full state machine.
- **`peer_groups` membership.** Currently 6 groups, 33 entries total:
  `ai_infra`, `cybersec`, `hyperscaler`, `semicap`, `software_ai`,
  `fintech_rails`. Small dict, refetch on every cursor change.

**Legacy fallback** (during cutover): if the parquet/JSON files are
absent, `quant.data.loader` transparently reads a local SQLite snapshot
at `data/snapshots/euieinvest.db`, refreshed by the legacy
`scripts/pull-snapshot.{sh,ps1}` rsync scripts. The legacy path will be
removed after the API is verified live in prod for ≥ 1 week — see
`plans/api-data-plane.md` PR #6 for the deletion criteria.

## 5. Methodology — 5 steps

**Step 1: Rich feature engineering.** 80–120 features per row drawn from the
7 module families in §7. No doctrine bias — include everything plausible
and let SHAP cull.

**Step 2: Supervised discovery.** XGBoost binary classifier with
`scale_pos_weight = #negatives / #positives`. The brief originally
estimated winners at ~4–7% — Step 1's smoke run on the live snapshot
gave **18.78%** non-null positive rate (so `scale_pos_weight ≈ 4.3`,
not ~14–23 as the brief assumed). Train on 2021-05-12 → 2023-12-31,
validate on 2024, **freeze before touching 2025**. SHAP summary
identifies the high-signal features.

**Step 3: Unsupervised winner clustering.** KMeans on the winner-only
subset for k ∈ {3, 5, 8}; pick best by silhouette score. Each cluster is a
candidate "fingerprint shape".

**Step 4: Counterfactual analysis.** For each winner cluster, find the
closest non-winners in feature space. The delta isolates *which* features
flipped the outcome.

**Step 5: Tier-3 doctrine comparison.** Live `anomaly_flags` is
~empty (1 row as of 2026-05-12; ~0–2/week incoming). Pre-2026-05-12
has zero baseline cohort. The original Step 5 design (overlap +
recall + missed-winner analysis vs hundreds of historical flags) is
**not viable** until the table accumulates more signal. Two
recommended redesigns:

1. **Forward overlap only.** Run the ML pipeline daily after the
   2026-05-12 cutoff; record for each new doctrine flag whether the
   ML top-decile cohort already flagged it earlier. Tracks
   ML→doctrine lead time. Useful but slow to evaluate.
2. **Re-simulate the doctrine** in this repo as a deterministic
   rule-based "shadow" cohort over the full 5y. Compare ML
   top-decile vs that synthetic cohort on the same 2025 holdout. Not
   the trading platform's actual doctrine — a re-implementation we
   own. Faster to evaluate but introduces a re-implementation risk.

Decision deferred to when Step 5 is implemented; pick (1) if the
live table grows fast, (2) if we want a Phase 2 gate decision sooner.

## 6. Target definition (exact)

Primary label, computed per row by `src/quant/labels.py` against
**`close_adj`** (split + dividend-adjusted total return per §4):

```
is_winner[t] := (max(close_adj[t+1 .. t+30]) / close_adj[t]) >= 1.20
```

Last 30 rows per symbol have null `is_winner`. `src/quant/labels.py`
takes a `price_col` parameter; `scripts/discover.py` passes
`price_col="close_adj"` to match this spec. Features still operate on
`close` (split-adj) per CLAUDE.md §11 — keeping label-basis (total
return) separate from feature-basis (split-only) is intentional.

Three comparison variants to run side-by-side at the analysis stage:

- 15% / 30 days  (looser threshold, same window)
- 30% / 60 days  (stronger move, longer window)
- close-to-close 20% / 30 days (`close[t+30] / close[t] >= 1.20`, not max)

## 7. Feature category map

Each module owns a category. Function signatures are spec — fill them in
next session.

- `features/price.py` — `sma_distance`, `sma_slope`, `band_position`,
  `n_day_high_low`. Close vs SMA{10,20,50,200}, slope of SMA over lookback,
  Bollinger band position, % of N-day high/low.
- `features/volume.py` — `vol_mult`, `obv_slope`, `accumulation_distribution`.
  Volume relative to SMA{5,10,30,60}, OBV slope, A/D line.
- `features/volatility.py` — `atr_pct`, `bb_squeeze`, `nr4_nr7`, `hv_ratio`.
  ATR as % of close, Bollinger squeeze ratio, NR4/NR7 inside-bar flags,
  short/long HV ratio.
- `features/momentum.py` — `rsi`, `macd`, `roc`, `consecutive_run`.
  RSI{2,5,14}, MACD line+signal+hist, ROC{5,10,20,60}, run-length of
  consecutive up/down days.
- `features/relative.py` — `rel_strength_spy`, `rel_strength_sector`,
  `peer_zscore`. Symbol vs SPY, vs its peer-group mean, z-score within cluster.
- `features/gaps.py` — `gap_pct`, `range_expansion`, `body_range_ratio`,
  `inside_bar`. Overnight gap, true-range expansion, candle body / range,
  inside-bar flag.
- `features/behavioral.py` — `days_since_last_20pct`, `market_regime`,
  `cap_bucket`. Recency of last 20% run, SPY trend regime, market-cap bucket.

## 8. Validation discipline — HARD rules

```
Train:   2021-05-12 → 2023-12-31
Val:     2024-01-01 → 2024-12-31
Holdout: 2025-01-01 → today   (touched ONCE at the end)
```

- **No random splits.** Time-series only.
- **No peeking at holdout while iterating.** It is touched once, after the
  val set has frozen hyperparameters and feature set.
- Splits are produced by `src/quant/backtest/temporal.py::split_by_date`,
  which asserts every row lands in exactly one bucket.

## 9. Class imbalance

Winners are **~18.94% of non-null rows** at the +20%/30d threshold on
the `close_adj` (total-return) label, per the 2026-05-12 post-cleanup
rebaseline on the live snapshot (2,430,543 rows → 448,741 winners
across 2,369,196 non-null rows). The bootstrap brief estimated 4–7%;
that turned out to be a rough guess. The prior figure (18.78%, on
`close`) is documented here for traceability — see §11 for the drift
breakdown and the DEC contamination correction. Therefore:

- `scale_pos_weight ≈ 4.28` for the full population (recompute
  exactly per the train slice; not 14–23 as the brief assumed).
- **Report** precision@top-decile AND recall, not accuracy/F1 alone.
- The discovery question is "what does the top-decile predicted-positive
  cohort look like", not "did the model beat 50% accuracy".
- 18.94% is higher than typical price-discovery setups. Sanity check
  before training: confirm the label spec in §6 still feels right;
  it's the threshold + lookahead that drives this number.

## 10. Target leakage — defend against it

- Never include any column derived from future rows in features.
- The only forward-looking column in the dataset is `is_winner`, and it is
  only used as `y`.
- Add a CI test (Phase 2) that asserts no feature column references a
  future-dated row.

## 11. Honest caveats

- **Survivorship bias.** Delisted symbols stay in `price_history` but
  stop receiving Yahoo updates. No `delisted_at` column today.
  Win-rate estimates systematically overstate reality until the
  server team adds `/api/v1/symbols` with `status` + `last_seen`.
- **Dividend handling.** The stored `close` is split-adjusted only
  (NOT dividend-adjusted). Total-return features must use the new
  `close_adj` column from the /ohlcv migration tracked in
  [PR #10](https://github.com/lore-line/euieInvest-quant/pull/10) +
  the server-side flip (now merged + deployed). **Pre-cleanup
  rebaseline (2026-05-12 morning):** positive rate on `close_adj`
  was **18.9627%** (449,387 / 2,369,842), vs **18.7822%**
  (445,108 / 2,369,842) on `close` — apparent +0.18 pp upward drift.
  Net 4,319 → winner flips, 40 → non-winner flips; legitimate lift
  on high-yield REITs (BXMT, RPT), shipping/MLP-style payers (LPG,
  GNK, FLNG), and small-cap banks (LKFN, BCC) as expected.
  **DEC contamination correction (2026-05-12 afternoon, supersedes
  prior amendment):** the largest outlier — `DEC` with 604 lift, 14%
  of all flips in a single ticker — was **not** a real high-yielder.
  Pre-2023-12-05 rows for `DEC` were LSE-listed data leaking into
  the NYSE symbol slot, with `close` in GBP single-digits and Yahoo
  `adjclose = 0.0` (no historical close_adj on the LSE side). The
  +604 lift was a div-by-zero artifact in the label function:
  `max(close_adj[t+1..t+30]) / 0 = inf >= 1.2 → True`. An earlier
  spot-check that claimed "smooth `close_adj / close` slope" was
  reading the post-2023-12-05 NYSE rows only, missing the 646
  pre-NYSE rows where `close_adj = 0`. **Server cleaned up
  2026-05-12** — `price_history` row count 2,431,189 → 2,430,543
  (Δ -646). **Post-cleanup**: positive rate on `close_adj` is
  **18.9406%** (448,741 / 2,369,196); the cleanup removed exactly
  646 spurious winners (close_adj-zero rows counted as winner under
  the inf-comparison), and the base rate barely moves because the
  denominator drops by the same amount. The +0.18 pp pre-cleanup
  drift narrows to **+0.16 pp** post-cleanup (18.9406% vs ~18.78%
  on `close`, denominator-adjusted) — still small, still legitimate
  dividend lift on the REIT/shipping/MLP cohort. See
  PR #1, 2026-05-12 comments for the full table.
- **(Corrected.)** An earlier draft of this caveats list claimed
  "mixed adjustment basis" (close split+div-adjusted, raw OHLV). That
  was wrong — all stored columns are consistently split-adjusted. See
  §4 for the corrected description. Per-bar features mixing close
  with high/low/volume/open are **not** distorted on split days.
- **Alpha decay is real.** Anything we find in 2021–2024 may not work in
  2025+. The holdout is our only honest read on this.
- **Sector confounding.** The 2023–2024 AI tape may dominate winner
  clusters. Step 4 counterfactuals should partial out the obvious
  AI-adjacent bias.
- **Live-doctrine target leakage (theoretical).** The hand-crafted
  doctrine on claudehost is now live (`anomaly_flags` table is the
  trading platform's actual state, not a backtest). When the doctrine
  exits a position, that close + volume ends up in `price_history`
  for the next bar. At today's signal volume (1 row total) this is
  noise. If the doctrine scales meaningfully, feature data downstream
  of doctrine-driven trades will reflect the doctrine's own exits.
  Worth re-checking before each training run if signal volume has
  grown.

## 12. NOT building yet

- No deep learning. No LSTM / Transformer. XGBoost is enough until it isn't.
- **No model-serving / inference server on heaven-pc.** No live trading
  hooks. No continuous training. (The source-data API on claudehost —
  see §3 and §4 — is allowed and in fact the operational data plane.
  The prohibition here is about exposing the *model's* predictions over
  HTTP.)
- No alternative data (news, options flow, fundamentals). Phase 1 is
  price/volume/peer only.

## 13. Verdict format — `reports/winner-fingerprint.md`

```markdown
# Winner fingerprint — <run timestamp>

## Cohort
- Period: <train_end> → <holdout_end>
- Universe size: <N> symbols, <M> rows
- Winners: <#winners> (<rate>%)

## Holdout performance
- AUC: <…>
- Top-decile precision: <…>
- Top-decile vs SPY 30d delta: <±X.X%>

## Top SHAP features
| Feature | Direction | Cluster bias |
|---------|-----------|--------------|
| …       | + / −     | …            |

## Clusters (best k = <k>)
For each cluster: defining features, size, win rate, AI/sector exposure.

## Tier-3 doctrine overlap
- Hand-crafted flags: <N>
- ML top-decile: <M>
- Intersection: <K>
- Winners only ML found: <…>
- Winners only doctrine found: <…>

## Verdict
- Phase 2 go/no-go: <pass/fail>  (see CLAUDE.md §14)
- Caveats: <list>
```

## 14. Phase 2 go/no-go gate

All three must hold on the **untouched holdout**:

1. AUC ≥ 0.55
2. Top-decile predicted-positive cohort beats SPY by ≥ +2% on 30d forward return
3. ≥ 1 cluster has interpretable distinguishing features that are not
   merely "AI-adjacent in 2024"

If any fails: do not advance to Phase 2 (inference server / live integration).
Iterate on features, not on the holdout.

## 15. Tailnet topology

- `claudehost` (Tailscale `100.68.86.56`) — source of truth; runs
  euieInvest trading prod and hosts the data API on a Tailscale-bound
  port.
- `heaven-pc`  (Tailscale `100.103.175.27`) — this repo, RTX 5090 compute.

**Primary data plane (once API is live)**: HTTP over Tailscale.
`heaven-pc` resolves `EUIEINVEST_API_BASE_URL` (e.g.
`http://100.68.86.56:8443`) and calls `/api/v1/*`. No SSH required for
routine data fetches.

**Legacy data plane (during cutover)**: SSH+rsync. The legacy
`scripts/pull-snapshot.{sh,ps1}` scripts rsync the SQLite snapshot
directly. Still operational; deleted once the API is verified in prod.

SSH setup (still useful for ad-hoc admin and the legacy path):

```sh
# First-run host-key TOFU:
ssh -o StrictHostKeyChecking=accept-new euie@claudehost true

# Public-key auth (one-time, if not already set up):
ssh-copy-id euie@claudehost
```

## 16. Docker usage cheat sheet

```sh
docker compose build                         # build the image
docker compose run --rm test                 # pytest
docker compose run --rm discover             # run scripts/discover.py
docker compose run --rm dev                  # interactive bash
docker compose run --rm dev nvidia-smi       # GPU sanity (should see RTX 5090)
```

The compose file uses `runtime: nvidia`. The
`deploy.resources.reservations.devices` form is the v3-spec equivalent —
switch if the v1 form ever breaks on a future Docker version.
