# CLAUDE.md — euieInvest-quant project brief

Read this first. Every session.

---

## 1. Mission

**Discovery, not validation.** Find the pre-breakout fingerprint of equities
that run **+20% in 30 trading days**. Let the data tell us what winners
share — do NOT bias the feature set toward any prior doctrine.

A previous hand-crafted "Tier-3 anomaly" doctrine (volume breakout + trend +
persistence + cluster) was backtested on this exact dataset (Russell 2000 ∪
watchlist, 5y, 2,046 symbols, ~2.43M bars) and produced only **+0.4% alpha**
vs SPY across **641 flags**. That doctrine is the **baseline cohort to
beat** in Step 5 — not a prior we encode into features.

## 2. Hardware

- Host: `heaven-pc` (Win11)
- GPU: NVIDIA RTX 5090 (Blackwell, compute_cap 12.0 / sm_120)
- Driver: 596.36 supports CUDA 12.8+
- Runtime: Docker Desktop with NVIDIA Container Toolkit on the WSL2 backend
- Network: Tailscale tailnet member (peer: `claudehost`)

## 3. Architecture decisions

- **Offline-of-production.** This repo does not run alongside the trading
  platform. It reads a periodic SQLite snapshot, runs analysis, and emits
  batch JSON / Markdown reports. The trading platform on `claudehost`
  consumes outputs out-of-band.
- **No FastAPI / inference server in Phase 1.** Add one only after the
  Phase 2 go/no-go gate (§14) is cleared.
- **Polars, not pandas.** Single source of truth for tabular dtypes.
- **uv-managed**, Python 3.11 pinned via `.python-version` and `pyproject.toml`.

## 4. Source data shape

The snapshot is `data/snapshots/euieinvest.db` (rsync target of
`/home/euie/nextcloud/CODE/euieInvest/data/euieinvest.db.bak` on claudehost).
Three relevant tables:

```sql
price_history(symbol TEXT, date TEXT, close REAL, high REAL, low REAL, volume INTEGER)
peer_groups(group_name TEXT, symbol TEXT)
anomaly_flags(id INTEGER, symbol TEXT, flag_date TEXT, fire_date TEXT,
              pivot_price REAL, vol_mult REAL, rsi REAL, sma20 REAL, sma50 REAL,
              peer_group TEXT, tier TEXT, status TEXT, …)
```

`price_history` covers 2021-05-12 → 2026-05-11 (5y). `anomaly_flags` is the
prior doctrine's flag log — used as a baseline cohort, not a feature source.

## 5. Methodology — 5 steps

**Step 1: Rich feature engineering.** 80–120 features per row drawn from the
7 module families in §7. No doctrine bias — include everything plausible
and let SHAP cull.

**Step 2: Supervised discovery.** XGBoost binary classifier with
`scale_pos_weight = #negatives / #positives` (winners are ~4–7%). Train
on 2021-05-12 → 2023-12-31, validate on 2024, **freeze before touching
2025**. SHAP summary identifies the high-signal features.

**Step 3: Unsupervised winner clustering.** KMeans on the winner-only
subset for k ∈ {3, 5, 8}; pick best by silhouette score. Each cluster is a
candidate "fingerprint shape".

**Step 4: Counterfactual analysis.** For each winner cluster, find the
closest non-winners in feature space. The delta isolates *which* features
flipped the outcome.

**Step 5: Tier-3 doctrine comparison.** Overlap, recall, and missed-winner
analysis vs `anomaly_flags`. Hard question: does ML find winners the
hand-crafted doctrine missed, and does it filter out the doctrine's false
positives?

## 6. Target definition (exact)

Primary label, computed per row by `src/quant/labels.py`:

```
is_winner[t] := (max(close[t+1 .. t+30]) / close[t]) >= 1.20
```

Last 30 rows per symbol have null `is_winner`.

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

Winners are ~4–7% of rows. Therefore:

- XGBoost uses `scale_pos_weight = #negatives / #positives` from the train set.
- **Report** precision@top-decile AND recall, not accuracy/F1 alone.
- The discovery question is "what does the top-decile predicted-positive
  cohort look like", not "did the model beat 50% accuracy".

## 10. Target leakage — defend against it

- Never include any column derived from future rows in features.
- The only forward-looking column in the dataset is `is_winner`, and it is
  only used as `y`.
- Add a CI test (Phase 2) that asserts no feature column references a
  future-dated row.

## 11. Honest caveats

- **Survivorship bias.** Yahoo Finance does not serve cleanly-delisted
  names; the universe is *survivors*. We will systematically over-estimate
  win rates.
- **Alpha decay is real.** Anything we find in 2021–2024 may not work in
  2025+. The holdout is our only honest read on this.
- **Sector confounding.** The 2023–2024 AI tape may dominate winner
  clusters. Step 4 counterfactuals should partial out the obvious
  AI-adjacent bias.

## 12. NOT building yet

- No deep learning. No LSTM / Transformer. XGBoost is enough until it isn't.
- No inference server. No live trading hooks. No continuous training.
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

- `claudehost` (Tailscale `100.68.86.56`) — source of truth, runs euieInvest trading prod
- `heaven-pc`  (Tailscale `100.103.175.27`) — this repo, RTX 5090 compute

SSH from `heaven-pc` → `claudehost` must work before `scripts/pull-snapshot.*` runs:

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
