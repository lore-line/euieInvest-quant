# `lore-line/euieInvest-reports` — repo layout

> **Status**: contract, live since 2026-05-12. Repo at
> [`lore-line/euieInvest-reports`](https://github.com/lore-line/euieInvest-reports).
> First real run committed 2026-05-12 (Step 2 XGB, `runs/2026-05-12/`).
> The reports-repo README points here — **schema changes go in this
> file**, not in the README. Cumulative discussion: PR #1 thread, with
> the per-arch / per-day / cross-pipeline-attribution decisions pinned
> in [issuecomment-4435765226](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4435765226).

The euieInvest trading platform on `claudehost` consumes the quant
side's discovery output via a git-pull-and-parse cron. This is that
output's repo layout.

## Tree

```
euieInvest-reports/
├── README.md
└── runs/
    ├── 2026-05-12/                          # historical (Step 2 first run, kept as-is — no rename)
    │   └── (XGB Step 2 artifacts — see §Files)
    └── YYYY-MM-DD-<pipeline_step>/          # canonical going forward
        └── (artifacts — see §Files)
```

Directory naming:

- **`runs/2026-05-12/`** — historical exception. Pre-dates the suffix
  convention; kept as-is, never renamed.
- **`runs/YYYY-MM-DD-<pipeline_step>/`** — canonical for all runs
  after 2026-05-12, **including future Step 2 reruns**. `pipeline_step`
  is the exact string written to `manifest.pipeline_step`, e.g.
  `step2_supervised_discovery`, `step2_no_edge_found`,
  `step2b_dl_discovery_cnn`, `step2b_dl_discovery_lstm`,
  `step2b_dl_discovery_transformer`, `step2b_dl_discovery_hybrid`,
  `step2b_dl_discovery_ensemble`, `step3_clusters`,
  `step4_counterfactuals`.

Server-side cron parser:

```sh
cd /path/to/euieInvest-reports
git pull --ff-only
latest=$(ls runs/ | sort | tail -1)                   # most recent run, any pipeline_step
latest_xgb=$(ls runs/ | sort | grep -E 'step2_' | tail -1)
latest_dl=$(ls runs/ | sort | grep step2b_dl | tail -1)
./parse-run "runs/$latest/"
```

The date prefix keeps everything sortable; the suffix filters by
pipeline. The `runs/2026-05-12/` historical entry sorts before any
suffixed run on the same date, which is the natural ordering anyway.

## Files in each `runs/<dir>/`

### `manifest.json` (required, every run)

Run metadata so the server side can correlate to the consumer's code
state. Fields:

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | unique per run; convention `YYYY-MM-DD-NNN` |
| `train_end` | `YYYY-MM-DD` | inclusive |
| `val_end` | `YYYY-MM-DD` | inclusive |
| `holdout_end` | `YYYY-MM-DD` | inclusive |
| `model_sha` | string | content-hash of the trained model file (xgb json / pytorch state-dict) |
| `feature_count` | int | number of feature columns at training time (XGB) or input channels × window (DL) |
| `positive_rate_train` | float | empirical `is_winner` rate on the train slice |
| `holdout_precision_at_topdecile` | float | **the headline edge number** — fraction of true winners among the top 10% of holdout rows by predicted probability. Compare against `holdout_base_rate` for lift. Used to rank XGB vs DL variants. |
| `holdout_recall_at_topdecile` | float | fraction of all holdout winners captured by the top decile |
| `holdout_auc` | float | ROC-AUC on the holdout |
| `holdout_base_rate` | float | unconditional `is_winner` rate on holdout — denominator for the "edge vs base" judgement |
| `holdout_n_rows` | int | size of the holdout slice after dropping last-30-per-symbol nulls |
| `holdout_top_decile_k` | int | row count of the top decile (≈ `holdout_n_rows // 10`) |
| `top_per_day_k` | int | per-day K used in `top-per-day.parquet` (default 20; tunable per server-side dashboard ergonomics) |
| `universe_size` | int | distinct symbol count in `price_history` at run time |
| `git_commit_of_quant_repo` | string | full SHA of the `euieInvest-quant` HEAD when the run started |
| `pipeline_step` | string | which CLAUDE.md §5 step produced this run. See "Directory naming" above for the enumerated values. |
| `runtime_device` | string | device the fitted model actually ran on. XGB: from `booster.save_config()`. DL: from `next(model.parameters()).device`. `"cuda:0"` on heaven-pc; `"cpu"` would surface a silent CUDA fallback. |
| `train_wall_clock_s` | float | wall-clock seconds inside the model's `fit()` / training loop, excluding data prep. A 5090 trains XGB step 2 in single-digit seconds and a 1D-CNN in low-minutes; CPU fallback would push this 1-2 orders of magnitude up. |

DL-specific manifest fields (Step 2b and Phase A DL tracks):

| Field | Type | Notes |
|---|---|---|
| `architecture` | string | `"cnn"` / `"lstm"` / `"transformer"` / `"hybrid"` / `"ensemble"` / Phase-A-track-specific (e.g. `"foundation_transformer"`, `"protopnet"`, `"concept_bottleneck"`, `"vae"`) |
| `param_count` | int | total trainable parameters |
| `epochs_trained` | int | total epochs the training loop ran (including any epoch that triggered early stop). Distinct from `best_epoch` for early-stopped runs. |
| `best_epoch` | int | epoch the saved model state is from (the best val-metric epoch). Equal to `epochs_trained` when early stopping did not fire. |
| `mixed_precision` | bool | whether `torch.cuda.amp` was used during training |

### `top-decile.parquet` (required, every run)

Global top 10% of holdout rows by predicted probability — the
apples-to-apples cross-architecture comparison surface.

Columns: `symbol` (Utf8), `date` (Date), `predicted_proba` (Float64).
Sorted by `(date, predicted_proba DESC)`. Row count ≈ `holdout_n_rows // 10`,
exactly `holdout_top_decile_k`.

### `top-per-day.parquet` (required, every run)

Per-day top-K by predicted probability — the **dashboard surface**.
Sized for "today's picks" display; server-side reads this directly.

Columns:

| Column | Type | Notes |
|---|---|---|
| `symbol` | Utf8 | ticker |
| `date` | Date | holdout date |
| `predicted_proba` | Float64 | model output |
| `rank_within_day` | Int64 | 1-indexed rank; 1 = highest proba that day |

Sorted by `(date ASC, rank_within_day ASC)`. K (typically 20) is
written to `manifest.top_per_day_k`. Days with fewer than K eligible
rows include all available rows (no padding).

### `shap-summary.parquet` (required, every run)

Mean absolute feature attribution per the model family. The column
name `mean_abs_shap` is retained across pipelines for parser
compatibility — its **content** depends on the model:

- **XGB (Step 2)**: TreeSHAP via xgboost's native `pred_contribs`
- **DL (Step 2b)**: Captum `IntegratedGradients`, aggregated across
  timesteps by `sum(|attribution|)` per feature

Columns: `feature_name` (Utf8), `mean_abs_shap` (Float64),
`direction` (Utf8: `"+"` / `"-"` / `"mixed"`). Sorted by
`mean_abs_shap DESC`.

### `timestep-attribution.parquet` (required, Step 2b sequence models)

Window-relative per-timestep attribution — the "which days mattered"
view, only meaningful for sequence models (CNN/LSTM/Transformer). XGB
runs do not produce this file.

Columns: `feature_name` (Utf8), `timestep` (Int64),
`mean_abs_attribution` (Float64). `timestep` is window-relative:
`0` = day t-1 (most recent), `59` = day t-60 (oldest, assuming a
60-day window). Sorted by `(feature_name ASC, timestep ASC)`.

### `rules.parquet` (required for Track 1 / 4 / 5 — rule-extraction tracks)

Per-rule discovery output. Each row is one filtered conjunctive rule
of the form ``feature_X op threshold AND feature_Y op threshold AND
…`` that holds on the holdout at the precision/coverage/lift bar
specified in the brief.

| Column | Type | Notes |
|---|---|---|
| `rule_id` | Int64 | rank order, 0 = best by `lift × coverage` |
| `conditions_json` | Utf8 | JSON list of `{feature, op ("<" / ">="), threshold}` objects, sorted canonically (feature, op, threshold) so two trees with identically-shaped paths dedupe to the same rule |
| `n_conditions` | Int64 | conjunction length |
| `coverage_n` | Int64 | rows in holdout that satisfy all conditions |
| `coverage_pct` | Float64 | `100 × coverage_n / holdout_n_rows` |
| `precision` | Float64 | `winners / coverage_n` (fraction in `[0, 1]`) |
| `lift` | Float64 | `precision / holdout_base_rate` |
| `example_symbol_dates_json` | Utf8 | up to 10 example matching `(symbol, date)` pairs for human spot-checking |

Default filter (Track 1 brief): `lift ≥ 1.5 AND coverage_pct ≥ 0.5 AND
precision ≥ 0.35`. Filter values are echoed in
`manifest.filter_min_{lift, coverage_pct, precision}`.

Track-1 manifest fields beyond the common set (none of the DL ones apply):

| Field | Type | Notes |
|---|---|---|
| `source_model_run_id` | string | the predictor run whose model was walked (e.g. `"2026-05-12-001"`) |
| `source_model_sha` | string | sha256 of the walked model file — pins exactly which booster the rules came from |
| `n_paths_walked` | int | total root-to-leaf paths across all trees, pre-dedup |
| `n_rules_unique` | int | after canonical-ordering dedup, pre-filter |
| `n_rules_kept` | int | after filter, written to rules.parquet |

### `clusters.parquet` (required for Track 2 / 7)

Per-cluster signature output. Each row is one cluster from one
clustering configuration (algorithm × K).

| Column | Type | Notes |
|---|---|---|
| `algorithm` | Utf8 | `"kmeans"` / `"gmm"` / `"hdbscan"` |
| `k` | Int64 | `n_clusters` / `n_components` for KMeans/GMM; 0 for HDBSCAN (k auto-determined) |
| `cluster_id` | Int64 | per-algorithm, per-k cluster ordinal |
| `size` | Int64 | number of members in this cluster |
| `signature_features_json` | Utf8 | top features by `|cluster_mean_z|` (the z-score of the cluster centroid in population z-space); JSON list of `{feature, cluster_mean_z, direction}` |
| `example_symbol_dates_json` | Utf8 | up to 20 example `(symbol, date)` members |

### `cluster-membership.parquet` (required for Track 2 / 7)

Per-window cluster assignments. Used to cross-reference cluster
membership with predictions from other tracks.

| Column | Type | Notes |
|---|---|---|
| `symbol` | Utf8 | |
| `date` | Date | |
| `algorithm` | Utf8 | matches `clusters.parquet` |
| `k` | Int64 | matches `clusters.parquet` |
| `cluster_id` | Int64 | matches `clusters.parquet` |

### `stable-features.parquet` (required for Track 4)

Cross-label feature stability: which features appear in the top-K
rules of multiple labels. Features in 3+ labels are *structurally
stable* signals.

| Column | Type | Notes |
|---|---|---|
| `feature_name` | Utf8 | |
| `n_labels_top20` | Int64 | how many of the 5 labels' top-20 rules contain a condition on this feature |
| `label_ids_present` | Utf8 | comma-separated label-id list (e.g. `"L1,L2,L4"`) |

Track 4 also produces 5 per-label `rules_L{1..5}.parquet` files, each
with the same schema as Track 1's `rules.parquet` plus a `label_id`
column for join keys.

### `regime-stability.parquet` (required for Track 5)

Cross-regime lift comparison. Per (Track 1) rule, lift in each of
the 4 regime slices + durability flag.

| Column | Type | Notes |
|---|---|---|
| `source_rule_id` | Int64 | matches Track 1's `rules.parquet#rule_id` |
| `lift_bull` | Float64 (nullable) | rule's lift on bull regime; null if it didn't pass filter there |
| `lift_bear` | Float64 (nullable) | |
| `lift_chop` | Float64 (nullable) | |
| `lift_recovery` | Float64 (nullable) | |
| `n_regimes_durable` | Int64 | count of regimes where `lift >= min_lift` |
| `is_durable` | Boolean | `n_regimes_durable >= manifest.durable_min_regimes` |

Track 5 also produces 4 per-regime `rules-{bull,bear,chop,recovery}.parquet`
files (same schema as `rules.parquet` plus `source_rule_id`).

### `winner-deltas.parquet` (required for Track 6)

Per-feature mean delta from winner to its 5 nearest non-winners in
the 47-dim feature space. Sorted by `|z_delta|` descending — the
features that flip the outcome appear at the top.

| Column | Type | Notes |
|---|---|---|
| `feature_name` | Utf8 | |
| `rank` | Int64 | 0 = strongest discriminator |
| `mean_delta_winner_minus_nearest_losers` | Float64 | raw-units mean delta |
| `std_delta` | Float64 | spread across winners |
| `z_delta` | Float64 | mean_delta / population_std — cross-feature comparable |
| `direction` | Utf8 | `"+"` or `"-"` |

### `nearest-non-winners.parquet` (required for Track 6)

Per-winner: 5 nearest non-winner windows. For human spot-checking
(why is this winner not a loser?).

| Column | Type | Notes |
|---|---|---|
| `winner_symbol` | Utf8 | |
| `winner_date` | Date | |
| `neighbor_symbol` | Utf8 | |
| `neighbor_date` | Date | |
| `neighbor_rank` | Int64 | 1..k (k from `manifest.k_nearest`) |
| `distance` | Float64 | in the chosen metric (`manifest.metric`) |

### `winner-fingerprint.md` (required when Step 4 has run)

Human-readable verdict per CLAUDE.md §13's verdict format. Includes
the Phase 2 go/no-go gate result (§14). Not produced by Step 2 / 2b
runs — clustering + counterfactual analysis are prerequisites for a
meaningful fingerprint narrative.

## What this repo does NOT contain

- Training data (lives in the trading-platform's `price_history`,
  served via `/api/v1/ohlcv`)
- Model weights / pickled artifacts (lives in
  `euieInvest-quant/runs/<dir>/` on `heaven-pc`, gitignored, never
  published)
- Code (lives in `euieInvest-quant`)
- Logs / full training metrics (manifest captures the summary; full
  logs stay on `heaven-pc`)

## Write cadence

Per-run, ad-hoc. Each pipeline run that ships an edge claim (Step 2,
Step 2b, future Step 3/4) commits its own `runs/<dir>/`. No fixed
schedule.

Consumer commits via:

```sh
cd /path/to/euieInvest-reports
git pull --ff-only
mkdir -p "runs/$(date -I)-<pipeline_step>"
# ... write the required files for that pipeline_step ...
git add "runs/$(date -I)-<pipeline_step>/"
git commit -m "run: $(date -I) — <pipeline_step>"
git push origin main
```

Server-side cron picks it up on next pull.

## Permissions

Write from `lore-line/euieInvest-quant` (heaven-pc deploy key); read
from `lore-line/euieInvest` (claudehost cron). Pull-only on the
server side.
