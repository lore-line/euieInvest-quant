# `lore-line/euieInvest-reports` — repo layout

> **Status**: layout decided 2026-05-12, repo not yet created. Server team
> creates `lore-line/euieInvest-reports` with this structure as the seed
> commit. Linked discussion: PR #1, comment by `lore-line` 2026-05-12.

The euieInvest trading platform on `claudehost` consumes the quant
side's discovery output via a git-pull-and-parse cron. This is that
output's repo layout.

## Tree

```
euieInvest-reports/
├── README.md                            # human-readable repo intro
└── runs/
    └── YYYY-MM-DD/                      # ISO date of pipeline run
        ├── manifest.json                # see §Manifest below
        ├── winner-fingerprint.md        # human-readable, CLAUDE.md §13 format
        ├── top-decile.parquet           # (symbol, date, predicted_proba)
        ├── shap-summary.parquet         # (feature_name, mean_abs_shap, direction)
        └── clusters.parquet             # (symbol, date, cluster_id, distance_to_centroid)
                                         # winner-only rows
```

No `latest/` symlink — git tracks symlinks poorly and a cron-driven
`git pull` parses the most recent dated subdir anyway. Server-side
cron pattern:

```sh
cd /path/to/euieInvest-reports
git pull --ff-only
latest=$(ls runs/ | sort | tail -1)
./parse-run "runs/$latest/"
```

## Files in each `runs/YYYY-MM-DD/`

### `manifest.json` (required)

Run metadata so the server side can correlate to the consumer's code
state. Fields:

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | unique per run; convention `YYYY-MM-DD-NNN` |
| `train_end` | `YYYY-MM-DD` | inclusive |
| `val_end` | `YYYY-MM-DD` | inclusive |
| `holdout_end` | `YYYY-MM-DD` | inclusive |
| `model_sha` | string | content-hash of the trained xgb model file |
| `feature_count` | int | number of feature columns at training time |
| `positive_rate` | float | empirical `is_winner` rate on the train slice |
| `universe_size` | int | distinct symbol count in `price_history` at run time |
| `git_commit_of_quant_repo` | string | full SHA of the `euieInvest-quant` HEAD when the run started — lets the server bisect feature builds against report deltas |
| `pipeline_step` | string | which CLAUDE.md §5 step produced this run (e.g. `"step2_supervised_discovery"`, `"step4_counterfactuals"`) |

### `winner-fingerprint.md` (required)

Human-readable verdict per CLAUDE.md §13's verdict format. Includes
the Phase 2 go/no-go gate result (§14).

### `top-decile.parquet` (required)

Columns: `symbol` (Utf8), `date` (Date), `predicted_proba` (Float64).
Sorted by `(date, predicted_proba DESC)`. One row per top-decile
prediction on the holdout window.

### `shap-summary.parquet` (required)

Columns: `feature_name` (Utf8), `mean_abs_shap` (Float64),
`direction` (Utf8: `"+"` / `"-"` / `"mixed"`). Sorted by
`mean_abs_shap DESC`.

### `clusters.parquet` (required when Step 3 has run)

Columns: `symbol` (Utf8), `date` (Date), `cluster_id` (Int64),
`distance_to_centroid` (Float64). Winner-only rows. `cluster_id`
matches `manifest.cluster_labels` (added to manifest when
`pipeline_step >= "step3"`).

## What this repo does NOT contain

- Training data (lives in the trading-platform's `price_history`,
  served via `/api/v1/ohlcv`)
- Model weights / pickled artifacts (lives in
  `euieInvest-quant/data/models/` on `heaven-pc`, never published)
- Code (lives in `euieInvest-quant`)
- Logs / training metrics (manifest captures the summary; full logs
  stay on `heaven-pc`)

## Write cadence

When discovery Step 4 (clustering + reporting) lands on the consumer
side. Estimated arrival: several sessions out. First commit can be
seeded now as just the README + empty `runs/` directory.

## Permissions

Write-from-`euieInvest-quant`-consumer, read-from-`euieInvest`-server.
Either via push from `heaven-pc` (deploy key, same pattern as the
quant repo) or via the consumer's existing `lore-line` auth.

## When the first real `runs/<date>/` appears

The consumer side commits via:

```sh
cd /path/to/euieInvest-reports
git pull --ff-only
mkdir -p "runs/$(date -I)"
# ... write the five files ...
git add "runs/$(date -I)/"
git commit -m "run: $(date -I) — <pipeline_step>"
git push origin main
```

Server-side cron picks it up on next pull.
