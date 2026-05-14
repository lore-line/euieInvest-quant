# Session handoff — 2026-05-14 (after repo move out of Nextcloud)

## Why this file exists

The repo was moved from `D:\Nextcloud\LORELINE\CODE\euieInvestDeepLearn`
to `D:\repos\euieInvestDeepLearn` to eliminate the
`OSError: [Errno 22] Invalid argument: '/workspace/src'` crashes that
were stalling Track 8 (and would have hit Tracks 9-12 next).

Root cause: `pyproject.toml` installs `quant` as an **editable** package
via `uv sync`. That puts `/workspace/src` in Python's `sys.path` via a
`.pth` file in site-packages. Every Python invocation calls
`_fill_cache.os.listdir('/workspace/src')` during module discovery, and
Nextcloud's transient file-handles on the mounted source tree
intermittently trigger `OSError EINVAL`.

The fix is structural: source code is no longer in a cloud-sync directory.

## Phase A state at handoff

### Done
- Track 1 (xgb_rule_extraction) — 1,100 rules
- Track 2 (handcrafted_clustering)
- Track 4 (multi_label_rules)
- Track 5 (per_regime_rules) — 431 regime-durable rules after bull-fix
- Track 6 (classical_counterfactual)
- Track F (foundation_pretrain) — 50/50 epochs, 15.8h wall clock
- Track 7 v2 (embedding_clustering, k-means k=10 fallback) — **top cluster 41.1% winner rate vs 22.7% base rate (1.81× lift)**

All seven shipped to `D:/Nextcloud/LORELINE/CODE/euieInvest-reports/`
(reports repo stays in Nextcloud — small artifacts, write-once, no
Python imports).

### Open
- Track 8 (prototype_learning) — code OOM bug fixed (commit 27a0eb3), but
  blocked by the `/workspace/src` OSError. Re-launch after rebuild.
- Tracks 9, 10, 11, 12 — code complete, gated on Track 8.
- Synthesis v2 — gated on Tracks 7-12 landing.

### Findings during Track 7/8 saga (worth remembering)
- Track 7's HDBSCAN finds **0 clusters** on Track F encoder embeddings
  (no density gaps in the manifold). K-means k=10 fallback added in
  commit 4005865. K-means confirmed encoder is mode 2 (smooth manifold,
  strong label signal); top cluster 41.1% winner rate.
- Track 8 v1 trained successfully through 20 epochs (val_prec@TopDecile
  0.39-0.41) before OOM-killing on a 235 GB broadcast in the
  archetype-finding step. Fix landed in commit 27a0eb3.
- All bug fixes pushed to `lore-line/euieInvest-quant`. Most recent
  commit: `27a0eb3`.

## What needs to happen in the new session

### 1. Verify the move worked
```sh
cd D:\repos\euieInvestDeepLearn
git status              # should show clean working tree on main
git log --oneline -5    # most recent commit should be 27a0eb3
ls D:\quant-runs        # training output dir; unchanged location
```

### 2. Rebuild the Docker image
The image was built against the OLD path. Rebuild from the new location:
```sh
cd D:\repos\euieInvestDeepLearn
docker compose build
```

### 3. Re-launch Track 8 v2
```sh
.\scripts\ops\quant-start.ps1 -Track step3h_prototype_learning
docker logs --follow euieinvest-quant-step3h_prototype_learning
```

Expected: ~30 min total. Should see `epoch 1/20` ~1.5 min in,
all 20 epochs land in 20-30 min, then "finding archetypal window per
prototype" — should NOT OOM now (chunked distance computation),
should NOT crash on /workspace/src (no editable install pulling it).

### 4. Continue the chain: Tracks 8 → 9 → 10 → 11 → 12 → Synthesis v2

Monitor each track and ship to reports when each lands. PR #1 ack
per-track or batched per the prior cadence.

### 5. Restart the tray
```sh
.\scripts\ops\quant-tray.ps1
```

(Tray was stopped before the move to release any open file handles.)

## What was NOT moved

- `D:\Nextcloud\LORELINE\CODE\euieInvest-reports` — reports repo
  stays in Nextcloud. Small artifacts, write-once, no Python imports
  → no risk.
- `D:\quant-runs\` — training output dir, already outside Nextcloud
  since commit 4cf9ba6.

## Open questions for the new session

1. **Should Tracks 9-12 also expect `/workspace/src` issues?** No — the
   structural fix applies to all Python imports, not just Track 8.
2. **Server team comments on PR #1 since last ack?** Check
   [PR #1](https://github.com/lore-line/euieInvest-quant/pull/1) for
   anything new after issuecomment-4446332913 (the Track 7 comprehensive
   ack).
3. **Phase B and D briefs** ([issuecomment-4441437671](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4441437671),
   [issuecomment-4441445284](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4441445284))
   — still queued post-Phase-A. No action mid-chain.

## Latest commits at handoff time

```
27a0eb3 fix(track8): chunked pairwise distance for archetype-finding (OOM fix)
745be1f fix(track7): tolerate cloud-sync OSError on UMAP import + reduce restart retries
4005865 fix(track7): k-means fallback when HDBSCAN finds no density clusters
2afd338 fix(ops): restart policy on-failure:5 — clean exits no longer loop
f7e667d docs(theses): synthesis v1.1 — pairwise findings + Thesis 7
```
