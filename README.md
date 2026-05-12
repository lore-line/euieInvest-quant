# euieInvest-quant

ML/quant research platform — **outlier-breakout pattern discovery** on the
Russell 2000 ∪ watchlist universe.

## Mission

Discover the **pre-breakout fingerprint** of equities that run +20% in 30
trading days. Discovery, not validation — let the data tell us what
winners share, not what a hand-crafted doctrine predicts.

A previous "Tier-3 anomaly" doctrine on this exact dataset produced near-zero
alpha (+0.4% across 641 flags). It is treated here as a **baseline cohort to
beat**, not as a feature prior.

## Quickstart — Docker (preferred)

Prereqs: Docker Desktop + NVIDIA Container Toolkit + any CUDA 12.8+ capable
GPU. Verified on **heaven-pc** with RTX 5090 (Blackwell sm_120).

```sh
# 1. Pull the snapshot
# Preferred (once the data API is live on claudehost):
EUIEINVEST_API_BASE_URL=http://100.68.86.56:8443 \
  docker compose run --rm dev python scripts/pull-via-api.py

# Legacy fallback (rsync the SQLite file directly):
bash   scripts/pull-snapshot.sh        # WSL2 / Linux / Git Bash
.\scripts\pull-snapshot.ps1            # native PowerShell

# 2. Build the image
docker compose build

# 3. Verify tests + GPU
docker compose run --rm test
docker compose run --rm dev nvidia-smi

# 4. Run the discovery pipeline
docker compose run --rm discover       # raises NotImplementedError until
                                        # the feature modules are filled in
```

The loader prefers parquet/JSON files in `data/snapshots/` (written by
`pull-via-api.py`) and transparently falls back to the legacy
`euieinvest.db` SQLite snapshot if those aren't present.

## Docs site (local)

The API contract and server implementation guide render as a small
MkDocs Material site:

```sh
docker compose up docs
# then visit http://localhost:8000
```

Live-reload while editing files under `docs/`. The site is **local
only** — it binds to `127.0.0.1:8000` on the host and is not exposed
to the network.

## Native fallback (no Docker)

Prereqs: Python 3.11 and [uv](https://docs.astral.sh/uv/). CUDA 12.8+ on
the host is required for GPU-backed xgboost; CPU works for Phase 1 tests.

```sh
uv sync --extra dev
uv run pytest
uv run python scripts/discover.py
```

## Architecture

```
                ┌────────────────────────────┐
                │  claudehost (Tailscale)    │
                │  euieInvest trading prod   │
                │  SQLite snapshot:          │
                │  data/euieinvest.db.bak    │
                └──────────────┬─────────────┘
                               │  rsync / scp
                               ▼
                ┌────────────────────────────┐
                │  heaven-pc (RTX 5090)      │
                │  data/snapshots/*.db       │
                │  ┌──────────────────────┐  │
                │  │  Docker (CUDA 12.8)  │  │
                │  │  src/quant/          │  │
                │  │  scripts/discover.py │  │
                │  └──────────────────────┘  │
                │  reports/*.md, *.png       │
                └────────────────────────────┘
```

## File layout

| Path                                | Role                                              |
|-------------------------------------|---------------------------------------------------|
| `src/quant/data/loader.py`          | Read-only SQLite snapshot accessors               |
| `src/quant/labels.py`               | Forward-looking +20%/30d winner labels            |
| `src/quant/backtest/temporal.py`    | Strict train/val/holdout splits by date           |
| `src/quant/features/*.py`           | 7 feature modules (Phase 1: scaffold only)        |
| `src/quant/models/xgb_discovery.py` | XGBoost + SHAP discovery (scaffold)               |
| `src/quant/clusters/winners.py`     | KMeans winner clustering (scaffold)               |
| `scripts/discover.py`               | 5-step pipeline orchestrator (scaffold)           |
| `scripts/pull-snapshot.{sh,ps1}`    | Snapshot rsync/scp from claudehost                |
| `tests/`                            | pytest — loader, labels, temporal split           |
| `CLAUDE.md`                         | **Load-bearing project brief — read first**       |

## Status

Phase 1 scaffold. Loader, labels, and temporal split are real with passing
tests. Feature modules, XGBDiscovery, clustering, and the `discover.py`
pipeline raise `NotImplementedError` with actionable messages.

**Next session's first task**: implement the 7 feature modules per
`CLAUDE.md` §7.

## Snapshot path overrides

The loader resolves the snapshot at `data/snapshots/euieinvest.db` by
default. Override with the `EUIEINVEST_SNAPSHOT` environment variable
(used by the test suite to point at synthetic SQLite fixtures).
