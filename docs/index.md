# euieInvest-quant

ML/quant research platform — **outlier-breakout pattern discovery**.

Compute runs on `heaven-pc` (RTX 5090) and consumes source data from
the euieInvest trading platform on `claudehost` via an HTTP API over
Tailscale. This site documents the API contract that both sides build
against.

---

## For the server team (claudehost)

You're building the HTTP server that satisfies the contract below.

1. **[API contract (v1)](api-contract.md)** — read this first. The
   wire-format truth: endpoints, parquet schemas, query parameters,
   error format, versioning policy.
2. **[Server implementation guide](server-implementation-guide.md)** —
   how to satisfy the contract. FastAPI skeletons per endpoint,
   deployment notes for claudehost, acceptance criteria, contract test
   suite.

When your dev server passes the contract tests, you're done:

```sh
git clone https://github.com/lore-line/euieInvest-quant.git
cd euieInvest-quant
uv sync --extra dev
EUIEINVEST_API_BASE_URL=http://localhost:8443 uv run pytest tests/contract/ -v
```

## For the consumer team (heaven-pc — this repo)

- **Project brief**: [`CLAUDE.md`](https://github.com/lore-line/euieInvest-quant/blob/main/CLAUDE.md)
  — mission, hardware, methodology, validation discipline, Phase 2 gate
- **API client**: `src/quant/data/api_client.py` (typed httpx client
  against the contract above)
- **Pull script**: `scripts/pull-via-api.py` (refreshes the local
  parquet cache from the API)
- **Loader**: `src/quant/data/loader.py` (reads the local parquet
  cache; transparently falls back to a legacy SQLite snapshot during
  cutover)

## Architecture at a glance

```
                       ┌──────────────────────────────────┐
   SOURCE DATA         │  claudehost                      │
   ─────────────       │  ─────────                       │
   trading platform    │  • runs the trading platform     │
   writes to DB        │  • DB lives here                 │
                       │  • serves /api/v1/* (parquet)    │
                       └─────────────┬────────────────────┘
                                     │  HTTP over Tailscale
                                     │
                                     ▼
   COMPUTE             ┌──────────────────────────────────┐
   ─────────────       │  heaven-pc                       │
   model training,     │  ───────                         │
   SHAP, clustering    │  • RTX 5090                      │
                       │  • CONSUMES the API              │
                       │  • runs discovery pipeline       │
                       │  • writes report files           │
                       └──────────────────────────────────┘
```

The trust boundary is **Tailscale**. The server binds to its Tailscale
IP, not `0.0.0.0`. No bearer token in v1.

## Status

- API contract: drafted (PR #1) — not yet implemented on the server
  side
- Server implementation: not started (server team's work, separate
  repo)
- Client side: implemented and shipped (PRs #2–#6)
- Discovery pipeline step 1 (feature engineering): implemented and
  smoke-tested end-to-end on heaven-pc (PR #7)
- Discovery steps 2–5 (XGBoost + SHAP, clustering, counterfactuals,
  tier-3 comparison): scaffolds remain. Next session.

See the [GitHub repo](https://github.com/lore-line/euieInvest-quant)
for the full PR ladder and codebase.
