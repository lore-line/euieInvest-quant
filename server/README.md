# `server/` — euieInvest data API (FastAPI)

Canonical server-side implementation of the contract in
[`docs/api-contract.md`](../docs/api-contract.md). This is the code that
answers the contract tests in [`tests/contract/`](../tests/contract/).

> **Where it runs:** claudehost (`100.68.86.56:8443`) via systemd.
> The trading platform that owns the source DB lives at
> `/home/euie/nextcloud/CODE/euieInvest/` on that host. This `server/`
> directory is a vendored copy for code review and contract-impl colocation
> — the deployed copy is in the trading-platform repo.

## What it does

Five read-only HTTP GET endpoints under `/api/v1/`:

| Path | Format | Notes |
|---|---|---|
| `/health` | JSON | liveness probe |
| `/snapshot-cursor` | JSON | high-water marks per table |
| `/ohlcv` | parquet (default) / ndjson | `since=`, `until=`, `symbols=`, `format=` |
| `/peer-groups` | JSON | full `{group: [symbol, ...]}` mapping |
| `/anomaly-flags` | parquet / ndjson | `since=`, `status=`, `format=` |

Errors are RFC 7807 `application/problem+json`. No bearer auth in v1 —
Tailscale ACL is the access boundary.

## Stack

- FastAPI + uvicorn
- polars (or polars-lts-cpu on hosts without AVX2 — see below)
- pyarrow (parquet writer)
- sqlite3 stdlib, opened read-only with `mode=ro`

## Local development

The default DB path resolves relative to this checkout, which won't have
the live trading DB. Set `EUIEINVEST_DB_PATH` to point at a working SQLite
file with the same schema (`price_history`, `peer_groups`, `anomaly_flags`):

```sh
cd server
python3 -m venv --without-pip .venv

# Bootstrap pip into the venv from system pip:
PYTHONPATH=$(python3 -c "import site; print(site.getusersitepackages())") \
  .venv/bin/python -m pip install --ignore-installed \
    'fastapi>=0.115.0' 'uvicorn[standard]>=0.32.0' \
    'polars-lts-cpu>=1.12.0' 'pyarrow>=18.0.0' \
    'httpx>=0.27.0' 'pytest>=8.0.0'

EUIEINVEST_DB_PATH=/path/to/your/euieinvest.db \
  PYTHONPATH=src \
  .venv/bin/python -m uvicorn euieinvest_api.app:app \
    --host 127.0.0.1 --port 8443 --reload
```

> **Why `polars-lts-cpu`?** claudehost (and many CI / older VMs) lacks
> AVX/AVX2. Vanilla `polars` wheels crash with `Illegal instruction` on
> those hosts. `polars-lts-cpu` is the same Python module (`polars`),
> compiled for older CPUs.

## Running the contract tests locally

From the repo root (not `server/`):

```sh
# In one terminal, start the server (see above).
# In another:
EUIEINVEST_API_BASE_URL=http://127.0.0.1:8443 \
  uv run pytest tests/contract/ -v
```

Expected: `11 passed`.

## Deployment on claudehost

Two systemd unit options ship in [`deploy/`](deploy/):

| File | Scope | When to use |
|---|---|---|
| `euieinvest-quant-api.service` | system-wide (`/etc/systemd/system/`) | Production. Includes `User=`, `Group=`, `ProtectSystem=strict`, `ReadOnlyPaths=` hardening. |
| `euieinvest-quant-api-user.service` | user-level (`~/.config/systemd/user/`) | Initial bring-up or operator-managed deployments. Survives reboots once `loginctl enable-linger` is set. |

System-wide install:

```sh
sudo cp deploy/euieinvest-quant-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now euieinvest-quant-api

# Verify the listener bound to the Tailscale IP, not 0.0.0.0:
ss -tlnp | grep :8443
```

User-level install (current claudehost deployment):

```sh
mkdir -p ~/.config/systemd/user
cp deploy/euieinvest-quant-api-user.service ~/.config/systemd/user/euieinvest-quant-api.service
systemctl --user daemon-reload
systemctl --user enable --now euieinvest-quant-api
sudo loginctl enable-linger $USER   # so it survives reboots
```

Edit the `WorkingDirectory=`, `Environment=PYTHONPATH=`, and `ExecStart=`
paths in the unit file to match your install location.

## Files

```
server/
├── README.md                    # this file
├── pyproject.toml               # FastAPI + polars-lts-cpu pins
├── src/euieinvest_api/
│   ├── __init__.py
│   ├── app.py                   # 5 routes + RFC 7807 error handler
│   ├── db.py                    # read-only sqlite3 helper
│   └── peer_hash.py             # canonical sha256 helper (matches contract §5.2)
├── tests/test_smoke.py          # local TestClient smoke tests
└── deploy/
    ├── euieinvest-quant-api.service        # system-wide systemd unit
    └── euieinvest-quant-api-user.service   # user-level systemd unit
```

## Gotchas pinned from real deployments

- **`peer_groups.hash` canonicalization** uses Python's *default* JSON
  separators (`, ` and `: `), not compact (`,` and `:`). The contract test
  in `tests/contract/test_api_contract.py` validates this; if you ever
  refactor `peer_hash.py`, do not add `separators=(",", ":")`.
- **`anomaly_flags.status` enum** in the live DB is
  `{active, invalidated, entered, exited, dismissed}` — the contract
  reflects this as of PR #1's correction commit (the original draft said
  `{open, ...}`).
- **`price_history` is consistently split-adjusted** (Yahoo `chart()`
  `close`). The migration that added `open` and `close_adj` columns is
  in the trading-platform repo (commit `8d8894c` in `lore-line/euieInvest`),
  not here.
- **SQLite WAL mode** allows concurrent readers without blocking the
  writer. Open the DB with `mode=ro` (already done in `db.py`).
