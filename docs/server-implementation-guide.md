# euieInvest data API v1 — server implementation guide

> **Audience**: the team building the HTTP server side of the
> euieInvest data API on `claudehost`.
>
> **Read first**: [`docs/api-contract.md`](api-contract.md). That document
> is the canonical wire-format spec. This guide tells you HOW to satisfy
> it.
>
> **Owner**: euieInvest-quant consumer team (heaven-pc, RTX 5090).
>
> **Repo where the runnable contract tests live**:
> [`lore-line/euieInvest-quant`](https://github.com/lore-line/euieInvest-quant)
> — `tests/contract/test_api_contract.py`.

---

## 0. Quick start

1. Read [`docs/api-contract.md`](api-contract.md) end-to-end (~10 min).
2. Stand up a FastAPI app from the templates in §3.
3. Wire each endpoint to the trading-platform DB.
4. Run the contract test suite (§5) against your local dev server.
5. Bind the production listener to claudehost's Tailscale IP and ship.

## 1. Acceptance criteria — the "done" definition

The server is acceptable when **all** of the following hold:

- [ ] `GET /api/v1/health` — 200 with the schema in contract §5.1
- [ ] `GET /api/v1/snapshot-cursor` — 200 with the schema in §5.2
- [ ] `GET /api/v1/ohlcv` — valid parquet body with the columns and
      types in §5.3; supports `since`, `until`, `symbols`, `format`
      query params
- [ ] `GET /api/v1/peer-groups` — 200 JSON `{group: [symbol, ...]}`
- [ ] `GET /api/v1/anomaly-flags` — 200 parquet with the 22 columns in
      §5.5; supports `since`, `status`, `format`
- [ ] Errors use `application/problem+json` per contract §4.4
- [ ] Listener bound to claudehost's Tailscale IP, **not** `0.0.0.0`
- [ ] Contract test suite passes:
      `EUIEINVEST_API_BASE_URL=http://<your-IP>:<port> uv run pytest tests/contract/`
      (clone the consumer repo; `uv sync --extra dev` once)
- [ ] End-to-end smoke: consumer's `pull-via-api.py` runs to completion
      against the live server and writes valid parquet files

## 2. Recommended stack

**FastAPI + uvicorn + polars + pyarrow** — lowest friction Python path.
The consumer uses polars/pyarrow already, so picking the same gives you
identical schema handling.

> ⚠️ **`polars-lts-cpu` on hosts without AVX2.** The default `polars`
> wheels on PyPI assume an AVX2-capable CPU; on older / virtualized
> hosts (e.g. claudehost is a Proxmox VM without AVX2 passed through)
> the process crashes at first import with `Illegal instruction`. Same
> Python module name (`polars`), just compiled for the older baseline.
> If you hit this, pin `polars-lts-cpu` instead of `polars` in your
> server's requirements. The consumer side runs in a CUDA Docker image
> where AVX2 is always present, so the consumer keeps vanilla `polars`.

Other stacks (Go, Node, Rust) are fine if you have one already wired
into the trading platform. The contract is language-agnostic; only the
wire format matters. The contract test suite is the source of truth —
if those tests pass against your impl, the language doesn't matter.

## 3. Endpoint implementations

### 3.1 App skeleton

```python
from datetime import date, datetime, timezone
from io import BytesIO
from typing import Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

SERVICE_VERSION = "2026.05.12"  # bump per server release
app = FastAPI(title="euieInvest data API", version="1.0.0")
```

### 3.2 `GET /api/v1/health`

```python
@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "euieInvest",
        "service_version": SERVICE_VERSION,
        "api_version": "1",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
```

### 3.3 `GET /api/v1/snapshot-cursor`

```python
@app.get("/api/v1/snapshot-cursor")
def snapshot_cursor() -> dict:
    return {
        "ohlcv": {
            "max_date": db.scalar("SELECT MAX(date) FROM price_history"),
            "row_count": db.scalar("SELECT COUNT(*) FROM price_history"),
            "symbol_count": db.scalar(
                "SELECT COUNT(DISTINCT symbol) FROM price_history"
            ),
        },
        "anomaly_flags": {
            "max_id": db.scalar("SELECT COALESCE(MAX(id), 0) FROM anomaly_flags"),
            "max_flag_date": db.scalar("SELECT MAX(flag_date) FROM anomaly_flags"),
            "row_count": db.scalar("SELECT COUNT(*) FROM anomaly_flags"),
        },
        "peer_groups": {
            "hash": peer_groups_hash(),
            "group_count": db.scalar(
                "SELECT COUNT(DISTINCT group_name) FROM peer_groups"
            ),
            "entry_count": db.scalar("SELECT COUNT(*) FROM peer_groups"),
        },
    }
```

### 3.4 `GET /api/v1/ohlcv`

```python
_MAX_SYMBOLS = 5000

@app.get("/api/v1/ohlcv")
def ohlcv(
    since: date | None = None,
    until: date | None = None,
    symbols: str | None = None,
    format: Literal["parquet", "ndjson"] = "parquet",
) -> Response:
    where, params = [], []
    if since:
        where.append("date >= ?")
        params.append(since.isoformat())
    if until:
        where.append("date <= ?")
        params.append(until.isoformat())
    if symbols:
        symbol_list = symbols.split(",")
        if len(symbol_list) > _MAX_SYMBOLS:
            raise HTTPException(
                status_code=400,
                detail=f"symbols list exceeds max of {_MAX_SYMBOLS}",
            )
        where.append(
            f"symbol IN ({','.join('?' * len(symbol_list))})"
        )
        params.extend(symbol_list)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.execute(
        f"SELECT symbol, date, close, high, low, volume "
        f"FROM price_history{clause}",
        params,
    ).fetchall()

    df = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "date": pl.Utf8,
            "close": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "volume": pl.Int64,
        },
        orient="row",
    ).with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )

    if format == "parquet":
        buf = BytesIO()
        df.write_parquet(buf)
        return Response(
            buf.getvalue(),
            media_type="application/vnd.apache.parquet",
        )
    return Response(df.write_ndjson(), media_type="application/x-ndjson")
```

### 3.5 `GET /api/v1/peer-groups`

```python
@app.get("/api/v1/peer-groups")
def peer_groups() -> dict[str, list[str]]:
    rows = db.execute(
        "SELECT group_name, symbol FROM peer_groups "
        "ORDER BY group_name, symbol"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for group, symbol in rows:
        out.setdefault(group, []).append(symbol)
    return out
```

### 3.6 `GET /api/v1/anomaly-flags`

Same pattern as `/ohlcv`. Use `SELECT *` so columns added in the future
flow through automatically (additive evolution, contract §2).

### 3.7 `peer_groups_hash` helper

```python
import hashlib
import json

def peer_groups_hash() -> str:
    """Canonical hash matching contract §5.2."""
    g = peer_groups()
    g = {k: sorted(v) for k, v in g.items()}
    body = json.dumps(g, sort_keys=True)   # NB: DEFAULT separators, not compact
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()
```

> ⚠️ **Hash canonicalization gotcha.** Use `json.dumps(..., sort_keys=True)`
> with **default separators** (`, ` and `: `). The first server-side
> implementation passed `separators=(',', ':')` for compact output and the
> hash silently diverged from the consumer's expectation — contract test 8
> (`test_peer_groups_hash_in_cursor_matches_canonicalization`) caught it.
> If you re-implement the hash on either side, do NOT pass `separators=`.

### 3.8 Error handler (RFC 7807 problem+json)

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(HTTPException)
async def problem_json_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://euieinvest.invalid/errors/{exc.status_code}",
            "title": str(exc.detail)[:60],
            "status": exc.status_code,
            "detail": str(exc.detail),
            "instance": request.url.path,
        },
        media_type="application/problem+json",
    )

@app.exception_handler(RequestValidationError)
async def validation_problem_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=400,
        content={
            "type": "https://euieinvest.invalid/errors/400",
            "title": "Invalid request parameters",
            "status": 400,
            "detail": exc.errors(),
            "instance": request.url.path,
        },
        media_type="application/problem+json",
    )
```

## 4. Deployment on claudehost

### 4.1 Binding (this is the security model)

```sh
uvicorn euieinvest_api:app \
    --host 100.68.86.56 \      # the Tailscale IP of claudehost
    --port 8443 \
    --workers 1                # raise if you measure CPU saturation
```

Confirm with `ss -tlnp | grep :8443` — source IP must be `100.68.86.56`,
not `0.0.0.0` or `127.0.0.1`. Tailscale's WireGuard transport is the
encryption layer and the access-control layer.

### 4.2 systemd unit

```ini
[Unit]
Description=euieInvest data API
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/uvicorn euieinvest_api:app \
    --host 100.68.86.56 --port 8443
Restart=on-failure
RestartSec=5
User=euie
WorkingDirectory=/home/euie/euieInvest

[Install]
WantedBy=multi-user.target
```

### 4.3 TLS — optional in v1

Tailscale's WireGuard already encrypts the wire end-to-end between
claudehost and heaven-pc, so HTTP-on-Tailscale is acceptable. If you
want belt-and-suspenders:

```sh
tailscale cert claudehost.tail7a1fa.ts.net
# then uvicorn ... --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

### 4.4 Logging

JSON access logs with at minimum: `method`, `path`, `status`,
`response_bytes`, `duration_ms`. The consumer team will tail these when
diagnosing failed fetches.

## 5. Contract test suite

Maintained in the consumer repo at `tests/contract/test_api_contract.py`.
They are **skipped** unless `EUIEINVEST_API_BASE_URL` is set, so they
don't run in normal CI but DO run on demand against a live server.

To validate your local dev server:

```sh
git clone https://github.com/lore-line/euieInvest-quant.git
cd euieInvest-quant
uv sync --extra dev
EUIEINVEST_API_BASE_URL=http://localhost:8443 uv run pytest tests/contract/ -v
```

To validate the deployed server from heaven-pc:

```sh
EUIEINVEST_API_BASE_URL=http://100.68.86.56:8443 \
    docker compose run --rm dev uv run pytest tests/contract/ -v
```

All tests must pass before declaring the server done. They cover:

- All five endpoints return 200 with valid bodies
- Parquet columns and dtypes match the contract exactly
- `since=` and `symbols=` query params filter correctly
- The 5000-symbol cap returns `400` problem+json
- Cursor consistency: `cursor.ohlcv.max_date == max(ohlcv.date)`
- `peer_groups.hash` is reproducible and matches the canonicalization

## 6. Out of scope for v1

- **Model-serving / inference API** — not your concern. Separate service
  that will run on heaven-pc *if and when* discovery clears its Phase 2
  go/no-go gate.
- **TLS** — optional (see §4.3)
- **Bearer auth** — defer until the tailnet grows beyond `claudehost`
  ↔ `heaven-pc`
- **Pagination** — full dataset fits in memory (~120 MB raw OHLCV).
  Revisit at v2 if row count > 100M.
- **Write endpoints** — read-only API in v1. No PUT / POST / PATCH / DELETE.
- **Webhooks / SSE / streaming** — clients poll the cursor endpoint to
  decide refetch cadence. No push.

## 7. Open questions to confirm with the consumer team

Carried over from PR #1's open questions. Resolve before shipping:

1. **`peer_groups` schema** — is the prod schema still
   `(group_name, symbol)`, or has it grown columns? If it grew, decide
   whether to expose new columns or keep the API shape stable.
2. **`anomaly_flags` schema stability** — the 22-col schema in the
   contract was captured 2026-05-12. If the prod DB has added or
   renamed columns since, raise it before shipping — the contract's
   "additive evolution" rule allows new columns but not renames.
3. **TLS** — required in the contract, or "leave to server impl's
   judgment"? Default is the latter; flip if you'd rather mandate.

Raise these as comments on
[PR #1](https://github.com/lore-line/euieInvest-quant/pull/1) or as
issues in your own repo.

## 8. Performance expectations

| Endpoint | Expected p95 | Notes |
|---|---|---|
| `/health` | < 10 ms | trivial |
| `/snapshot-cursor` | < 100 ms | 4-5 indexed SELECTs |
| `/ohlcv` (full) | < 30 s over Tailscale | ~40 MB parquet for 2.4M rows |
| `/ohlcv` (`since=` last 30 days) | < 5 s | ~1 MB parquet |
| `/peer-groups` | < 50 ms | < 1 KB JSON |
| `/anomaly-flags` (full) | < 1 s | tiny table |

If `/ohlcv` full-pull is > 30 s consistently, raise it — we'll add
pagination to the contract before shipping.

## 9. Versioning policy reminder

- **Additive** (new endpoint, new optional query param, new column in
  a parquet response): non-breaking, stay on `/api/v1/`. Clients select
  columns by name and ignore extras.
- **Breaking** (rename column, change type, remove endpoint or column,
  change semantics of a status code): mint `/api/v2/`, deprecate v1
  with ≥ 90-day window.

When in doubt, lean conservative — bumping to v2 is cheap; breaking
existing clients is expensive.

## 10. Glossary

- **Cursor**: server-reported high-water marks per table. Clients call
  `/snapshot-cursor` and compare to their local cursor to decide whether
  a fetch is needed.
- **`since=` incremental fetch**: client supplies a date / datetime;
  server returns only rows with `date >= since`. The mechanism by which
  routine syncs avoid re-downloading the full table.
- **Tailscale-bound**: listener binds to the Tailscale interface IP, not
  `0.0.0.0`. The access-control mechanism, at the network layer.

## Changelog

| Date | Change |
|---|---|
| 2026-05-12 | Initial hand-off (this document) + contract v1.0.0-draft |
