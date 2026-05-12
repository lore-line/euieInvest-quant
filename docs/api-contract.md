# euieInvest data API — v1 contract

This document is the **single source of truth** for the HTTP API that
`euieInvest` (on `claudehost`) exposes to consumers of its market-data
snapshot — most notably the `euieInvest-quant` ML/quant repo on `heaven-pc`.

The contract is consumer-driven: it's defined here, in the consumer repo,
because the consumer's needs (column types, filter dimensions, wire
format, latency profile) drive the design. `euieInvest` implements its
HTTP server to match.

> Status: **draft — not yet implemented on server side.** While `euieInvest`
> builds out these endpoints, this repo's `scripts/pull-snapshot.{sh,ps1}`
> rsync path remains the operational mechanism. Cut over to the API when
> `GET /api/v1/health` returns `200` on the production tailnet.

---

## 1. Design principles

1. **Polars-native on both sides.** Tabular endpoints return Apache
   Parquet by default — it round-trips zero-copy through pyarrow, gives
   us a typed schema, and is ~10× smaller than NDJSON on this dataset.
2. **Tailscale is the trust boundary.** The listener binds to the
   Tailscale IP, not `0.0.0.0`. No bearer token in v1 — add one only when
   the tailnet grows beyond `heaven-pc ↔ claudehost`.
3. **Incremental by default.** Tabular endpoints accept `since=`. Clients
   keep a local cursor and only ask for new rows. Full table downloads
   are supported but discouraged for routine syncs.
4. **Cursor-before-fetch.** `GET /api/v1/snapshot-cursor` is cheap and
   lets the client decide whether a fetch is needed at all.
5. **Additive evolution, breaking changes get a new prefix.** Adding a
   column or a new optional query parameter is non-breaking. Removing or
   renaming columns, changing types, or removing endpoints means
   `/api/v2/`.
6. **Errors are RFC 7807 problem+json.** Machine-parseable, human-readable.

## 2. Base URL & versioning

```
https://<claudehost-tailscale-ip>:<port>/api/v1/
```

In v1, both `<claudehost-tailscale-ip>` and `<port>` are configured
client-side via `EUIEINVEST_API_BASE_URL` (e.g.
`https://100.68.86.56:8443`). TLS is recommended but not strictly
required on a Tailscale-only listener — Tailscale's WireGuard transport
already encrypts the wire. If TLS is omitted, use `http://`.

Version policy:

| Change | Treatment |
|---|---|
| New endpoint | non-breaking, keep `/v1` |
| New optional query param | non-breaking |
| New column in a parquet response | non-breaking (client must select columns by name, not position) |
| Change column type, rename column, remove column | **breaking** → `/v2` |
| Change response status semantics | **breaking** → `/v2` |
| Remove an endpoint | **breaking** → `/v2` (and ≥ 90 day deprecation window) |

## 3. Auth model

**v1: Tailscale-only listener, no token.**

- Server binds to its Tailscale IP. Connections from any other interface
  are not accepted at the network layer.
- No `Authorization` header expected. Servers MAY return `401` if one is
  sent in v1 but SHOULD ignore it.

**Forward-compatibility hook**: if/when a bearer token is added in a
future minor version, it'll be `Authorization: Bearer <token>` validated
against a server-side env var. The contract doesn't need to change to
accommodate this — clients that send the header before it's required get
ignored; clients that don't send it after it's required get `401`.

## 4. Common conventions

### 4.1 Content negotiation

Tabular endpoints support a `format` query parameter:

| `format=` | Content-Type | Notes |
|---|---|---|
| `parquet` (default) | `application/vnd.apache.parquet` | recommended; native to polars |
| `ndjson` | `application/x-ndjson` | one JSON object per line; useful for `curl` inspection |

If `format` is omitted, the server returns parquet. If `format` is set to
an unknown value, return `400` with a problem document listing the
supported values.

### 4.2 Date and time formats

- **Dates** (e.g. trading dates): ISO 8601 calendar date, `YYYY-MM-DD`,
  no time component. In parquet, use the `date32` logical type. In
  NDJSON, use the string form.
- **Timestamps** (e.g. flag fired_at): ISO 8601 with UTC offset,
  `YYYY-MM-DDTHH:MM:SSZ`. In parquet, use the `timestamp[us, tz=UTC]`
  logical type.

### 4.3 Symbol filter syntax

The `symbols=` query parameter takes a comma-separated list of ticker
symbols, e.g. `symbols=AAPL,MSFT,GOOG`. Case-sensitive. Maximum 5,000
symbols per request — return `400` if exceeded.

### 4.4 Error format

All non-2xx responses use `application/problem+json` per RFC 7807:

```json
{
  "type": "https://euieinvest.invalid/errors/<slug>",
  "title": "<short human-readable summary>",
  "status": 400,
  "detail": "<longer explanation>",
  "instance": "/api/v1/ohlcv"
}
```

| Status | Meaning |
|---|---|
| 400 | Malformed request (bad date, unknown format, too many symbols) |
| 401 | Auth required (not used in v1) |
| 404 | Endpoint not found |
| 500 | Server error |
| 503 | Transient unavailability (e.g., backing DB unreachable). Clients SHOULD retry with exponential backoff. |

---

## 5. Endpoints

### 5.1 `GET /api/v1/health`

Liveness probe. Cheap. Used by clients to verify reachability before
issuing a real fetch.

**Request**: no parameters.

**Response 200** (`application/json`):

```json
{
  "status": "ok",
  "service": "euieInvest",
  "service_version": "2026.05.12",
  "api_version": "1",
  "as_of": "2026-05-12T01:23:45Z"
}
```

Fields:

| Field | Type | Notes |
|---|---|---|
| `status` | string | always `"ok"` when 200. `"degraded"` permitted if some features are unavailable but the API is up. |
| `service` | string | always `"euieInvest"` |
| `service_version` | string | server's own version (semver or date-stamp; freeform) |
| `api_version` | string | always `"1"` for this contract |
| `as_of` | ISO 8601 timestamp UTC | server-side wall clock at the time of the response |

### 5.2 `GET /api/v1/snapshot-cursor`

Returns the high-water marks of each table. Clients call this first to
decide whether to refetch.

**Request**: no parameters.

**Response 200** (`application/json`):

```json
{
  "ohlcv": {
    "max_date": "2026-05-11",
    "row_count": 2431188,
    "symbol_count": 2046
  },
  "anomaly_flags": {
    "max_id": 1,
    "max_flag_date": "2024-01-02",
    "row_count": 1
  },
  "peer_groups": {
    "hash": "sha256:abc123…",
    "group_count": 6,
    "entry_count": 33
  }
}
```

`peer_groups.hash` is the hex SHA-256 of the canonicalized JSON
representation of the full peer-groups dict (keys sorted ascending,
inner arrays sorted ascending). Clients use it as a "did the small dict
change?" cache key.

### 5.3 `GET /api/v1/ohlcv`

Returns rows from `price_history`.

**Query parameters**:

| Param | Type | Default | Notes |
|---|---|---|---|
| `since` | `YYYY-MM-DD` | none | If set, return rows with `date >= since`. |
| `until` | `YYYY-MM-DD` | none | If set, return rows with `date <= until`. |
| `symbols` | csv | none (all symbols) | Filter to listed symbols. Max 5,000. |
| `format` | `parquet` \| `ndjson` | `parquet` | See §4.1. |

**Response 200**: parquet (or NDJSON). Columns:

| Column | Parquet type | NDJSON type | Notes |
|---|---|---|---|
| `symbol` | `string` (utf8) | string | |
| `date` | `date32` | string `YYYY-MM-DD` | exchange-local trading date (NYSE/NASDAQ, ET) |
| `open` | `float64` | number | split-adjusted opening price |
| `close` | `float64` | number | **split-adjusted only** (NOT dividend-adjusted) |
| `close_adj` | `float64` | number | split + dividend-adjusted (Yahoo's `adjclose`) |
| `high` | `float64` | number | split-adjusted |
| `low` | `float64` | number | split-adjusted |
| `volume` | `int64` | number | split-adjusted |

**Adjustment basis** (load-bearing — see consumer's CLAUDE.md §4 and §11 for
the long-form story):

- All six numeric columns are **consistently split-adjusted** from
  Yahoo's `chart()` endpoint. No split-day jumps. Verified empirically
  against NVDA 10:1 (2024-06-10) and TSLA 3:1 (2022-08-25).
- `close` and `close_adj` differ ONLY on ex-dividend dates. The
  `(close, close_adj)` divergence is the canonical ex-dividend
  detector.
- Use `close` for split-adjusted price-only analysis. Use `close_adj`
  for total-return labels and any feature that should incorporate
  dividends (e.g. high-yield REITs/utilities).
- `open`, `high`, `low` are split-adjusted but NOT
  dividend-adjusted, consistent with `close`.

Row ordering is unspecified. Clients SHOULD sort by `(symbol, date)`
client-side after load if order matters.

**Example**:
```
GET /api/v1/ohlcv?since=2024-01-01&symbols=AAPL,MSFT
```

### 5.4 `GET /api/v1/peer-groups`

Returns the full `peer_groups` mapping. Tiny; suitable to refetch on
every cursor change.

**Request**: no parameters. `format` is **not** accepted — the response
is always JSON.

**Response 200** (`application/json`):

```json
{
  "tech": ["AAPL", "MSFT", "GOOG", "..."],
  "energy": ["XOM", "CVX", "..."],
  "...": ["..."]
}
```

- Top-level keys are group names.
- Values are arrays of symbols. Order is undefined; clients SHOULD sort
  client-side if needed.

### 5.5 `GET /api/v1/anomaly-flags`

Returns rows from the doctrine's `anomaly_flags` table.

**Query parameters**:

| Param | Type | Default | Notes |
|---|---|---|---|
| `since` | `YYYY-MM-DDTHH:MM:SSZ` | none | If set, return rows with `flag_date >= since`. |
| `status` | csv of `active\|invalidated\|entered\|exited\|dismissed` | none (all) | Filter by status. See state-machine note below. |
| `format` | `parquet` \| `ndjson` | `parquet` | See §4.1. |

**`status` state-machine values** (drift-corrected 2026-05-12 vs. initial
draft):

- `active` — flag is currently in scope but no position has been
  opened. (The initial draft of this spec called this state `open` —
  the actual trading-platform DB uses `active`. Consumers filtering by
  status MUST use `active`, not `open`, or they will silently match
  zero rows.)
- `entered` — position has been opened on the flag.
- `exited` — position has been closed (see `exit_reason`).
- `invalidated` — pre-entry, the setup broke before a position was
  opened (e.g., pivot violated).
- `dismissed` — manually dismissed by an operator (see
  `dismissed_until` for re-eligibility).

**Response 200**: parquet (or NDJSON). Columns (full 22-col superset as
observed in the prod snapshot):

| Column | Parquet type | Notes |
|---|---|---|
| `id` | `int64` | primary key |
| `symbol` | `string` | |
| `flag_date` | `date32` | |
| `fired_at` | `timestamp[us, tz=UTC]` | nullable |
| `pivot_price` | `float64` | |
| `vol_mult` | `float64` | |
| `rsi` | `float64` | |
| `sma20` | `float64` | |
| `sma50` | `float64` | |
| `peer_group` | `string` | |
| `tier` | `string` | e.g. `"3"`, `"3B"` |
| `status` | `string` | one of the values listed above |
| `position_units` | `float64` | nullable |
| `entry_price` | `float64` | nullable |
| `entry_at` | `timestamp[us, tz=UTC]` | nullable |
| `peak_price` | `float64` | nullable |
| `trailing_stop` | `float64` | nullable |
| `invalidated_at` | `timestamp[us, tz=UTC]` | nullable |
| `exited_at` | `timestamp[us, tz=UTC]` | nullable |
| `exit_reason` | `string` | nullable |
| `dismissed_until` | `date32` | nullable |
| `notes` | `string` | nullable |

Columns added in future server versions are non-breaking; clients select
columns by name and ignore extras.

### 5.6 `GET /api/v1/symbols` — **[DRAFT — pending server implementation]**

> **Status**: consumer-team-decided 2026-05-12, not yet implemented on the
> server side. This section is the agreed-on shape; the server team builds
> when ready, then flips this section's status to "shipped". Linked
> discussion: PR #1, comment by `lore-line` 2026-05-12.

Returns per-symbol static-ish metadata. Used by the consumer's
`cap_bucket` and `market_regime` feature modules, plus survivorship
filtering in honest analysis.

**Request**: no parameters. `format` is **not** accepted — the response
is always JSON (dict-shaped, not tabular).

**Response 200** (`application/json`):

```json
{
  "AAPL": {
    "status": "active",
    "last_seen": "2026-05-12",
    "shares_outstanding": 15204000000,
    "sector": "Technology",
    "listing_date": "1980-12-12"
  },
  "DELISTED_X": {
    "status": "delisted",
    "last_seen": "2024-03-15",
    "shares_outstanding": null,
    "sector": "Industrial",
    "listing_date": "2003-04-01"
  }
}
```

Fields per symbol:

| Field | Type | Notes |
|---|---|---|
| `status` | string | `"active"` or `"delisted"`. Other values reserved. |
| `last_seen` | `YYYY-MM-DD` | last trading date the symbol had a price row. For active names this is recent; for delisted names this is the final price date. |
| `shares_outstanding` | int \| null | current shares outstanding (split-adjusted, consistent with `price_history.close`). Nullable for delisted names where the figure is unavailable. |
| `sector` | string \| null | freeform sector label. Nullable for unclassified symbols. |
| `listing_date` | `YYYY-MM-DD` \| null | original NYSE/NASDAQ listing date. Nullable when unknown. |

**Why current shares + point-in-time price** instead of point-in-time
shares: consumer derives `market_cap[t] = close[t] × shares_outstanding`
row-side. Share-count drift (~1-3%/yr) is small relative to 5y price
drift, so this is a defensible cheap approximation of point-in-time
market cap. The alternative (a per-date `shares_outstanding` time
series) was rejected as too expensive to source from Yahoo cleanly.

**Why this matters (target-leakage discussion)**: a current-only
`market_cap_usd` field would inject a tautology into the consumer's
training window — stocks that are now mega-cap were big winners in
2021-2023 by construction. Labeling 2021 rows by today's cap teaches
the model "be currently large", not a structural property. Honest
discovery requires the historical price × current-shares
approximation above.

**Fallback** if `shares_outstanding` is hard to source cleanly from
yahoo-finance2's `quote()` / `quoteSummary()`: server may ship
`market_cap_usd` (current-only) instead, and the consumer's
honest-caveats will note that `cap_bucket` is biased toward current
cap. Point-in-time via `shares_outstanding` is preferred but not a
hard requirement.

**Refresh cadence**: daily is fine. Static metadata doesn't churn.

**Test fixture** (when implemented): consumer team will add a
contract test `test_symbols_returns_per_ticker_metadata` that asserts:
- response is `application/json`
- each value is a dict with the five required keys
- `status ∈ {"active", "delisted"}`
- `last_seen` is parseable as ISO date
- if `shares_outstanding` is present, it's a positive integer

---

## 6. Worked client example

```python
import httpx
import polars as pl

BASE = "https://100.68.86.56:8443/api/v1"

# Step 1: cheap cursor check
cursor = httpx.get(f"{BASE}/snapshot-cursor").raise_for_status().json()
local_max = read_local_cursor()  # from data/snapshots/cursor.json

if cursor["ohlcv"]["max_date"] > local_max["ohlcv_max_date"]:
    # Step 2: incremental fetch
    r = httpx.get(
        f"{BASE}/ohlcv",
        params={"since": local_max["ohlcv_max_date"]},
        timeout=120,
    ).raise_for_status()
    new_rows = pl.read_parquet(r.content)
    merge_into_local_parquet(new_rows)

write_local_cursor(cursor)
```

The actual implementation lives in
`src/quant/data/api_client.py` and `scripts/pull-via-api.py` once the
server side ships.

## 7. Reference implementation notes (for the euieInvest server side)

Non-binding suggestions for whoever builds the server:

- **FastAPI + uvicorn** is the path of least resistance:
  ```python
  app = FastAPI()

  @app.get("/api/v1/ohlcv")
  def ohlcv(since: date | None = None, symbols: str | None = None,
            format: Literal["parquet","ndjson"] = "parquet") -> Response:
      df = query_price_history(since=since, symbols=symbols)
      if format == "parquet":
          buf = io.BytesIO()
          df.write_parquet(buf)
          return Response(buf.getvalue(),
                          media_type="application/vnd.apache.parquet")
      return Response(df.write_ndjson(), media_type="application/x-ndjson")
  ```
- Bind to the Tailscale IP at startup:
  `uvicorn ... --host 100.68.86.56 --port 8443`. Confirm with
  `ss -tlnp | grep :8443`.
- Date columns must be parquet `date32`, not `string` and not
  `timestamp`. polars writes this correctly by default if the source
  column is `pl.Date`.
- v1 has no pagination — full table fits in memory comfortably (2.4M
  rows × 6 cols ≈ 120 MB raw, ~40 MB parquet). Revisit at v2 if the row
  count grows past ~100M.
- Caching: respond with `ETag` and honor `If-None-Match` if cheap; clients
  may opt into client-side cache validation later. Not required for v1.

## 8. Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0.0-draft | 2026-05-12 | Initial draft. Not yet implemented on server. |
| 1.0.0 | 2026-05-12 | Server implementation shipped on claudehost (`100.68.86.56:8443`); 11/11 contract tests pass. Status-enum text in §5.5 corrected from `{open,…}` to `{active,entered,…}` to match the actual trading-platform DB state machine (additive clarification only — no wire-format change). |
| 1.1.0 | 2026-05-12 | §5.3 adds `open` (split-adjusted) and `close_adj` (split + dividend-adjusted) to the /ohlcv response. Additive per §2 — clients selecting columns by name continue to work unchanged. Adjustment-basis paragraph added to §5.3 documenting that all six numeric columns are consistently split-adjusted (the earlier "mixed basis" caveat was incorrect; verified empirically against NVDA 10:1 and TSLA 3:1 splits). |
| 1.2.0-draft | 2026-05-12 | New §5.6 `GET /api/v1/symbols` (DRAFT) added. Per-symbol static-ish metadata: status, last_seen, shares_outstanding, sector, listing_date. Consumer derives point-in-time `market_cap[t] = close[t] × shares_outstanding`. Not yet implemented server-side; section will move from `1.2.0-draft` to `1.2.0` when shipped. |
