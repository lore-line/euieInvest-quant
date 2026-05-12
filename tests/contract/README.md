# Contract tests

Live HTTP tests that validate a **real** running implementation of the
euieInvest data API against the contract in
[`docs/api-contract.md`](../../docs/api-contract.md).

These tests are **skipped** unless `EUIEINVEST_API_BASE_URL` is set, so
they don't run in the default `docker compose run --rm test`. They are
intended for:

- **The server team** — to validate their implementation matches the
  contract before deploying.
- **The consumer team** — to validate the deployed server before
  cutting over from the legacy rsync path.

## Run against a local dev server

```sh
EUIEINVEST_API_BASE_URL=http://localhost:8443 uv run pytest tests/contract/ -v
```

## Run against the deployed claudehost server (from heaven-pc)

```sh
EUIEINVEST_API_BASE_URL=http://100.68.86.56:8443 \
    docker compose run --rm dev uv run pytest tests/contract/ -v
```

## What they cover

- All five endpoints return 200 with the documented schemas
- Parquet columns and dtypes match the contract **exactly**
- `since=` and `symbols=` query params filter correctly
- The 5000-symbol cap returns `400 problem+json`
- Cursor consistency: `cursor.ohlcv.max_date` equals `max(ohlcv.date)`
- `peer_groups.hash` is reproducible across calls

## What they do NOT cover

- Performance / latency (eyeball this manually; targets are in
  [`server-implementation-guide.md`](../../docs/server-implementation-guide.md) §8)
- Behavior under server-side errors (the server impl will exercise its
  own error paths in its own unit tests)
- TLS / auth (deferred per the contract)
