# Deployment Guide

## Keystore migration

Commit `d668f58` (2026-07-28) changed key hashing from `hmac.new(...)` to
`hashlib.sha256(...)`. This invalidates every hash in any pre-d668f58
`ruitong-keys.db`.

**If you have an existing keystore from before d668f58:** delete it and
regenerate all keys. The `hash_scheme` column defaults to `sha256` and the
`authenticate` method refuses rows with an unknown scheme — you cannot
accidentally authenticate against an HMAC hash. The error surfaces as a
generic "invalid key", not a confusing traceback.

**Clean start (recommended for all deployments):**
```bash
rm -f ruitong-keys.db   # if it exists
export RUITONG_ADMIN_KEY="<your-admin-key>"
ruitong serve
```

Then create keys via:
```bash
curl -X POST <host>/v1/admin/keys \
  -H "X-API-Key: <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-key"}'
```

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `RUITONG_ADMIN_KEY` | No | — | If set, enables multi-key auth via KeyStore |
| `RUITONG_API_KEY` | No | — | Legacy single-key auth (ignored when admin_key is set) |
| `RUITONG_KEY_DB_PATH` | No | `ruitong-keys.db` | Path to SQLite keystore |
| `RUITONG_JOB_DB_PATH` | No | `:memory:` | Path to job persistence store |

## Startup order

1. Set env vars
2. Delete any pre-d668f58 `ruitong-keys.db`
3. Start the server — it creates tables on first connect