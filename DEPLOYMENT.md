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
  -H "X-API-Key: *** \
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
| `RUITONG_CUDA_BASE_URL` | No | — | CUDA vLLM endpoint (e.g. `http://10.0.0.1:8000`). Unset = fake backend |
| `RUITONG_ASCEND_BASE_URL` | No | — | Ascend vllm-ascend endpoint (e.g. `http://10.0.0.2:8000`). Unset = fake backend |

### validation_level

Every `/v1/port` report includes a `validation_level` field that tells you what
backends actually ran:

| Value | Meaning |
|-------|---------|
| `simulated` | At least one backend is a fake — the numbers are not from real hardware |
| `live` | Both backends are real vLLM instances — the report reflects actual hardware |

When neither `RUITONG_CUDA_BASE_URL` nor `RUITONG_ASCEND_BASE_URL` is set, the
server falls back to `FakeCuda`/`FakeAscend` and every report returns
`validation_level="simulated"`. To get live reports, set both env vars before
starting the server.

## Startup order

1. Set env vars
2. Delete any pre-d668f58 `ruitong-keys.db`
3. Start the server — it creates tables on first connect