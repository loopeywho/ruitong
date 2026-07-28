# Ruitong Bridge — Phase 4 Audit (Claude Opus 5)

*Audited: 2026-07-28 01:47 UTC | Model: anthropic/claude-opus-5*
*Input tokens: 163181 | Output tokens: 30994 | Duration: 385s*

# Ruitong Bridge — Phase 5 Audit (async REST API: keys, pricing, auth)

**Scope reviewed:** `src/ruitong/auth/keystore.py`, `src/ruitong/auth/router.py`, `src/ruitong/main.py` (middleware stack), `src/ruitong/pricing/*`, `src/ruitong/api/*`, `src/ruitong/config.py`, `tests/test_auth.py`, `tests/test_pricing.py`, `tests/test_api.py`.

**Test/lint state as given:** `235 passed`. That is consistent with what I found: every P1 below is invisible to this suite because the suite never constructs the configuration in which it fires, and in two cases the suite *asserts the buggy behaviour*.

Line numbers are given as `line ~N` where I counted from the file content rather than from a checkout.

---

## Verdict: **FAIL**

Three of the P1s are in the security-sensitive path the brief specifically asks about (admin key verification, rate limiting), one is a data-loss defect in the headline feature (SQLite KeyStore is in-memory by default), and two are false-PASS defects in the equivalence report the business sells. Per `DECISIONS.md` D4 the API is built-but-not-deployed, so these are launch gates rather than live incidents — but P1.1 and P1.2 are exploitable in the *default* and the *most likely upgrade-path* configurations respectively, and must not survive into a commit that anyone can `uvicorn`.

Path to CONDITIONAL PASS: fix P1.1–P1.5 and add the four tests named in §T1.

---

# P1 findings

## P1.1 · Admin API is fully open when no keys are configured
**File:** `src/ruitong/auth/router.py` · **Function:** `_check_admin_key` · **Lines ~31–37**

```python
    else:
        # Legacy: no admin_key configured — accept api_key for admin ops
        if not hmac.compare_digest(
            provided.encode("utf-8", "surrogateescape"),
            config.api_key.encode("utf-8"),
        ):
            raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")
```

`hmac.compare_digest(b"", b"") is True`. With the **default** config (`api_key=""`, `admin_key=""` — `config.py` lines ~66 and ~69), a request with **no `X-API-Key` header at all** passes this check. And `auth_middleware` (`main.py`, `has_auth = bool(config.api_key) or bool(config.admin_key)`) also lets it through, because no auth is configured.

Traced end to end for `POST /v1/admin/keys` with a clean environment:

1. `auth_middleware` → `has_auth` False → sets principal `"anonymous"` → `call_next`
2. `rate_limit_middleware` → passes
3. `payload_size_middleware` → passes
4. `create_key` → `_check_admin_key` → `else` branch → `compare_digest("", "")` → **authorised**
5. Returns `{"key_id":…, "plaintext_key":"rt_…"}`

So an unauthenticated caller can mint API keys, enumerate every key's metadata (`GET /v1/admin/keys`), and revoke every key (`DELETE /v1/admin/keys/{id}` — a one-request denial of service against every paying customer).

This also directly contradicts the documented intent. `config.py` line ~69 says:

```python
    # Admin key for API key management endpoints. Empty = admin API disabled.
    admin_key: str = ""
```

Empty does not disable the admin API; it opens it.

**Fix — fail closed:**

```python
def _check_admin_key(request: Request) -> None:
    config = getattr(request.app.state, "config", None) or BridgeConfig.from_env()
    if not config.admin_key:
        raise HTTPException(503, "Admin API disabled: RUITONG_ADMIN_KEY is not set")
    provided = request.headers.get("X-API-Key", "")
    if not provided:
        raise HTTPException(401, "Missing X-API-Key")
    if not hmac.compare_digest(
        provided.encode("latin-1"), config.admin_key.encode("utf-8")
    ):
        raise HTTPException(403, "Forbidden: invalid admin key")
```

Never branch on an empty secret. Add a startup assertion in `lifespan` too: if the admin router is mounted and `admin_key` is empty, log a loud warning (or refuse to mount the router).

## P1.2 · Every ordinary API-key holder is a full admin in "legacy" mode
**File:** `src/ruitong/auth/router.py` · **Function:** `_check_admin_key` · **Lines ~31–37**

Same `else` branch, second failure mode. In the configuration `RUITONG_API_KEY` set / `RUITONG_ADMIN_KEY` unset — which is exactly what an existing Phase 4 deployment looks like the moment the admin router is mounted — any customer holding the data-plane key can create keys, list all keys, and revoke all keys. There is no separation of duties at all.

Note the test suite believes it covers this. `tests/test_auth.py::TestAdminAPI::test_admin_rejects_non_admin_key` only runs with `RUITONG_ADMIN_KEY="admin-secret"` set (see `_get_client`), i.e. the *safe* branch. The unsafe branch has no test.

**Fix:** delete the legacy branch entirely (covered by the P1.1 patch). If backward compatibility is genuinely required, gate it behind an explicit opt-in (`RUITONG_ALLOW_LEGACY_ADMIN=1`) and document that it grants admin to all data-plane keys.

## P1.3 · Failed authentication is not rate limited at all → unlimited credential brute force with query amplification
**File:** `src/ruitong/main.py` · **Functions:** `auth_middleware`, `rate_limit_middleware`

The H2 fix moved auth to the outermost position (correct: Starlette's `add_middleware` inserts at index 0, so the last-registered `auth_middleware` runs first). But `auth_middleware` returns the 401 `JSONResponse` **without calling `call_next`**, so `rate_limit_middleware` never executes for a rejected request. Consequences:

* An attacker can send **unlimited** `X-API-Key` guesses against `/v1/port` at whatever rate the network allows. Both the KeyStore keys and `config.admin_key` are guessable this way (the admin key is accepted as a data-plane credential — `main.py` `auth_middleware`, `request.state.api_key_principal = "admin"`).
* Each guess costs the server a `SELECT key_id FROM api_keys WHERE key_hash = ? AND is_active = 1` — and `api_keys` has **no index on `key_hash`** (`keystore.py::_init_schema`, lines ~55–65). That is a full table scan per guess, executed while holding `self._lock`, serialising every other request in the process. Cheap request, expensive response: an amplification DoS.
* There is no lockout, no backoff, no logging of failures (see P2.8), so the attempt is also invisible.

The 192-bit key entropy makes *guessing* infeasible, so this is a DoS and detection failure rather than a key-recovery path — but it is the classic "the fix traded one availability bug for another" pattern that `LESSONS.md` warns about, and it must not ship.

**Fix:**
1. Add a failed-auth counter keyed on the trusted client identity, evaluated *inside* `auth_middleware` before touching the KeyStore, with exponential backoff (e.g. 10 failures/minute → 429, then lengthening). Keep it separate from the authenticated-principal bucket so the two cannot starve each other.
2. `CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);` in `_init_schema`.
3. Cap `X-API-Key` length (reject > 256 chars) before hashing.

## P1.4 · The SQLite-backed KeyStore is in-memory by default — every issued key is lost on restart and invisible to sibling workers
**Files:** `src/ruitong/auth/keystore.py` (`__init__` line ~42, `default` lines ~35–39), `src/ruitong/config.py` (`key_db_path` line ~78), `src/ruitong/auth/router.py::_get_key_store`

```python
    def __init__(self, db_path: str = "") -> None:
        path = db_path or ":memory:"
```

`config.key_db_path` defaults to `""`, and `lifespan` does `KeyStore(db_path=config.key_db_path)`. So unless the operator sets `RUITONG_KEY_DB_PATH`, the entire key store is `:memory:`:

* Every key minted through `POST /v1/admin/keys` — the plaintext of which is **unrecoverable by design** — vanishes on restart or redeploy. The customer's credential is silently dead and cannot be reissued to the same identity.
* Under `uvicorn --workers N`, each worker has its own store. A key created on worker 1 returns 401 on worker 2. This is the same defect already recorded as `SECURITY_AUDIT.md` M4 for jobs, reintroduced for credentials, where it is worse.
* `_get_key_store` compounds it: if `app.state.key_store` is absent (lifespan didn't run) it silently falls back to `KeyStore.default()`, which is a process-wide `:memory:` singleton — and then caches it on `app.state`, so the auth middleware happily authenticates against a store that will evaporate.

**Fix:**
* Make persistence mandatory for the key store: if `key_db_path` is empty, either default to a concrete path (`./ruitong-keys.db`) or refuse to start when the admin router is mounted. `:memory:` should only be reachable from tests via an explicit sentinel.
* Delete `KeyStore.default()` or make it raise unless a path was configured; a credential store must never be silently ephemeral.
* For the file-backed case, set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` in `_init_schema` (multi-worker writes to `last_used_at` will otherwise raise `database is locked`).

## P1.5 · Any authenticated key can read any other customer's job report
**Files:** `src/ruitong/jobs/persistence.py::_init_schema` (no owner column), `src/ruitong/api/router.py::get_port_job` (line ~168 area), `submit_port_job`

Phase 5 finally created a per-caller identity (`request.state.api_key_principal`) — and then did not use it. `jobs` still has no owner column, `store.create(job, model=…, target=…)` records no principal, and `get_port_job` is:

```python
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, ...)
    return job
```

`job_id` is `uuid4().hex` so it is not enumerable, but it is a bare bearer capability in a URL *path*: it lands in Cloudflare logs, proxy logs and `Referer` headers, with no second factor. This is `SECURITY_AUDIT.md` H1, which was previously blocked on "there is no caller identity to scope by." That blocker is gone. Per `DECISIONS.md` D5 the report *is* the product; cross-tenant read of the deliverable is the most commercially serious defect in this phase.

**Fix:**
* `ALTER TABLE jobs ADD COLUMN owner TEXT NOT NULL DEFAULT ''` (or a migration to a new schema).
* `submit_port_job` passes `owner=request.state.api_key_principal`.
* `get_port_job` scopes: `store.get(job_id, owner=principal)` and returns 404 (not 403) on owner mismatch, so the endpoint does not confirm existence.
* Add the same scoping to any future list endpoint.

## P1.6 · `POST /v1/port` returns `passed: true` for a backend compared with itself, and for partially-errored runs
**File:** `src/ruitong/api/router.py` · **Functions:** `_make_runner` (lines ~26–43), `run_port`, `submit_port_job`, `_report_to_response`

Two safeguards that `cli.py` enforces are simply absent from the HTTP surface:

1. **Self-comparison is permitted.** `_make_runner` for `target="cuda"` returns `EquivalenceRunner(cuda, cuda)`; for `"ascend"`, `EquivalenceRunner(ascend, ascend)`. `cli.py::_run_port` exits 2 for exactly this (*"comparing an endpoint with itself certifies nothing"*), and `tools/compare_corpora.py` prints `REFUSED`. The API instead emits a report with `passed: true`.
2. **No coverage gate.** `cli.py` computes `passed = report.passed and not synthetic and not incomplete` and refuses to pass when `errored_prompts > 0`. `_report_to_response` uses `report.passed` verbatim and drops `compared_prompts` / `errored_prompts` from the response model entirely (see `api/__init__.py::PortReport`). So an API report can certify a port where 63 of 64 prompts errored — the exact false-PASS `runner.py`'s own comments say is "the worst failure mode for a tool whose output is a trust claim."

The suite **asserts** both behaviours, which is why they survived:

* `tests/test_api.py::TestPortEndpoint::test_port_ascend_single` — `assert data["passed"] is True`
* `tests/test_api.py::TestPortPreviewEndpoint::test_preview_accepts_target` — `assert report["passed"] is True  # self-comparison always passes`

`validation_level: "simulated"` is honest mitigation but it is a separate field a consumer can ignore; `passed` is the field that gates.

**Fix:**
* Reject `target` values that produce a self-comparison with 400, or drop `cuda`/`ascend` as targets and require two distinct backends (matching `--reference NAME=URL --candidate NAME=URL` from D8).
* Move the coverage/synthetic gate out of `cli.py` into a shared helper (`equivalence/gate.py`) used by both surfaces, so the two can never diverge again.
* Add `compared_prompts`, `errored_prompts` and `synthetic` to `PortReport`, and force `passed=False` whenever `errored_prompts > 0` or the run was synthetic.
* Rewrite the two tests above to assert `passed is False` / a 400.

---

# P2 findings

## P2.1 · Non-ASCII admin or API keys can never authenticate
**Files:** `src/ruitong/auth/router.py::_check_admin_key` (lines ~27, ~34), `src/ruitong/main.py::auth_middleware`

```python
provided.encode("utf-8", "surrogateescape"),
config.admin_key.encode("utf-8"),
```

Starlette decodes header bytes as **latin-1**. To recover the wire bytes you must `provided.encode("latin-1")`. Encoding a latin-1-decoded string as UTF-8 turns every byte ≥ 0x80 into two bytes, while `os.environ` gives you a UTF-8-decoded string whose `.encode("utf-8")` *is* the original bytes. The two can therefore never match for a non-ASCII secret — the admin API 403s permanently and the data plane 401s permanently, with no diagnostic. `SECURITY_AUDIT.md` M1 fixed the 500; the silent lockout replaced it.

**Fix:** `provided.encode("latin-1")` on the header side (both call sites), and document that keys must be ASCII. Better: validate at startup that `admin_key`/`api_key` are ASCII and reject otherwise.

## P2.2 · HMAC is used with no server-side pepper; the docstring overclaims
**File:** `src/ruitong/auth/keystore.py` · **Functions:** `create_key` (line ~90), `authenticate` (line ~112)

```python
key_hash = hmac.new(plaintext.encode("utf-8"), b"", "sha256").hexdigest()
```

The plaintext is the HMAC *key* and the message is empty. There is no secret the attacker doesn't already hold, so this is cryptographically equivalent to `sha256(plaintext)` with a fixed IV — a single-round, unsalted, unpeppered digest. The class docstring (line ~28, *"Each key is stored as an HMAC-SHA256 digest"*) implies keyed hashing that isn't happening.

With 192 bits from `secrets.token_hex(24)` this is not currently exploitable, and that is the only reason it isn't P1. But it removes all defence in depth: the moment anyone adds operator-supplied or shorter keys, a stolen DB is trivially crackable, and a global rainbow table over `rt_`-prefixed keys becomes viable.

**Fix:** introduce `RUITONG_KEY_PEPPER` and compute `hmac.new(pepper_bytes, plaintext_bytes, "sha256")`. Store an algorithm/version tag column so keys can be rehashed. Refuse to start if the pepper is unset and any key exists. Fix the docstring either way.

## P2.3 · The `prefix` column stores the constant `"rt_"`, so you cannot tell which row a leaked key is
**File:** `src/ruitong/auth/keystore.py::create_key` (line ~100), consumed by `list_keys`

```python
(key_id, key_hash, name, "rt_", now),
```

The whole purpose of a prefix column is operational identification (`sk-live-8f2a…`). Every row is identical, so `GET /v1/admin/keys` gives an operator no way to correlate a key found in a log, a client config, or a leak with a `key_id` to revoke. Since the plaintext is unrecoverable, the only recourse is revoking everything.

**Fix:** store `plaintext[:11]` (e.g. `rt_8f2a3b1`) and return it from `create_key` and `list_keys`. Consider also storing the last 4 chars.

## P2.4 · `authenticate` writes and commits on every request
**File:** `src/ruitong/auth/keystore.py::authenticate` → `update_last_used` (lines ~120, ~145)

Every authenticated request performs `UPDATE api_keys SET last_used_at = ?` plus `commit()` while holding `self._lock`, on the event-loop thread. Combined with the missing index (P1.3) this makes auth the throughput ceiling of the whole service, and once `key_db_path` is set it means an fsync per request blocking the loop.

**Fix:** debounce — only update when `last_used_at` is older than N minutes (`UPDATE … WHERE key_id = ? AND (last_used_at IS NULL OR last_used_at < ?)`), and/or offload to `anyio.to_thread`. Add the index from P1.3.

## P2.5 · `create_key` will 500 on malformed input
**File:** `src/ruitong/auth/router.py::create_key` (lines ~56–62)

```python
    body = await request.json()
    name: str = body.get("name", "")
```

* Non-JSON body → `json.JSONDecodeError` → caught by `main.py::catch_all_handler` → **500**, should be 400.
* JSON that isn't an object (`[]`, `"x"`, `3`) → `AttributeError: 'list' object has no attribute 'get'` → 500.
* `{"name": {"a": 1}}` → truthy non-str passes the `if not name` guard → `sqlite3.InterfaceError` → 500.
* `name` has no length bound → unbounded DB row from a single request.

**Fix:** use a Pydantic body model, which also fixes the OpenAPI schema:

```python
class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)

@router.post("/keys")
async def create_key(req: CreateKeyRequest, request: Request) -> dict:
```

Also set `Cache-Control: no-store` on the response that carries `plaintext_key`.

## P2.6 · Pricing config is parsed but never validated → 500s from an env var
**Files:** `src/ruitong/config.py::from_env` (lines ~95–105), `src/ruitong/pricing/router.py::list_pricing`, `get_pricing`

`from_env` only checks that `RUITONG_PRICING` is *valid JSON*, then annotates the result `dict[str, dict]` without checking. Reachable 500s:

* `RUITONG_PRICING='[1,2]'` → `list_pricing` → `pricing.items()` → `AttributeError` → 500. `get_pricing` → `pricing[model]` → `TypeError` → 500.
* `RUITONG_PRICING='{"Qwen":0.5}'` → `tier_data.get(...)` → `AttributeError` → 500.
* `RUITONG_PRICING='{"Qwen":{"price_per_input_token_cny":"free"}}'` → `float("free")` → `ValueError` → 500.
* Negative prices are accepted silently — a CNY billing surface should not permit that.

**Fix:** validate at config load with a Pydantic model (`dict[str, PricingTierConfig]` with `ge=0` on both prices) and raise a clear `ValueError` naming the offending model, consistent with how `_int_from_env` already behaves. Then the routers need no defensive coercion.

## P2.7 · Rate limiting collapses to one global bucket in dev mode; the IP fallback is dead code
**File:** `src/ruitong/main.py::rate_limit_middleware` (lines ~200–212)

```python
    principal: str | None = getattr(request.state, "api_key_principal", None)
    if principal is None:
        principal = request.client.host if request.client else "unknown"
```

`auth_middleware` always sets `api_key_principal` — including `"anonymous"` in the no-auth branch — so `principal is None` is never true and the IP fallback is unreachable. With no auth configured (the default), *all* traffic shares one `"anonymous"` bucket: a single client sending 30 requests in a minute 429s every other client. The comment claiming *"in production with auth configured, this path is never reached"* is describing behaviour the code doesn't have.

Related: for exempt paths with auth configured, the middleware also sets `api_key_principal = "anonymous"`, which will silently mislabel the principal for any future handler that reads it.

**Fix:** distinguish "unauthenticated/anonymous" from "authenticated principal" explicitly — set `request.state.api_key_principal = None` in the no-auth branch and let the IP fallback run, or add `request.state.auth_mode`. Then S4 applies again: derive the client IP from `CF-Connecting-IP` **only** when the peer is in a configured trusted-proxy list.

## P2.8 · No audit log for admin operations or auth failures
**Files:** `src/ruitong/auth/router.py` (all three handlers), `src/ruitong/main.py::auth_middleware`

Key creation, key revocation, and failed authentication produce no log line anywhere. For a credential-management subsystem this is a finding in its own right: after an incident there is no way to establish which keys were minted, by whom, or when the brute force started. It also makes P1.1/P1.2/P1.3 undetectable in production.

**Fix:** structured log on create/revoke (`key_id`, `name`, admin principal, timestamp, client IP) and on auth failure (principal-less, IP, path, reason). Never log `plaintext_key` or the presented header value.

## P2.9 · A test that cannot fail
**File:** `tests/test_api.py::TestPayloadCap::test_large_payload_rejected` (lines ~198–216)

```python
        if resp.status_code == 413:
            assert resp.json()["error"] == "Payload too large"
        else:
            assert resp.status_code == 200
```

Both branches pass, and the comments in the test body (*"exceeds 100-byte limit? No, 50 < 100"*) show the author knew the payload doesn't trigger the cap. This is exactly `LESSONS.md` L4: it would pass identically with `payload_size_middleware` deleted.

**Fix:** send a body that provably exceeds `RUITONG_MAX_PAYLOAD_BYTES` and assert `413` unconditionally. Add the `Transfer-Encoding: chunked` bypass case from `SECURITY_AUDIT.md` S2 as an xfail so the open finding is tracked in code.

## P2.10 · The production wiring path (`lifespan`) is never exercised, and `KeyStore.default()` leaks state across tests
**Files:** `tests/test_api.py` (line ~11, `client = TestClient(app)`), `tests/test_auth.py`, `tests/test_pricing.py`

Every test constructs `TestClient(app)` without the context-manager form, so `lifespan` never runs. Consequently `app.state.config`, `app.state.job_store`, `app.state.key_store`, `app.state.pricing_config` and `app.state.rate_limit_*` are never set, and every request takes the `getattr(..., None)` fallback path. So:

* `KeyStore(db_path=config.key_db_path)` and `JobStore(db_path=…)` — the actual production constructors — have **zero** coverage. P1.4 lives entirely in untested code.
* `_get_key_store` falls back to `KeyStore.default()`, a class-level singleton on `ruitong.auth.keystore.KeyStore`. `importlib.reload(ruitong.main)` does *not* reload `ruitong.auth.keystore`, so the same singleton — and every key in it — is shared by all tests in the session. `test_list_keys` asserting `len(keys) >= 1` rather than `== 1` is the tell.
* Combined with module-level `client = TestClient(app)` in `test_api.py` plus `importlib.reload` elsewhere, the suite is order-dependent.

**Fix:** use `with TestClient(app) as client:` (or `httpx.ASGITransport` with an explicit lifespan) so the real wiring is what's tested; add a fixture that constructs `KeyStore(db_path=str(tmp_path/"keys.db"))` and injects it via `app.dependency_overrides` / `app.state`; reset `KeyStore._default_instance` in an autouse fixture.

## P2.11 · Missing tests on the exact behaviours the brief asks about
See §T1 below. Notably: **no test asserts that `/v1/pricing` requires auth.** `tests/test_pricing.py` never sets `RUITONG_API_KEY`, so it only ever exercises the unauthenticated path. The endpoint *is* correctly non-exempt (it's absent from `AUTH_EXEMPT_PATHS` in `main.py` line ~24) and therefore does inherit middleware auth — but nothing pins that, so removing it or adding `/v1/pricing` to the exempt set would be a silent, green regression.

## P2.12 · Middleware ordering comments contradict the code and each other
**File:** `src/ruitong/main.py` (lines ~110–120, and the docstrings of `payload_size_middleware` and `rate_limit_middleware`)

Actual execution order is `auth → rate_limit → payload_size → handler`. `payload_size_middleware`'s docstring says *"Number 2 in the execution order (runs after auth, before rate-limit)"* (it is number 3, after rate-limit) and `rate_limit_middleware`'s says *"Number 3 … runs after auth + payload check"* (it is number 2, before the payload check). The consequence is real if minor: an oversized body consumes rate-limit budget before being rejected.

**Fix:** correct the comments, and consider registering `payload_size` last-but-one so a 413 doesn't cost the caller quota. Add a test that asserts the order (e.g. an oversized unauthenticated request returns 401, not 413).

## P2.13 · Carryovers still open and now reachable from Phase 5 routes
* `src/ruitong/main.py::lifespan` (lines ~40–41) still registers `FakeCuda()`/`FakeAscend()`; `GET /v1/models` instantiates fakes per request. The production ASGI app serves fabricated data (`QA_PHASE2.md` F1, unfixed).
* `src/ruitong/jobs/persistence.py::count_active` is defined and called from nowhere — the job concurrency cap remains dead code; no `DELETE`, TTL or vacuum (`SECURITY_AUDIT.md` H3). `submit_port_job` is now reachable by any of N keys.
* `src/ruitong/api/router.py::submit_port_job` — `asyncio.create_task(_run_job())` discards the reference (GC-able mid-flight), with no timeout, shutdown drain or reaper (M3).
* `run_port` executes the CPU-bound comparison loop on the event loop (C1 remainder).

---

# P3 findings

| # | Location | Issue / fix |
|---|---|---|
| 1 | `auth/keystore.py::KeyStoreError` (line ~19) | Defined, never raised. Either use it (wrap `sqlite3.Error`) or delete it. |
| 2 | `auth/keystore.py::revoke_key` (line ~135) | Returns `True` for an already-revoked key (SQLite counts matched rows). Add `AND is_active = 1` so the endpoint can distinguish "revoked now" from "was already revoked". |
| 3 | `auth/keystore.py` schema | No `expires_at`, no `scopes`/`permissions`. Multi-key auth without expiry or least-privilege will need a migration later; add the columns now even if unused. |
| 4 | `auth/keystore.py::close` + `default()` | The singleton is never closed, and reusing it after `close()` raises `sqlite3.ProgrammingError`. Tests call `close()` on their own instances only. |
| 5 | `api/router.py::_make_runner` (line ~41) | The `else: raise HTTPException(400)` branch is unreachable — `PortRequest.target` has `pattern=r"^(cuda\|ascend\|auto)$"`. Dead code. |
| 6 | `api/router.py::_report_to_response` | Emits only the metrics D9 **retired** (`cosine_similarity`, `max_absolute_difference`) plus index-based top-1/top-5, and reports `cosine_min`/`max_abs_diff_max` as `thresholds` — none of which gate. The gating `token_matched_prob_diff`, `probability_mass_delta` and their thresholds are omitted, so a consumer cannot see why `passed` is what it is. Sync the API models to `Thresholds` and `report.metrics`. (Raise to P2 if the API is intended to ship as the deliverable.) |
| 7 | `main.py` | `AUTH_EXEMPT_PATHS` exempts `/v1/models` and `/openapi.json` from **both** auth and the rate limiter — unauthenticated, unlimited model-inventory disclosure and a full public schema of the admin API. Consider gating `/openapi.json` and `/docs` in production. |
| 8 | `main.py::rate_limit_middleware` | 429 lacks a `Retry-After` header (the value is only in the JSON body); 401 lacks `WWW-Authenticate`. |
| 9 | `main.py::rate_limit_middleware` | `timestamps.pop(0)` is O(n); use `collections.deque` + `popleft`. Eviction of the "stalest" bucket resets that principal's counter — an authenticated attacker with many keys could exploit it; prefer a TTL sweep only. |
| 10 | `main.py` | Unused import `ChatRequest`. `api/router.py`: unused `JSONResponse`, unused `request: Request` on `run_port`; `api/__init__.py::PortError` is defined and never used. |
| 11 | `config.py` | `pricing_config: dict[str, dict]` is a mutable field on a frozen dataclass, and `lifespan` publishes the same object as `app.state.pricing_config`. Freeze it (`MappingProxyType`) or model it. |
| 12 | `auth/router.py::_get_key_store` | Mutates `request.app.state` from inside a request handler — benign in a single-threaded loop but a surprising side effect; do the wiring in `lifespan` only. |
| 13 | `scripts/phase_audit.py::main` | (a) `FINDINGS_FILE.write_text(full)` **overwrites** `QA_FINDINGS.md`, which is precisely the process failure `QA_PHASE2.md` opens with (*"never overwrite — the loop's memory is the point"*); write `QA_PHASE5.md` or append. (b) The header hardcodes `"Phase 4 Audit"` while auditing Phase 5. (c) The docstring advertises `--phase N`, which is never parsed — there is no `argparse`. (d) `trim_to_budget` truncates mid-file with no marker, so a trimmed audit silently reviews half a file. |

---

# T1 · Minimum tests to add before this can pass

1. **Admin fail-closed:** with `RUITONG_ADMIN_KEY` and `RUITONG_API_KEY` both unset, `POST/GET/DELETE /v1/admin/keys` with no header **and** with an arbitrary header must not return 2xx. (Fails today — catches P1.1.)
2. **No privilege escalation:** with `RUITONG_API_KEY` set and `RUITONG_ADMIN_KEY` unset, the api_key must not be accepted on `/v1/admin/keys`. (Fails today — catches P1.2.)
3. **Failed auth is charged:** with `RUITONG_ADMIN_KEY` set and `rate_limit_per_minute=2`, five requests with a wrong key must not all return 401 — one must be 429. (Fails today — catches P1.3.)
4. **Keys survive a restart:** create a key against `KeyStore(db_path=tmp)`, close, reopen, `authenticate(plaintext)` returns the same `key_id`; plus an HTTP-level test that a revoked key gets 401 from the middleware (currently only tested at the KeyStore layer). (Catches P1.4.)
5. **Per-principal buckets:** two distinct KeyStore keys each get their own quota; exhausting one must not 429 the other. (The H2/S4 fix is currently unverified.)
6. **Job isolation:** key A submits a job, key B polling that `job_id` gets 404. (Catches P1.5.)
7. **Pricing requires auth:** with `RUITONG_API_KEY` set, `GET /v1/pricing` and `/v1/pricing/{model}` return 401 without a key and 200 with one. (Catches P2.11.)
8. **Malformed inputs return 4xx not 5xx:** non-JSON body, JSON array body, and non-string `name` on `POST /v1/admin/keys`; `RUITONG_PRICING='[1,2]'` and `'{"m":{"price_per_input_token_cny":"free"}}'` on the pricing routes. (Catches P2.5, P2.6.)
9. **Reflexivity/self-comparison refusal at the API layer:** `POST /v1/port {"target":"cuda"}` must not return `passed: true`. Replaces the two tests that currently assert the opposite. (Catches P1.6.)

Also: run the suite with `-p no:randomly`-style ordering variation, or simply reverse `testpaths` order once — the shared `KeyStore.default()` singleton and module-level `TestClient(app)` make the current green result order-dependent.

---

# Summary table

| ID | Sev | File · function | Issue |
|---|---|---|---|
| P1.1 | P1 | `auth/router.py::_check_admin_key` ~31–37 | `compare_digest("","")` → admin API open with no credentials |
| P1.2 | P1 | `auth/router.py::_check_admin_key` ~31–37 | Any api_key holder becomes full admin |
| P1.3 | P1 | `main.py::auth_middleware` / `rate_limit_middleware` | Failed auth never rate limited; unindexed table scan per guess |
| P1.4 | P1 | `auth/keystore.py::__init__`/`default`, `config.py::key_db_path` | Key store is `:memory:` by default — keys lost on restart, per-worker |
| P1.5 | P1 | `jobs/persistence.py` schema, `api/router.py::get_port_job` | No job ownership → cross-tenant report read |
| P1.6 | P1 | `api/router.py::_make_runner`, `_report_to_response` | API emits `passed: true` for self-comparison and errored runs |
| P2.1 | P2 | `auth/router.py`, `main.py` | latin-1 vs utf-8: non-ASCII keys can never match |
| P2.2 | P2 | `auth/keystore.py::create_key`/`authenticate` | HMAC with no pepper ≡ unsalted SHA-256; docstring overclaims |
| P2.3 | P2 | `auth/keystore.py::create_key` | `prefix` is the constant `"rt_"` → can't identify a key to revoke |
| P2.4 | P2 | `auth/keystore.py::authenticate` | Write + commit + full scan on every request |
| P2.5 | P2 | `auth/router.py::create_key` | Malformed body → 500; `name` unvalidated/unbounded |
| P2.6 | P2 | `config.py::from_env`, `pricing/router.py` | Pricing config unvalidated → 500 from env var; negative prices allowed |
| P2.7 | P2 | `main.py::rate_limit_middleware` | Single global `"anonymous"` bucket; IP fallback unreachable |
| P2.8 | P2 | `auth/router.py`, `main.py::auth_middleware` | No audit log for key create/revoke or auth failure |
| P2.9 | P2 | `tests/test_api.py::test_large_payload_rejected` | Test cannot fail |
| P2.10 | P2 | all test modules | `lifespan` never runs; `KeyStore.default()` singleton leaks across tests |
| P2.11 | P2 | `tests/test_pricing.py` et al. | Missing tests (see T1), incl. pricing auth |
| P2.12 | P2 | `main.py` middleware docstrings | Ordering comments contradict code |
| P2.13 | P2 | `main.py::lifespan`, `jobs/persistence.py`, `api/router.py` | Fake backends in prod app; dead job cap; orphaned tasks; blocking loop |
| P3.1–13 | P3 | see table above | Dead code, missing headers, unused imports, `phase_audit.py` overwriting `QA_FINDINGS.md` |

**Verdict: FAIL.** Re-audit at a pinned SHA (per `PLAN.md` rule 9) once P1.1–P1.6 and the §T1 tests land.