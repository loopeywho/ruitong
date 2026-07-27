# 瑞通 Ruitong — Security Audit

**Auditor:** Claude (Sonnet 5) · **Date:** 2026-07-27 · **Revision:** `892f4e7` + working tree
**Method:** every finding below was read from source and, where fixable, verified by re-running
`pytest` and `mypy`. No finding is inferred.

**Scope:** the HTTP surface added under `DECISIONS.md` D4 (`/v1/port`, job store, middleware).
Per D4 this API is **built but not deployed** — Phase 6 remains unapproved. These findings are
therefore **launch gates**, not live incidents.

---

## ✅ Clean

- **SQL injection — not present.** Every query in `jobs/persistence.py` uses `?` placeholders with
  a parameter tuple. No f-strings, no `.format()`, no string concatenation into SQL.
- **Error handlers don't leak internals.** The catch-all returns `type(exc).__name__` only, not
  the exception message or traceback.
- ~~**No secrets in the repo.** `.gitignore` covers `.env`.~~ **← THIS CLAIM WAS FALSE.**
  `.gitignore` contained **zero** `.env` entries. Nothing sensitive was tracked, so the conclusion
  happened to hold — but by luck, not by the control cited. See Round 2 · L1. Fixed since.

---

## S1 · Timing-unsafe API key comparison — HIGH — **FIXED THIS SESSION**

`main.py` compared the presented key with `!=`:

```python
if provided != config.api_key:      # before
```

Python's `str.__eq__` short-circuits on the first differing byte. Response latency therefore leaks
how many leading bytes are correct, allowing byte-by-byte recovery of the key over enough requests.
The rate limiter does not prevent this — it caps requests per minute, and key recovery needs only
patience.

**Fixed:**

```python
if not hmac.compare_digest(provided, config.api_key):
```

Verified: `194 passed`, `mypy: Success`.

*Note: this is a regression from an established pattern in this portfolio — ShangQiao's `/preview`
gate already uses `hmac.compare_digest` (LOO-126 A3). Worth adding to the house rules so it doesn't
recur in a third codebase.*

## S2 · Payload cap is bypassable — MEDIUM — open

`main.py` `payload_size_middleware` only enforces the cap when a `Content-Length` header is present:

```python
content_length = request.headers.get("content-length")
if content_length is not None:
    ...
```

A request using `Transfer-Encoding: chunked` carries **no** `Content-Length`, so the check is
skipped entirely and an unbounded body is accepted. The `except (ValueError, TypeError): pass`
branch also silently accepts a malformed header.

**Fix:** enforce the limit while reading the body, not from the header. Reject once accumulated
bytes exceed `max_payload_bytes`, and treat a malformed `Content-Length` as a 400 rather than
passing it through.

## S3 · Rate limiter grows without bound — MEDIUM — open

```python
request.app.state.rate_limit_buckets = defaultdict(list)
```

Buckets are keyed by client IP and pruned **only for IPs that make a request**. An IP that appears
once leaves a permanent dict entry. Rotating source addresses — trivial over IPv6 — grows the dict
until the process exhausts memory. The rate limiter is itself a denial-of-service vector.

**Fix:** evict stale keys. Simplest correct approach is a periodic sweep dropping buckets whose
newest timestamp is older than the window, or a bounded LRU. Cap total tracked IPs.

## S4 · Rate limiter is blind behind Cloudflare — MEDIUM — open, and concretely relevant

```python
client_ip = request.client.host if request.client else "unknown"
```

**`ruitong.io` is already proxied through Cloudflare** (orange cloud, set up this session). Behind
that proxy `request.client.host` is a *Cloudflare* address, not the caller's. Consequences:

- Every user shares one bucket → 30 requests/minute total across all customers, then everyone gets
  429s. The limiter becomes an outage.
- Abuse is invisible, since one attacker is indistinguishable from all legitimate traffic.

**Fix:** read `CF-Connecting-IP` (or the rightmost untrusted entry of `X-Forwarded-For`) **only
when the peer is a trusted proxy**, with the trusted-proxy list in config. Never trust the header
unconditionally — that makes the limiter trivially spoofable, which is worse than not having one.

## S5 · Auth is disabled by default — MEDIUM — open (launch gate)

`config.api_key` defaults to `""`, and `auth_middleware` is a no-op when it's empty. Deploying
without setting `RUITONG_API_KEY` silently yields an unauthenticated `/v1/port`.

**Fix:** fail closed. On startup, if the app binds anything other than loopback and `api_key` is
empty, refuse to start. A dev default of "open on localhost" is fine; an accidental production
default of "open to the internet" is not.

---

## Launch gate — must all clear before any public exposure

Per Boss's standing rule (security review before **any** public exposure, including temporary
tunnels), and `PLAN.md` Phase 6:

| # | Finding | Severity | Status |
|---|---|---|---|
| S1 | Timing-unsafe key comparison | HIGH | ✅ fixed |
| S2 | Payload cap bypass (chunked encoding) | MEDIUM | open |
| S3 | Rate-limiter unbounded memory | MEDIUM | open |
| S4 | Rate limiter blind behind Cloudflare | MEDIUM | open |
| S5 | Auth off by default | MEDIUM | open |

**Not yet audited** (no code exists yet, or out of scope this round): TLS termination and HSTS;
CORS policy; job-store access control (can any authenticated caller poll any `job_id`?); log
redaction of API keys; dependency CVE scan; and container hardening (the image currently runs as
root — add a non-root `USER` before deployment).

**Recommended next:** job-store access control is the highest-value unaudited item. If `job_id` is
guessable or unscoped, one customer can read another's equivalence report — and the report is the
product.

---

# Round 2 — 2026-07-27 01:00 · adversarial review

Every exploit below was **executed**, not reasoned about. Reviewed at `fa001b2`.

## ⚠️ Correction to Round 1 — two of my own claims were wrong

**L1 · "No secrets in the repo — `.gitignore` covers `.env`." FALSE.**
The file contained **zero** `.env` entries. Nothing sensitive was tracked, so the conclusion held —
but by luck, not by the control I cited. Given H3 below, a `RUITONG_JOB_DB_PATH` pointing inside the
repo would have committed customer equivalence reports. **Fixed:** added `.env`, `.env.*`, `*.db`,
`*.sqlite*`.

**M1 · The S1 fix I applied introduced a new defect.**
`hmac.compare_digest` raises `TypeError` on non-ASCII `str`. Starlette decodes headers as latin-1,
so any byte ≥ 0x80 in `X-API-Key` turned a 401 into an **unauthenticated 500** — and a non-ASCII
*configured* key would have 500'd every request, including correct ones. Fails closed, so not a
bypass, but a free error-rate generator. **Fixed:** compare bytes with `surrogateescape`.

*Lesson, added to `LESSONS.md`: a security fix is a code change like any other and needs its own
adversarial pass. Mine did not get one.*

## C1 · CRITICAL — one request wedges the whole process ✅ FIXED

`prompts` had `min_length=1` and **no upper bound**. The comparison loop is CPU-bound and contains
no `await`, so it never yields to the event loop. Measured against live uvicorn:

```
baseline /v1/health worst latency : 6.3 ms
during one 400k-prompt POST       : 10.5 s      (returns 202 — attacker pays nothing)
```

~2 million prompts fit under the 10 MB cap; the rate limiter permits 30 such requests per minute.
The Dockerfile healthcheck (`--timeout=5s --retries=3`) then kills a container that is merely busy,
turning the stall into a crash-loop.

**Fixed:** `max_length=64` on the list, 8192 chars per prompt.
**Still open:** the runner must move off the event loop (`run_in_executor`/`anyio.to_thread`), and
`JobStore` calls with it. Capping input reduces the blast radius; it does not make the service
concurrent.

## H1 · HIGH — the job store has no concept of an owner ⬜ open

`CREATE TABLE jobs` has no owner/tenant column. `store.get` is `SELECT * FROM jobs WHERE job_id = ?`
with no scoping. `config.api_key` is a **single string**, so the system has no caller identity to
scope by even if it wanted one.

**Good news, precisely:** `job_id = uuid.uuid4().hex` — 122 bits from `os.urandom`. Not enumerable.

**The real exposure:** the job id is a bare bearer capability carried in a URL *path* — it lands in
Cloudflare logs, proxy logs, `Referer` headers and browser history, with no second factor. On the
second customer you either share one key (any customer with an id reads any report) or you need
scoping that does not exist.

**This is the most commercially important finding in the file.** Per `DECISIONS.md` D5 the
equivalence report *is* the product, and the system currently cannot tell one customer from another.

## H2 · HIGH — rate limiting runs before authentication ⬜ open

Starlette's `add_middleware` inserts at index 0, so the **last** registered runs **first**. Auth is
registered first and the limiter last, so the limiter is outermost. Rejected requests still execute
`timestamps.append(now)` before auth ever runs. Executed:

```
unauth codes: [401, 401, 401, 401, 401, 429, 429, 429]
legit caller after the flood: 429
```

Five requests **with no credentials** consumed the entire budget and locked out the paying caller.
Compounded by S4 — behind Cloudflare every customer shares one bucket.

**Fix:** register the limiter *before* `auth_middleware` so it runs after it, and key the bucket on
the authenticated principal rather than the IP.

## H3 · HIGH — unbounded jobs, 29× disk amplification, cap function is dead code ⬜ open

`count_active()` exists and is **called from nowhere** — the concurrency cap was written and never
wired. No `DELETE`, no TTL, no vacuum. Executed: 500 KB of requests produced a **14.5 MB** SQLite
file (29×); a single job's stored report measured 63 MB.

## M2 · MEDIUM — backend errors relay upstream response bodies verbatim ⬜ open

200 bytes of the upstream vLLM body reach the client through the 502 handler. Demonstrated payload
included an internal traceback, a model path under `/models/customer-a/`, and an `hf_token=`.

**Currently unreachable** — `VllmHttpBackend` is not yet wired to any route. **It becomes live the
moment it is**, which is the next step in `deploy/README.md`. Fix before wiring, not after.

## M3–M5 · MEDIUM ⬜ open

- **M3** `asyncio.create_task(_run_job())` — reference discarded, so a suspending task can be
  garbage-collected mid-flight. No timeout, no shutdown drain, no reaper. A `SIGKILL` leaves a row
  stuck in `running` forever.
- **M4** in-memory job store is the default; with `--workers 4`, **18 of 20** valid jobs returned
  404 on at least one poll. File-backed needs WAL and a `busy_timeout`.
- **M5** Dockerfile: runs as **root**; `uv pip install .` reads `pyproject.toml` (`>=` floors), so
  **`uv.lock` is never used** and builds are not reproducible; `uv:latest` is a mutable tag;
  `urlopen(...) or exit(1)` is dead code; no `--start-period`.

## Confirmed CLEAN (tested, not assumed)

`job_id` entropy (uuid4, not enumerable) · SQL injection (re-verified parameterised) · rate-limiter
async race (**no race** — the critical section contains no `await`) · SQLite thread safety
(`RLock` held throughout, safe in-process) · **SSRF — none** (no request field reaches a URL;
`follow_redirects=False`, `verify=True`) · path-based auth bypass (case, trailing slash, `//`,
`..`, `#`, `%23` — all fail closed) · dangerous sinks (no `eval`/`subprocess`/`pickle`) ·
**dependency CVEs — none substantiated** (h11 0.16.0 is post-fix; the risk is that the Dockerfile
ignores the lockfile, not the lockfile itself) · catch-all handler (leaks nothing).

## Launch-gate order

**C1 ✅ → H2 → H3 → M1 ✅** are pure availability, exploitable today from one laptop, cheap to fix.
**H1** needs a design decision (per-customer keys) and matters most commercially. **M2 must land
before `VllmHttpBackend` is wired to a route.**
