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
- **No secrets in the repo.** `.gitignore` covers `.env`; nothing sensitive is tracked at `4cd7835`.

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
