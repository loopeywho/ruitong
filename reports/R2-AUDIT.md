# R2 Audit — Opus 5 (Fix Round)
*Audited: 2026-07-28 22:07 | R1 SHA: 0592fc5 → R2 SHA: ce1a10b5d9db37e664189b14275c42df51cbbcee | Model: anthropic/claude-opus-5*
*Input: 9495 | Output: 7575*

# Ruitong Bridge R2 — Fix Round Audit (`ce1a10b5`)

## VERDICT: **CONDITIONAL**

Both P1 cross-tenant defects are genuinely fixed, at two layers, with tests that exercise the store layer. What blocks a clean RESOLVED is that **the fix is only tested where it is easiest to test** (store unit layer) and the P2 items from R2 are almost entirely untouched — including one (`JobStore.default()` fallback) that is a silent store-divergence path and one (delete/worker race) that the diff does not address at all.

---

## Audit criteria

**1. Does LIST scope by owner? — YES.**
`list_port_jobs` → `_require_owner(request)` → `store.list_by_owner(owner)` → `SELECT * FROM jobs WHERE owner = ?`. Missing owner never reaches the store: 403 at the route, and `ValueError("owner is required")` at the store if it somehow did. Fail-closed is stronger than the criterion asked for (403 rather than empty list) and leaks nothing.

**2. Does DELETE scope by owner? — YES.**
`delete_port_job` → `_require_owner` → `DELETE FROM jobs WHERE job_id = ? AND owner = ?`; `deleted == False` → 404. Cross-tenant delete is a no-op reported as 404. Missing owner → 403 before the store. Correct.

**3. Does GET return 404 (not 403) cross-tenant? — YES.**
`store.get(job_id, owner)` is owner-scoped, returns `None`, route raises `404 Job {job_id} not found`. Existence is not confirmed. The 403 case is orthogonal (it says "enable authentication", not "that job is someone else's"), so it does not leak existence either.

**4. Edge cases covered? — PARTIALLY. This is the main gap.**
`test_persistence.py` covers empty/None owner on get/list/delete, and cross-tenant get/list/delete. Nothing in the evidence shows the following, all of which are the cases a regression would actually reappear in:
- HTTP-level: tenant A submits → tenant B `GET` = 404, `DELETE` = 404, `LIST` = `[]` (route wiring is what P1-1 was about; it is currently asserted nowhere).
- Fresh key → `GET /v1/port/preview` returns `200 []`.
- Delete a non-existent job you own → 404; double-delete → second call 404.
- Dev mode (no `RUITONG_API_KEY`) → 403 on all four routes.
- Assertion that the persisted `owner` column is **not** the raw key (the F4 claim is unasserted).

269 passing tests do not evidence any of these; the only test change visible in the diff is adding a key header to the preview tests.

**5. Any path where missing/empty owner grants access? — None found in the reviewed code.** Two residual soft spots, neither an access grant:
- `create(..., owner: str = "")` is **not** validated. Empty-owner rows can still be written (any direct caller, or a future route that forgets `_require_owner`). They are unreachable via get/list/delete, so they are orphans, not a leak — but the fail-closed rule is applied asymmetrically.
- Not verifiable from the material provided: whether any *other* JobStore consumer exists (worker status/result updates, admin routes in `auth/router.py`) that reads or writes by `job_id` alone. Router file is truncated below `_report_to_response`. **Must be confirmed by grep before sign-off** — `grep -rn "job_store\|JobStore\|_principal(" src/` should show `_principal` called only from `_require_owner`, and no unscoped read helper reachable from a route.

---

## Previous findings

| # | Finding | Status | Evidence |
|---|---|---|---|
| P1-1 | LIST/DELETE use `_require_owner` not `_principal` | **RESOLVED** | All four routes (submit/get/list/delete) now call `_require_owner`. `_principal` has no remaining direct call site in the shown code. Caveat: assert via grep on the truncated file. |
| P1-2 | Store-level fail-closed on empty owner | **RESOLVED** | `get`/`list_by_owner`/`delete` each raise `ValueError("owner is required")`; unit-tested for empty and `None`. Incomplete only in that `create` is unguarded. |
| P1-3 | Dev-mode auth-off collapse | **RESOLVED (security), with functional cost** | Dev mode leaves `api_key_principal` unset → `""` → 403. All async job endpoints are now unusable without auth; the diff had to rewrite the preview tests to inject a key. That is the correct trade, but it is a behaviour change that must be in release notes, not just a docstring. |
| P2-1 | `JobStore.default()` fallback | **NOT RESOLVED** | Unchanged in all four routes: `store = getattr(app.state, "job_store", None) or JobStore.default()`. Owner scoping applies to both stores so this is not a leak, but it silently binds requests to a process-global in-memory singleton when lifespan did not run — jobs written to one store and read from the other produce phantom 404s and make isolation testing unreliable. Should raise 500 instead of inventing a store. |
| P2-2 | Auth-required tests for new routes | **NOT RESOLVED** | No test asserts 401/403 for the job routes, nor 404 for cross-tenant access at HTTP level. The store tests do not cover route wiring — which is precisely what P1-1 was. |
| P2-3 | Delete/worker race | **NOT RESOLVED** | Nothing in the diff touches the background task or the update path. Unverified risk: if the worker's completion write is an upsert rather than a scoped `UPDATE`, deleting a running job resurrects the row with `owner=''` — unreachable (so not a leak) but unbounded orphan growth. Needs the `update_status`/`set_result` implementation shown, and a test: submit → delete → let worker finish → assert row count is 0. |
| P3-1 | Edge-case tests | **PARTIALLY RESOLVED** | Store-level isolation and fail-closed are covered. The list in criterion 4 above is not. |

---

## New findings this round

**N1 (P2) — F4 fix is forward-only; existing raw keys are still on disk.** Principals are now `sha256(key)[:16]`, but nothing migrates or scrubs `jobs.owner` rows that already contain raw API keys from before this commit. On any file-backed DB, the secret the fix was meant to stop persisting is still persisted. Needs a one-time `UPDATE jobs SET owner = ''` (or hash-forward) migration; note this also silently orphans every pre-existing job, which should be an explicit decision rather than a side effect.

**N2 (P3) — 64-bit truncated principal.** `[:16]` hex = 64 bits. Not exploitable against a specific tenant (2^64 offline preimage work), and collisions between two live keys are negligible at realistic key counts. Flagging only because the truncation buys nothing — store the full digest.

**N3 (P3) — key rotation orphans jobs.** Principal is derived from the key material, so rotating a key makes that tenant's job history invisible. Acceptable for preview jobs; document it.

**N4 (P3) — stale docstrings now actively misleading.** `_principal` still claims it returns "the real key when configured, 'anonymous' when not" — neither is true after this commit. `JobStore.get`'s docstring still says "An empty owner now matches only rows literally owned by `''`", which the new `raise ValueError` contradicts. These are the comments a future reader will trust when deciding whether `_principal` is safe to call directly.

**N5 (P3) — test hygiene in the rewritten preview tests.** `monkeypatch.setenv` + `importlib.reload(ruitong.main)` inside a test body mutates module state for the rest of the session and diverges from the module-level `from ruitong.main import app` used by the `client` fixture. Prefer an app-factory fixture; this pattern makes any future isolation test order-dependent.

**N6 (trivial)** — `router.py` has no trailing newline.

---

## Required before sign-off

1. **HTTP-level isolation tests** with two distinct keys: cross-tenant GET/DELETE = 404, LIST excludes, fresh key = `200 []`, dev mode = 403 on all four routes, own-nonexistent delete = 404. (Closes P2-2 and P3-1; this is the blocker.)
2. **Grep confirmation** that no other route or helper touches `JobStore` unscoped, and that `_principal` is called only by `_require_owner`.
3. **Show the worker update path** and add the submit→delete→worker-completes test (P2-3).
4. `JobStore.default()` fallback → raise instead (P2-1); add `if not owner: raise` to `create` for symmetry.
5. Migration/scrub for raw-key `owner` rows (N1) + fix the three stale docstrings (N4).

Items 1–3 are the ones I would hold the merge on; 4–5 can ship as a follow-up in the same release.
