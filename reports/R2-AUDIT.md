# R2 Audit — Opus 5
*Audited: 2026-07-28 21:50 | SHA: 0592fc5 | Model: anthropic/claude-opus-5*
*Input: 7646 | Output: 6489*

# Audit — Ruitong Bridge R2 (cross-tenant isolation)

## Verdict: **CONDITIONAL** (blocked on 2× P1)

The endpoint shapes are correct and the tests prove the *happy-path* isolation claim between two authenticated keys. But the actual enforcement lives entirely in `JobStore.list_by_owner` / `.delete(owner=)` / `.get(owner=)`, which is **not in this diff and not exercised for fail-closed behaviour**. Meanwhile `_principal()` is documented to return `""` in a supported operating mode, and nothing in the new code refuses to proceed on an empty owner. Those two facts together mean the diff's docstring claims ("cross-tenant list is impossible by design", "cross-tenant delete is impossible") are **not established by the evidence presented**.

---

## Criterion-by-criterion

| # | Criterion | Result | Basis |
|---|---|---|---|
| 1 | LIST scopes by owner | **Unproven** | Scopes by *argument*. `store.list_by_owner("")` behaviour unknown; no test with an empty/missing principal. |
| 2 | DELETE scopes by owner | **Unproven** | Same. `store.delete(id, owner="")` behaviour unknown. |
| 3 | GET 404 not 403 | **Pass (for the shown path)** | `get_port_job` raises 404 uniformly for "absent" and "not yours"; no 403 branch, no distinguishing detail string. Delete follows the same pattern. Still inherits the empty-owner caveat. |
| 4 | Edge cases | **Fail** | Empty-list, delete-your-own-nonexistent, double-delete, fresh-key-sees-nothing, no-header, dev-mode: all missing. |
| 5 | Any path where missing/empty owner allows access | **Yes — see F1** | `_principal` returns `""` by design; no guard; store semantics unverified. |

---

## Findings

### F1 — P1 — Empty principal is passed straight through to the store; no fail-closed guard
`_principal()` returns `""` when auth is disabled ("dev mode — jobs are not scoped to a tenant"), and returns `""` via `getattr(..., "")` if the middleware never ran for a route (ordering change, mounted sub-app, exception path). All three new/changed handlers then call the store with `owner=""` and take no defensive action.

If `JobStore` uses the extremely common pattern `if owner: rows = [r for r in rows if r.owner == owner]`, then `owner=""` means **no filter**:
- `GET /v1/port/preview` → dumps every job in the process, for every tenant, in one unauthenticated request.
- `DELETE /v1/port/preview/{id}` → 204 on any tenant's job.

Note the severity asymmetry the new LIST endpoint introduces: previously an attacker needed a 32-hex job ID to attempt cross-tenant read. LIST turns that into a single-request full-inventory disclosure *and* hands the attacker the IDs needed to drive DELETE. Even if the store happens to be fail-closed today, nothing in R2 pins that behaviour.

**Required fix** (belt and braces — both layers):
```python
def _require_owner(request: Request) -> str:
    owner = _principal(request)
    if not owner:                       # dev mode, or middleware bypassed
        raise HTTPException(status_code=403, detail="Tenant scoping unavailable")
    return owner
```
and in `JobStore`: `if not owner: raise ValueError("owner is required")` at the top of `get`, `delete`, `list_by_owner`. A tenant-scoped API must not have a mode where scoping silently disappears; if dev mode needs unscoped access, give it an explicit synthetic owner (`"__dev__"`) rather than a falsy sentinel.

### F2 — P1 — Security-critical code path is outside the diff and untested at the unit level
The entire isolation guarantee is `JobStore.list_by_owner` / `.delete(owner=)` / `.get(owner=)`. There are no store-level tests in this diff. 263 green tests say nothing about `list_by_owner("")`, `delete(id, owner=None)`, `delete(id, owner="' OR 1=1")`, or whether `list_by_owner` filters in SQL vs. in Python after a full table read. **Blocker: publish `persistence.py` plus direct store tests before this ships.** Minimum store tests:

```python
def test_list_by_owner_rejects_empty(store): 
    with pytest.raises(ValueError): store.list_by_owner("")
def test_delete_rejects_empty(store):
    with pytest.raises(ValueError): store.delete(jid, owner="")
def test_delete_rejects_none(store): ...
def test_get_rejects_empty(store): ...
def test_owner_matched_exactly(store):  # no prefix/LIKE/case-fold matching
    store.put(job, owner="tenantA"); assert store.list_by_owner("tenanta") == []
```

### F3 — P2 — Divergent store resolution: `JobStore.default()` fallback
All three read/delete handlers do `store = JobStore.default()` when `app.state.job_store` is missing, while `submit_port_job` skips its concurrency check on that branch (`if store is not None`). Two problems: (a) if `default()` constructs a *fresh* store, LIST silently returns `[]` and DELETE silently 404s — an availability bug that masquerades as correct isolation and will be misread as "isolation works"; (b) if `default()` is a process-wide singleton, you now have two possible stores and no guarantee reads hit the store writes went to. Resolve the store once in a dependency and fail loudly (500) if unconfigured — never silently substitute a different backing store on a security-relevant read.

### F4 — P2 — Owner identity appears to be the plaintext API key
The `_principal` docstring says it returns "the real key when configured". If so, the plaintext key is the partition key persisted in the job store and joined into every job row.
- Confirm `JobInfo` has **no** `owner`/`principal` field, or `GET /v1/port/preview` now returns the caller's plaintext key in a response body (and into any response log/cache).
- Keys in `WHERE owner = ?` land in query logs and slow-query logs.
- Key rotation orphans every job.
Use a stable non-secret `key_id` (the hash/row-id you already store in the key DB) as the owner. Add an explicit test: `assert "owner" not in a_list.json()[0]` and that no response body contains `key_a`.

### F5 — P2 — DELETE does not cancel the in-flight job; deleted jobs can resurrect
`submit_port_job` spawns background work that writes results back to the store. `delete_port_job` removes the row but does not signal that task. A delete during `pending`/`running` is likely followed by the worker re-inserting the row (upsert) or crashing on a missing row. The new test only deletes a job that has almost certainly finished under the fake backends, so the race is untested. Add: submit → delete immediately → poll for ~1s asserting 404 the entire time. Also confirm the deleted-while-running job releases its `count_active()` slot exactly once (double-decrement lets a tenant exceed `max_concurrent_jobs`).

### F6 — P2 — Missing fail-closed and edge-case tests (criterion 4)
None of these exist; each maps directly to a way F1/F3 could ship undetected:
1. Fresh key with zero jobs → `200` and `[]` (not other tenants' jobs, not 404).
2. `DELETE` a well-formed but nonexistent ID that "you own" → 404.
3. Double delete → 204 then 404.
4. `GET`/`DELETE`/`LIST` with **no** `X-API-Key` while auth is configured → 401, and body contains no job data.
5. `GET`/`DELETE`/`LIST` with a *revoked* key → 401.
6. Delete then list → job absent from own list.
7. Cross-tenant GET returns exactly `404` and a detail string **identical** to the nonexistent-ID case (the current 404 detail is `f"Job {job_id} not found"` in both branches — good; lock it with an assertion so nobody "helpfully" adds `"owned by another tenant"` later).
8. Auth-disabled mode: assert the documented behaviour explicitly, whatever you choose it to be. Right now the one operating mode with zero isolation has zero tests.

### F7 — P3 — Unbounded LIST
`response_model=list[JobInfo]` with no limit/cursor, and `JobInfo` embeds the full `result.report` (per-prompt metrics). A tenant with a few thousand jobs makes this a self-inflicted DoS and a large log/response payload. Add pagination and consider a slim summary model for list.

### F8 — P3 — `204` handler with `-> None`
Returning `None` from a `status_code=204` route emits a `null` body plus `content-length: 4` on older FastAPI versions, which is protocol-invalid and breaks strict clients. Pin the behaviour: `assert b_delete.content == b""` in the test, or return `Response(status_code=204)`.

### F9 — P3 — No audit logging on DELETE
A destructive, tenant-scoped endpoint with no log line for either success or the cross-tenant 404. Cross-tenant 404s on DELETE are the highest-signal intrusion indicator this service can produce — log `{key_id, job_id, outcome}` and alert on bursts.

### F10 — P3 — Test-suite hygiene
`importlib.reload(ruitong.main)` inside test bodies rebuilds the app while the module-level `from ruitong.main import app` and the `client` fixture still reference the pre-reload object. This makes isolation results order-dependent and will eventually produce a green run against a stale app. Move the reload into a fixture that yields the freshly-built app, or use a dedicated app factory. Also: both new tests only ever assert on data they created — none asserts the *absence* of a third tenant's jobs seeded by a different test.

---

## Gate for PASS

1. F1: `_require_owner` in the router **and** `if not owner: raise` in all three store methods.
2. F2: `persistence.py` posted; store-level fail-closed tests green.
3. F4: confirm no plaintext key in `JobInfo`/response bodies.
4. F6: items 1–8 added and green.
5. F3 and F5 either fixed or filed with an explicit owner and follow-up ticket.

Until then the correct reading of this diff is: *isolation holds between two authenticated keys against an unreviewed store; behaviour with a missing owner is unknown and the code is written to keep going rather than stop.* The docstrings' "impossible by design" should be softened to describe what is actually enforced.
