# R2 Audit: Cross-Tenant Job Isolation

**Auditor:** Opus 5 (audit gate, Finn Loop)  
**SHA:** 3a16cb6  
**Original commit:** eef25d8 — `fix(auth): R2 — cross-tenant job isolation (P1.5)`  
**Tests:** 263 passed (was 258)  
**Mypy:** clean, 26 files  
**Verdict:** **CONDITIONAL** (blocks 2 P3 test-gap items; no P1/P2 issues)

---

## What was changed

The R2 fix added owner scoping across three files:

| File | Change |
|---|---|
| `src/ruitong/api/router.py` | Extracted `_principal()` helper; pass `owner=principal` on create, get, list, delete |
| `src/ruitong/jobs/persistence.py` | Added `owner` column to schema; `get()`, `list_by_owner()`, `delete()` all scope by owner |
| `tests/test_api.py` | 3 new tests in `TestCrossTenantIsolation` |

---

## 1. Owner scoping logic — ✅ correct

### GET `/preview/{job_id}`
```python
# router.py line 219
owner = _principal(request)
job = store.get(job_id, owner=owner)
```
**JobStore.get():** `WHERE job_id = ? AND owner = ?` — fail-closed. If the job exists but is owned by a different principal, the query returns `None` → 404. ✅

### LIST `/preview`
```python
# router.py line 236-237
owner = _principal(request)
return store.list_by_owner(owner)
```
**JobStore.list_by_owner():** `WHERE owner = ? ORDER BY created_at DESC` — only returns jobs matching the caller's principal. ✅

### DELETE `/preview/{job_id}`
```python
# router.py line 250-253
owner = _principal(request)
deleted = store.delete(job_id, owner=owner)
if not deleted:
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
```
**JobStore.delete():** `DELETE FROM jobs WHERE job_id = ? AND owner = ?`. Returns `False` (rowcount=0) if the job doesn't exist OR if it's owned by someone else. Both cases map to 404. ✅

**Key design choice:** DELETE returns 404 (not 403) for cross-tenant access. This is correct — 403 would confirm the job exists (information leak), while 404 is blind to existence.

---

## 2. Fail-closed default-refuse — ✅ satisfied

Every persistence method enforces owner scoping:

| Method | Query | Missing owner behavior |
|---|---|---|
| `get()` | `WHERE job_id = ? AND owner = ?` | Returns `None` → 404 (deny) |
| `list_by_owner()` | `WHERE owner = ?` | Returns `[]` (empty list) |
| `delete()` | `WHERE job_id = ? AND owner = ?` | Returns `False` → 404 (deny) |

The `get()` docstring (line 87-104 of persistence.py) explicitly documents the **fail-closed rationale**: the previous version defaulted `owner=""` and treated empty as "read any job" — a job owned by "alice" was visible to `get(job_id, owner="")`. The new version is strictly scoped. Dead code removed. ✅

**Dev mode edge case:** When auth is disabled (`RUITONG_API_KEY` and `RUITONG_ADMIN_KEY` unset), `_principal()` returns `""`. All dev-mode jobs get owner `""` and are visible to each other. This is **by design** — the docstring at line 52 says "An empty string means auth is disabled (dev mode) — jobs are not scoped to a tenant." Not a bug; documented behavior.

---

## 3. Test coverage — ⚠️ 3 gaps (all P3)

### P3.1 — No test for empty list
**Gap:** No test asserts that `GET /v1/port/preview` returns `[]` (empty JSON array) when a key has no jobs. This is a valid edge case — a freshly-created API key should see an empty list, not an error.

**Risk:** Low. The implementation is straightforward (`WHERE owner = ?` returns `[]`), but an empty-list regression (e.g., accidentally returning `null` or an error) wouldn't be caught.

### P3.2 — No test for "delete already-deleted job"
**Gap:** `test_cross_tenant_delete_returns_404` tests: (a) cross-tenant delete → 404, (b) owner delete → 204, (c) re-read → 404. But it does **not** test that deleting an already-deleted job returns 404. This is a valid idempotency check.

**Risk:** Low. The `store.delete()` returns `False` for non-matching rows, and the handler maps that to 404. The behavior is correct, but no test verifies the idempotent--delete path.

### P3.3 — No test for "list isolation with no jobs"
**Gap:** The list isolation test creates jobs for both A and B, then checks filtering. But it doesn't test the pre-job state — after creating keys A and B but before submitting any jobs, neither key should see the other's (nonexistent) jobs. A fresh key's list should be `[]`.

**Risk:** Low. Same as P3.1 — the implementation is simple, but no test anchors the empty-list baseline.

---

## 4. Positive findings

1. **404 vs 403 distinction** is consistently applied across GET, LIST, and DELETE. No endpoint leaks existence via 403.
2. **`_principal()` helper** is clean, centralized, and handles the dev-mode edge case gracefully.
3. **Background task tracking** (P2.13) is correctly integrated alongside the R2 changes — orphaned tasks are tracked in `app.state.background_tasks` and cleaned on shutdown.
4. **Concurrency limit check** (429) correctly runs before job creation, and uses the same `store` path that's owner-scoped.
5. **SQLite schema change** (`owner TEXT NOT NULL DEFAULT ''`) is safe — the `NOT NULL` with default ensures no NULL-column injection can occur.
6. **The `get()` docstring** is excellent security documentation — it explains the previous vulnerability, why it was dangerous ("failure mode was allow everything"), and how the fix addresses it.

---

## 5. Issues summary

| # | Severity | Description | Fix |
|---|---|---|---|
| P3.1 | P3 (trivial) | No test for empty list on fresh key | Add `test_list_empty_returns_empty_array` |
| P3.2 | P3 (trivial) | No test for deleting already-deleted job | Add `test_delete_own_job_twice_second_returns_404` to existing test |
| P3.3 | P3 (trivial) | No test for list isolation baseline (pre-job state) | Add assertion after key creation, before job submission |

---

## 6. Verdict reasoning

**CONDITIONAL** — The security implementation itself is **correct and sound**. All three endpoints (GET, LIST, DELETE) properly scope by owner, fail-closed, and return 404 for cross-tenant access. The persistence layer is properly scoped, the `_principal()` helper is clean, and there are no P1 or P2 issues.

The CONDITIONAL flag is solely for the 3 P3 test gaps — none of which represent real security risks. They are defensive test additions that would strengthen the regression surface. These can be added as follow-up commits without blocking this fix.

**Recommendation:** Approve with the three P3 test additions logged as follow-ups. The security fix is production-ready as-is.
