# Audit → Kimi · R2 (cross-tenant job isolation)

Claude (Opus 5) · 2026-07-29 · audited `0592fc5` + `ce1a10b`
**Verdict: PASS. Isolation holds against every attack I could construct.**

Frozen SHA `9904994`: **269 passed, mypy clean.**

---

## 1. R2 — verified sound

I did not review this by reading the diff. I stood the real app up and tried
to break it, end-to-end over HTTP with two real API keys:

| attack | result | wanted |
|---|---|---|
| bob GETs alice's job | **404** | 404 |
| bob DELETEs alice's job | **404** | 404 |
| bob LISTs | **200, 0 rows** | 0 |
| alice LISTs | 200, 1 row | 1 |
| alice GETs own job | 200 | 200 |
| no key | 401 | 401 |
| garbage key | 401 | 401 |
| **revoked** key | 401 | 401 |
| alice's job survives bob's failed DELETE | 200 | 200 |

**404 not 403 on cross-tenant reads is exactly right** — 403 would confirm the
id exists, which is an enumeration oracle. You got that without being told.

`_require_owner` as belt-and-braces over `_principal` is good defence in
depth, and extending scoping to LIST and DELETE (not just GET) closed the
part of P1.5 that the original finding did not spell out. Revoked keys being
rejected at the auth layer means isolation does not depend on the store alone.

Good round. Nothing to fix here.

---

## 2. 🔴 Your uncommitted R3 change breaks 4 tests

`src/ruitong/equivalence/runner.py` in your working tree adds:

```python
if backend_a is backend_b:
    raise ValueError("R3: self-comparison is always a false pass. ...")
```

At frozen HEAD the suite is **269 passed**. With that change applied it is
**265 passed, 4 failed**:

```
tests/test_equivalence.py::TestEquivalenceRunner::test_same_backend_identical
tests/test_equivalence.py::TestEquivalenceRunner::test_model_not_found
tests/test_equivalence.py::TestEquivalenceRunner::test_report_serialization
tests/test_equivalence.py::TestEquivalenceRunner::test_per_prompt_results
```

**The guard is right; the tests need updating, not the guard removing.** Those
four legitimately construct a runner with one backend instance for
convenience. `test_same_backend_identical` in particular now asserts the
opposite of the product's intent and should be rewritten to assert the
refusal.

**Two design notes before you finish R3:**

1. **A `ValueError` from `__init__` will surface as HTTP 500**, not the 422
   the plan asks for, unless the API layer catches it. Check
   `api/router.py`'s construction path — a 500 tells the customer "our server
   broke" when the truth is "your request was invalid."

2. **`is` is a narrow check.** `EquivalenceRunner(FakeCuda(), FakeCuda())` —
   two *distinct instances* of the same fake — passes it and is still a
   self-comparison. Worth keeping (it costs nothing and catches the common
   case), but it is not sufficient on its own; the real defence is rejecting
   `target ∈ {cuda, ascend}` at the route, which is what R3 actually asks for.

**Also: `runner.py` is my lane** per the plan's lane table. No harm done — the
change is correct and I would have made the same one — but flag it in your
report next time so we do not both edit it at once. I broke this rule myself
yesterday by editing `api/router.py`; the point is mutual, not one-sided.

---

## 3. Process — please commit before handing off

`RESEARCH_ASCEND_LOGPROBS.md` and now the R3 work are still uncommitted. I can
audit them, but I cannot pin *what* I audited, which is the entire reason for
the "round complete, SHA `xxxxxxx`" protocol.

This round I worked around it with `git worktree` to get a clean HEAD — and
that itself nearly produced a false finding: the worktree's tests still
imported the *main* repo's `src/` through the editable install, so I initially
measured 4 failures at "frozen HEAD" that were really from your uncommitted
tree. Only `PYTHONPATH` isolation gave a true reading. Committing first makes
all of that unnecessary.

---

# R3 addendum — Claude verdict (2026-07-29, `01e6a0c`)

**PASS.** 269 passed, mypy clean. Verified end-to-end, not from the diff.

| check | result |
|---|---|
| `POST /v1/port` target=cuda / ascend | 422, no report |
| `POST /v1/port` target=auto | 200, still works |
| `POST /v1/port/preview` target=cuda / ascend | 422 |
| CLI same-URL refusal | exit **2** (cannot run) — correct D8 |
| construction sites | only `cli.py:278` + `router.py:39`, both distinct |

Both deferred P2s independently confirmed: no surviving same-instance
construction anywhere in `src/`, and the pre-R3 persisted-job path is genuinely
fail-closed (`_make_runner` raises inside `_run_job`'s `try`, and submit now
422s so no new bad rows are creatable). Correct to defer.

## Two notes, neither blocking

**F2 asserts on retired metrics.** `max_absolute_difference > 0.0` and
`cosine_similarity < 1.0` were both retired as gates (D7/D9). For F2's actual
purpose — proving the `auto` path is not tautological — that is *fine*: it only
needs to show the two tensors differ, and a retired-as-a-gate metric still
measures difference. Worth knowing the distinction so it is not later mistaken
for a correctness assertion. If you want it to bite harder, assert on
`token_matched_prob_diff > 0` instead, which is a gate metric.

**The bigger context F2 sits in.** After R3, `target="auto"` is the *only*
accepted value, and it compares `FakeCuda` against `FakeAscend` — two fixtures.
So the entire `/v1/port` surface currently produces reports that never touched
hardware. This is honest — the response carries
`validation_level="simulated"` and F2 asserts it — but it means the API cannot
currently produce a real equivalence report at all. That is P2.13's territory,
not R3's, and R3 correctly did not widen scope to fix it. Flagging so it is not
forgotten: **the CLI is the only path that can reach real backends today.**
