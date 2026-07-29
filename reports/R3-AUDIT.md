# R3 Audit — Opus 5
*Audited: 2026-07-29 01:29 | SHA: e516a80 (HEAD: 040c9575b976bfd38476f8779ee0324bc6e89c69) | Model: anthropic/claude-opus-5*
*Input: 6571 | Output: 6094*

# Audit — R3: removal of single-target self-comparison false pass

**Commit under review:** `e516a80` (verified against HEAD `040c957`, which carries the R2 fix on top)

---

## Criteria results

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | `POST /v1/port` rejects `target="cuda"` with 422 | **MET** | `PortRequest.target` pattern is now `^auto$` → pydantic rejects at body validation. `test_port_cuda_single` asserts 422 and passes. |
| 2 | `POST /v1/port` rejects `target="ascend"` with 422 | **MET** | Same mechanism; `test_port_ascend_single` asserts 422 and passes. |
| 3 | `POST /v1/port/preview` rejects single-target with 422 | **MET** | Preview shares `PortRequest`; rejection happens before job submission (no 202 + no job row). `test_preview_accepts_target` asserts 422 and passes. |
| 4 | `target="auto"` still produces a valid report | **MET** | `test_port_auto` gets 200, full `PortReport` shape, 3 per-prompt rows, and — importantly — asserts `max_absolute_difference > 0` and `cosine_similarity < 1.0`, i.e. it *proves* the auto path is not a disguised self-comparison. This is the correct guard; without it, someone could "fix" R3 by making `auto` return `FakeCuda` twice and all tests would still pass. |
| 5 | Any remaining path to single-target `passed: True` | **MET for the two demonstrated entry points**; residual verification gap (F1) | `_make_runner` is now the sole constructor for the request path, and its `else` branch fails closed with 422 rather than constructing `EquivalenceRunner(x, x)`. No `EquivalenceRunner(cuda, cuda)` / `(ascend, ascend)` construction survives in the reviewed file. |
| 6 | Old passing-report tests replaced, not duplicated | **MET** | The diff *rewrites* the three test bodies in place (no new test names added alongside old ones), and the current `tests/test_api.py` contains no assertion that a single-target run returns `passed: True` or all-1.0/0.0 metrics. Port suite: 25 passed, 0 skipped, 0 xfail. |

The core defect is genuinely gone: the tautological `EquivalenceRunner(cuda, cuda)` construction — which returned `cosine_similarity=1.0`, `max_absolute_difference=0.0`, `passed=True` for a backend compared against itself — no longer exists, and every removal is enforced by a test that fails if it comes back.

---

## Findings

### F1 — P2: Non-request entry points and full-suite evidence not shown
The test output is scoped (`244 deselected`). Two things are therefore unverified by the evidence presented:

- **Other callers / entry points.** `_make_runner` is a router-private helper. If a CLI (`ruitong port --target cuda`), a benchmark script, or a notebook constructs `EquivalenceRunner` directly with the same backend twice, the false-pass survives there — the schema pattern only guards the HTTP boundary. Required check: `rg -n "EquivalenceRunner\(" --glob '!tests/**'` and confirm no call site passes the same object/class twice; `rg -n '"cuda"|"ascend"' src/ruitong/cli*` for a surviving single-target flag.
- **Cross-file test residue.** Criterion 6 is verified for `tests/test_api.py` only. Any other test file (jobs, e2e, docs snippets) that still asserts a single-target run returns a report would now fail or, worse, still pass against a stale code path. Required check: full-suite green, plus `rg -n 'target.*(cuda|ascend)' tests/`.

### F2 — P2: Persisted pre-R3 job payloads replay through the worker, not through validation
`JobStore` persists submitted jobs. A job row written before R3 with `target="cuda"` is resumed/executed **without** re-passing `PortRequest` validation — it reaches `_make_runner` directly. The behaviour is *fail-closed* (the `else` branch raises), so **no false pass is produced** — this is the correct outcome and the reason the `else` branch must be kept rather than deleted as "dead code". However:

- `_make_runner` raises `HTTPException(422)` in a background-task context, where there is no request/response cycle to translate it. Depending on the worker's exception handling, that job either lands in `status="error"` with a confusing HTTP-shaped detail, or — if the worker only catches specific exception types — is left stuck in `running`/`pending` forever.
- Recommendation: have `_make_runner` raise a domain error (`ValueError` / `UnsupportedTargetError`) and translate it to 422 at the router edge, so both the sync and async paths degrade predictably. Confirm the worker's `except` is broad enough to mark the job `error`.

### F3 — P3: The informative error message is unreachable for the cases it was written for
The `detail` string in `_make_runner` explains the R3 rationale, but for `target="cuda"` the client never sees it — pydantic short-circuits with the generic `String should match pattern '^auto$'`. The friendliest message is only emitted for inputs that slip past the regex. Consider a `field_validator` on `target` (or `Literal["auto"]` plus a custom exception handler) so the rationale actually reaches the caller.

### F4 — P3: Prefer `Literal["auto"]` over `pattern=r"^auto$"`
Three reasons:
1. **OpenAPI quality.** `Literal` emits `enum: ["auto"]`; the regex emits an opaque `pattern`, which generated clients render as a free-form string and only fail at runtime.
2. **Anchor semantics.** Pydantic v2 `pattern` uses *search* semantics, so the `^`/`$` anchors are load-bearing here — they are correctly present, but the construct is engine-dependent (`rust-regex` default vs. `python-re` fallback, which treats `$` as also matching before a trailing `\n`). Under the Python engine, `target="auto\n"` would pass validation and fall through to the `else` branch — fail-closed, but it needlessly widens the surface that F2's exception-type problem applies to. `Literal` has no such quirk.
3. The field now has exactly one legal value, which is precisely what `Literal` expresses.

### F5 — P3: Unannounced breaking API change on a `/v1` path
`target="cuda"|"ascend"` went from 200 to 422 with no version bump, deprecation window, or changelog entry in the diff. This is the right call on safety grounds — a false `passed: True` is worse than a hard refusal, as the docstring correctly argues — but it should be recorded as a breaking change so downstream callers aren't surprised. Confirm no README/API-docs/example still advertises `target: cuda`.

### F6 — P3 (informational, out of R3 scope): `auto` still compares two fakes
R3 removes the *tautology*, not the *simulation*. `FakeCuda` vs `FakeAscend` diverge by construction, so `max_absolute_difference > 0` and the report is no longer self-referential — but `passed: True` on this path is still evidence about two fixtures, not about CUDA vs Ascend hardware. `validation_level: "simulated"` is present and asserted, which is the right mitigation. Flagging only so that "R3 closed the false-pass issue" is not later read as "the `auto` path validates real equivalence."

### F7 — P3: Check for now-unused test imports
The preview test's polling loop (`time.sleep`, `pytest.fail`) was deleted. If no other test in the file uses `time`, ruff `F401` will flag the module-level `import time`. The truncated view suggests other preview tests still poll, so this is likely fine — a lint run confirms.

---

## Verdict: **PASS**

All six audit criteria are met. The tautological self-comparison is removed at the only two entry points in evidence, the replacement behaviour fails closed with 422 rather than degrading to a weaker pass, the old false-pass assertions are rewritten in place rather than duplicated, and — the strongest part of this change — `test_port_auto` now asserts *divergence* on the surviving path, so the fix cannot be silently reverted by pointing `auto` at a single backend.

No P1 findings. Merge is not blocked. Before closing R3, land the two P2 follow-ups: (F1) grep-and-confirm no non-HTTP caller constructs `EquivalenceRunner` with the same backend twice, plus a full-suite green run; and (F2) replace the background-reachable `HTTPException` with a domain exception translated at the router edge, and confirm a pre-R3 persisted `target="cuda"` job lands in `status="error"` rather than hanging.
