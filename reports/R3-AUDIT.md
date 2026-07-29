# R3 Audit — Opus 5
*Audited: 2026-07-29 00:43 | SHA: e516a80 (HEAD: ce1a10b5d9db37e664189b14275c42df51cbbcee) | Model: anthropic/claude-opus-5*
*Input: 6552 | Output: 5770*

# Audit — Ruitong Bridge R3 (removal of single-target self-comparison false pass)

**HEAD:** `ce1a10b` · **R3 commit:** `e516a80` · **Port tests:** 25 passed / 0 failed

## Verdict: **CONDITIONAL PASS**

The API-surface defect is genuinely fixed: the tautological `EquivalenceRunner(x, x)` construction is *deleted* (not flagged, not warned about), the schema refuses the inputs that reached it, and the tests that used to *assert* the false pass are gone rather than skipped. Two verification gaps prevent an unconditional PASS — one about entrypoints outside `src/ruitong/api/`, one about whether the surviving `auto` path is itself a disguised self-comparison.

---

## Criterion-by-criterion

| # | Criterion | Result | Basis |
|---|---|---|---|
| 1 | `/v1/port` rejects `target="cuda"` with 422 | **PASS** | `pattern=r"^auto$"` rejects at Pydantic validation → FastAPI 422; `test_port_cuda_single` asserts it and passes |
| 2 | `/v1/port` rejects `target="ascend"` with 422 | **PASS** | Same mechanism; `test_port_ascend_single` passes |
| 3 | `/v1/port/preview` rejects single-target with 422 | **PASS** | Preview binds the same `PortRequest`; rejection happens *at submit* (422), not as a background-job error after a 202 — confirmed by the rewritten `test_preview_accepts_target` |
| 4 | `target="auto"` still produces a valid report | **PASS (shape only)** | `test_port_auto`, `test_port_response_shape`, `test_port_includes_calibrated_metrics` pass; see F2 for the semantic caveat |
| 5 | Any remaining path to single-target `passed: True` | **NOT PROVEN** — none in `api/`, unverified elsewhere | See F1 |
| 6 | Old false-pass tests replaced, not co-resident | **PASS** | The `assert data["passed"] is True  # self-comparison always passes` line and the `val == 1.0` / `val == 0.0` self-comparison assertions are deleted from `tests/test_api.py`; no `@pytest.mark.skip` or `xfail` shell left behind. 25 passed / 244 deselected with no skips in the port selection |

Defence-in-depth is correctly layered, which matters more than it looks: Pydantic v2's `pattern` is *search*, not *fullmatch*, and under the Python-`re` fallback engine `$` also matches before a trailing newline. So `target="auto\n"` can slip past the regex — and lands in `_make_runner`'s `else`, which raises 422. The redundant guard is load-bearing for that edge case. Good.

---

## Findings

### F1 — P2: Fix is scoped to the HTTP surface; other entrypoints unaudited
The diff touches only `api/__init__.py` and `api/router.py`. `EquivalenceRunner` accepts any two backends, and nothing in the runner itself refuses `runner.a is runner.b`. If a CLI (`ruitong port --target cuda`), notebook example, or `jobs/` helper constructs `EquivalenceRunner(cuda, cuda)`, the false `passed: True` survives — with the same operator-facing meaning, just not over REST. Criterion 5 cannot be answered PASS on the evidence supplied.

**Required before sign-off:**
```
rg -n 'EquivalenceRunner\(' --type py -g '!tests/**'
rg -n 'target' src/ruitong/cli* src/ruitong/jobs/ docs/ README*
```
Any site where both arguments are the same object/class must be removed or made to raise. **Strongest remedy:** put the guard in `EquivalenceRunner.__init__` (`raise ValueError` if the two backends are the same instance or same class), so the invariant holds for every caller and the API regex becomes a convenience, not the sole defence.

### F2 — P2: No test proves the surviving `auto` path isn't also a tautology
R3's premise is that self-comparison is worthless because it cannot fail. The `auto` path compares `FakeCuda` vs `FakeAscend` — two *simulated* backends. If both derive logits from the same generator/seed and differ only in label, `auto` is a self-comparison wearing a costume, and R3 removed the honest tautology while keeping a concealed one. Nothing in the test suite excludes this: `test_port_auto` asserts only shape and key presence; no test asserts `cosine_similarity < 1.0`, `max_absolute_difference > 0.0`, or that a deliberately divergent backend yields `passed: False`.

**Required:** inspect `backends/fake.py` for distinct divergence, then add a test that pins non-triviality, e.g. `assert data["metrics"]["max_absolute_difference"] > 0.0`, plus a negative test where an injected-divergence backend drives `passed is False`. Until a run *can* fail, `passed: True` from `/v1/port` carries no information — which is the exact defect R3 set out to remove.

### F3 — P3: The explanatory R3 message is unreachable in practice
Because Pydantic rejects `"cuda"` first, callers receive `String should match pattern '^auto$'` — not the carefully written *"single-target self-comparison was removed (R3)"* detail in `_make_runner`. The useful diagnostic only fires on the newline edge case. An integrator who upgrades and gets a regex error has no path to the rationale.

**Suggested:** replace the regex with `Literal["auto"]` (better OpenAPI: renders as an enum) plus a `field_validator` that raises the R3 message, or accept the legacy values in the schema and let `_make_runner`'s 422 do the refusing.

### F4 — P3: The redundant guard has no test
`_make_runner`'s `else` branch is unreachable via HTTP, so no test exercises it. If someone later reintroduces `elif target == "cuda": return EquivalenceRunner(cuda, cuda), ...` alongside a widened regex, the suite stays green.

**Suggested:** direct unit test — `pytest.raises(HTTPException)` on `_make_runner("Qwen3-8B", "cuda")`, asserting `.status_code == 422`.

### F5 — P3: Vestigial field / undocumented contract break
`target` now has exactly one legal value equal to its default, so it is inert. Clients previously sending `target="cuda"` break with no deprecation window, changelog entry, or version bump visible in this diff. Either document the removal in the API changelog/OpenAPI description (the field description does explain it — extend that to release notes) or drop the field.

---

## Bottom line
Criteria 1, 2, 3, 6 pass cleanly and the removal is done the right way — deleted code path, deleted false-pass assertions, layered rejection. Promote to unconditional PASS once **F1** (no self-comparison construction anywhere outside the API) and **F2** (the surviving `auto` comparison is demonstrably capable of failing) are closed. F3–F5 are polish and need not block.
