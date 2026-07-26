# Fable Audit — Phase 2: Router + Fake Backends

## Context

Phase 2 of the 瑞通 (Ruitong) Bridge — CANN/CUDA bridge middleware. Phase 1 (contracts) is already audited and fixed (62/62 tests, 100% coverage, mypy clean). This phase adds the routing layer, fake backends, and FastAPI endpoints.

## Files to Review

<!-- Fill in the actual file paths, line counts, and key sections after Qwen finishes -->

### `src/ruitong/backends/fake.py`
- [ ] FakeCuda — deterministic responses, injectable failure/latency
- [ ] FakeAscend — deterministic responses, injectable failure/latency
- [ ] Creates `FakeCuda(name="cuda")` and `FakeAscend(name="ascend")`
- [ ] Each fake should report which models it serves
- [ ] Can simulate: healthy, unhealthy, model-absent, latency

### `src/ruitong/registry.py`
- [ ] BackendRegistry class
- [ ] `register(name, backend)` — stores backend
- [ ] `get(name)` — returns backend or raises BackendUnavailable
- [ ] `resolve(model, preferred_backend="auto")` — routing logic
- [ ] `auto` resolution: prefer healthy AND serving the model, ties break by priority
- [ ] No silent cross-backend fallback (if chosen backend fails, error naming it)

### `src/ruitong/router.py`
- [ ] Router class (no FastAPI dependency — pure routing logic)
- [ ] `chat(req) -> ChatResponse` — resolves backend, calls chat()
- [ ] `stream(req) -> AsyncIterator[ChatChunk]` — resolves backend, calls stream()
- [ ] `health() -> list[HealthStatus]` — checks all registered backends
- [ ] `list_models() -> list[ModelInfo]` — aggregates across all backends
- [ ] Error mapping: BackendUnavailable → 503, ModelNotFound → 404, BackendError → 502

### `src/ruitong/main.py`
- [ ] FastAPI app with lifespan
- [ ] POST /v1/chat/completions — accepts ChatRequest, returns ChatResponse (or streams ChatChunks)
- [ ] GET /v1/models — returns list of ModelInfo
- [ ] GET /v1/health — returns list of HealthStatus
- [ ] Minimal — just wiring, no business logic

### `tests/test_router.py`
- [ ] End-to-end tests against fakes via TestClient
- [ ] `auto` routing: both-healthy, one-unhealthy, both-unhealthy, model-absent
- [ ] Error codes: 404 model, 503 unavailable, 502 backend error
- [ ] Coverage ≥ 90% on router.py and registry.py
- [ ] Existing tests still pass (62 from Phase 1)

## Acceptance Bar (from PLAN.md)

- All endpoints work end-to-end against fakes via `httpx.AsyncClient` or `TestClient`
- `auto` covered for: both-healthy / one-unhealthy / both-unhealthy / model-absent
- Errors map to correct codes: 404 model, 503 unavailable, 502 backend
- Coverage ≥ 90% on `router.py` and `registry.py`
- Suite runs offline in < 5s
- All Phase 1 tests still pass

## Key Design Decisions to Verify

1. **No silent failover** — if the chosen backend fails, the error names the backend. Silent failover makes equivalence claims unprovable.
2. **`stream()` is `def`, not `async def`** — see the Backend protocol and the F2 fix from Phase 1. An async generator returns the iterator directly; `async def` makes it a coroutine.
3. **Fakes are deterministic** — tests must be reproducible across runs. No randomness.
4. **Router is pure logic** — no FastAPI dependency in router.py. The FastAPI layer is a thin wrapper in main.py.
5. **Backend names match the schema** — "cuda" and "ascend", matching Literal["cuda", "ascend", "auto"] in ChatRequest.

## Review Prompt

@Opus 5 (Fable), please review Phase 2 of the Ruitong Bridge. Focus on:

1. **Correctness** — Does the routing logic correctly resolve `auto`? Does it handle all edge cases (both healthy, one unhealthy, both unhealthy, model absent)?
2. **No silent failover** — Are there any code paths where a backend failure silently switches to another backend?
3. **Protocol compliance** — Do the fake backends correctly implement the Backend protocol? Is `stream()` `def` not `async def`?
4. **Error mapping** — Do errors map to the correct HTTP status codes?
5. **Test coverage** — Are the acceptance tests thorough? Do they cover all states of the `auto` routing matrix?
6. **Code quality** — Is there any unnecessary abstraction, duplication, or dead code?

Please write findings to QA_FINDINGS.md in the same structured format as Phase 1:
- Status: ❌ FAIL / ⚠️ CONDITIONAL PASS / ✅ PASS
- P1 — Must Fix (deploy blockers)
- P2 — Should Fix
- P3 — Polish
- Notes