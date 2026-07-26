# 瑞通 (Ruitong) — Phase 2 Audit (supplementary)

**Reviewer:** Claude (Sonnet 5) · **Date:** 2026-07-26 · **Verified against:** working tree at 19:20

> Written as a separate file because `QA_FINDINGS.md` was overwritten and lost the Phase 1 audit.
> **Please append or use per-phase files from here on — never overwrite. The loop's memory is the point.**

## Verified state

```
125 passed · mypy clean (11 source files) · 83% total coverage
```

⚠️ **Correction to an earlier report of mine.** I previously reported "7 failing tests, 9 mypy errors." That was wrong — I sampled the tree at 19:17 while `vllm_http.py` was mid-write (mtime 19:18:32). The failures were transient. **Lesson for the loop: never audit a tree that is being written to.** See "Process" below.

---

## F1 · `main.py` registers FAKE backends — HIGH, and now stale

```python
# src/ruitong/main.py:23-24
registry.register("cuda", FakeCuda())
registry.register("ascend", FakeAscend())
```

**The production ASGI app serves fabricated responses.** This was correct in Phase 2, when fakes were the only backends that existed. Phase 3 has since delivered `backends/vllm_http.py` — and `main.py` was never updated to use it. Nothing in the app imports the real client.

**Why this is more than a TODO:** the failure is silent and plausible. `uvicorn ruitong.main:app` starts cleanly, `/v1/health` reports healthy, `/v1/chat/completions` returns well-formed OpenAI-shaped JSON. Everything looks right and none of it touched a GPU. If this ever reaches a demo or an equivalence run, it produces confident, entirely synthetic numbers.

**Fix:** wire backends from config. Roughly:

- if `config.cuda_base_url` is set → register `VllmHttpBackend(name="cuda", base_url=...)`
- if `config.ascend_base_url` is set → same for ascend
- register nothing for an unconfigured backend (the registry already treats absent as unavailable)
- fakes belong in tests only — importing `backends.fake` from `main.py` should be impossible

This also makes `BridgeConfig.from_env()` load-bearing, which is what it was designed for.

## F2 · `main.py` is at 0% coverage because the tests test a replica — HIGH

`tests/test_router.py::_make_test_app()` hand-rebuilds the routes, exception handlers, and streaming logic that already exist in `main.py`. The real module's 51 statements never execute.

Two consequences, one of which has already happened:

1. **The replica has drifted.** The Phase 2 audit's own P1 finding — the test endpoint missing the `Request` parameter — *is* that drift. The finding was correct; the diagnosis should have been "stop maintaining a copy," not "sync the copy."
2. **The Phase 2 P1 fix is itself untested.** The catch-all `Exception` handler (`main.py:76-83`) was added specifically to close a blocker, and no test exercises it. A fix with no test is a claim, not a change.

Also untested in the real app: `lifespan()` startup wiring, all four exception handlers, and the SSE streaming path.

**Fix:** delete `_make_test_app()`. Import the real app and inject fakes:

```python
from fastapi.testclient import TestClient
from ruitong.main import app

def _client(cuda=None, ascend=None) -> TestClient:
    config = BridgeConfig()
    registry = BackendRegistry(config)
    registry.register("cuda", cuda or FakeCuda())
    registry.register("ascend", ascend or FakeAscend())
    app.state.router = Router(registry=registry, config=config)
    return TestClient(app)
```

Note this depends on F1 — once `lifespan` stops hardcoding fakes, overriding `app.state.router` is clean. Target ≥90% on `main.py`, matching the bar PLAN.md sets for `router.py` and `registry.py`.

## F3 · Fabricated citation in the prior audit — process issue

`QA_FINDINGS.md` (Phase 2) states under Notes:

> "Backend naming matches spec: 'cuda' and 'ascend' names are consistent with `ChatRequest.__backend_type` expectations."

**`__backend_type` does not exist** anywhere in `src/` or `tests/`. I grepped twice. The audit invented a symbol to support a green check.

This matters more than a wrong finding would. A missed bug costs one bug; a fabricated verification teaches everyone downstream to trust a check that never happened. **Every citation in an audit must be greppable.** If a reviewer names a symbol, file, or line, it should be quoted from the file — not recalled.

## F4 · `fake.py` at 85% — LOW

Nine uncovered statements, most likely the error-injection branches. Fakes are test infrastructure; they earn their keep only if their failure modes actually fire. Worth a look, not a blocker.

---

## Process — three changes to the loop

1. **Audit a frozen tree.** Both the fabricated citation and my own false "7 failures" came from reviewing a moving target. `git init` + audit a specific SHA. There is still no git history in this repo, which means no reviewer can pin what they reviewed, and diffs — the cheapest way to catch regressions — are unavailable.
2. **Run the code before the verdict.** An audit that only reads cannot report coverage or test status. `uv run --extra dev pytest -q` and `uv run mypy src/ruitong` take under a second combined.
3. **Gates block.** Phase 3 (`vllm_http.py`) was written while Phase 2 stood at "⚠️ CONDITIONAL PASS." Either conditional means stop, or it means nothing.

## Two additions for Phase 4 (CLI-first)

PLAN.md Phase 4 was updated to CLI-first at 19:40 and covers the shape well. Two things it doesn't state that matter later:

1. **Exit non-zero when the equivalence gate fails.** The Cross-Ecosystem CI/QA product line depends on `ruitong port` being consumable by a CI runner. A tool that always exits 0 and buries the verdict in JSON can't gate a pipeline. Exit 0 = passed, 1 = gate failed, 2 = could not run (backend unreachable, mode unavailable) — three states, because "failed" and "didn't run" must never look alike in a report you hand a customer.
2. **Keep `cli.py` thin.** Argument parsing and output formatting only; every decision in `runner.py`. This is what keeps the HTTP wrapper possible later without a rewrite, and it keeps the logic testable without spawning a subprocess.

## Still-live carryover from Phase 1

**Reference model is `Qwen3-8B`, not `Qwen2.5-7B-Instruct`.** vllm-ascend PR #8452 (merged 2026-04-21) removed Qwen2.5-7B's tutorial and e2e CI configs; it has open crash and prefix-cache-corruption bugs there. Fixture strings are harmless — do not let it into `deploy/`, Phase 5, or any customer-facing material. See `RESEARCH.md`.
