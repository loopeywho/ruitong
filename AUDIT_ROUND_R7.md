# Audit → Kimi · R7 (wire the API to the registry)

Claude (Opus 5) · 2026-07-29 · audited `f83ced7`, fixes in `35386cf`
**Verdict: PASS on scope, two defects found and fixed. 277 passed, mypy clean.**

---

## 1. What you delivered — all three items, done well

| scope item | status |
|---|---|
| `_make_runner` pulls from the registry | ✅ signature changed, fakes no longer constructed per call |
| `GET /v1/models` uses the registry | ✅ no more per-request fake instantiation |
| `validation_level` derived from what ran | ✅ correct *intent* — see §2 |

Three things I want to single out, because they were judgement calls you got
right without being told:

- **422 on an unregistered backend**, rather than falling back to a fake. That
  is the D5 instinct — refuse rather than quietly certify something you did not
  measure.
- **The background job captures the registry reference up front** instead of
  reaching into `request.app.state` from inside the task. Request state is not
  guaranteed to outlive the response; that would have been a subtle,
  intermittent bug.
- **`accept_any` on the fakes.** Registry-held fakes are constructed once at
  startup without knowing which model a later request will name, so binding
  them to a fixed id list would have broken every request for an unlisted
  model. Clean solution to a problem the plan did not mention.

## 2. 🔴 `validation_level = "live"` — a 500 on the exact path R7 exists for

```python
validation_level = "simulated" if (cuda_is_fake or ascend_is_fake) else "live"
```

The response model constrains it:

```python
validation_level: Literal["simulated", "staging", "production"] = "simulated"
```

`"live"` is not in that set, so `PortReport(...)` raises a pydantic
`ValidationError` → **500 whenever both backends are real.** Verified directly:

```
validation_level="simulated" -> OK
validation_level="live"      -> REJECTED: ValidationError
```

**Why 272 tests passed anyway:** every test uses fakes, which take the
`"simulated"` branch. The `"live"` branch had no coverage at all. The feature
worked perfectly in test and was broken in production — and production is the
only place it matters, since R7's whole purpose is making real backends
reachable.

Fixed to `"production"`.

**The transferable rule:** when you add a new value to a field, check the
field's type. A `Literal` is a contract, and the only branch your tests
exercise is the one they were already exercising.

## 3. 🟠 Bare `app.state.router` — fail-open against your own codebase's convention

Three sites: `main.py::models` once, `api/router.py` twice.

```python
router: Router = app.state.router          # KeyError if lifespan hasn't run
```

`app.state.router` is set in `lifespan`. Without it this raises `KeyError`,
which surfaces as an **opaque 500**. Every other state lookup in this
repository already guards:

```python
config = getattr(request.app.state, "config", None)        # main.py:133
key_store = getattr(request.app.state, "key_store", None)  # auth/router.py:63
    -> HTTPException(503, "Key store not initialised — server may still be starting")
```

This is what made one of your own new tests fail in-suite while passing in
isolation: run alone, the fixture's lifespan runs; run after another test, it
does not. That is a *symptom*, not the bug — the bug is that the code cannot
tolerate the state being absent.

Added `_registry(request)` returning a clear 503, and switched `/v1/models`
from the module-global `app` to `request.app`.

## 4. My own failure this round — worth more than either finding

**My first two regression tests were vacuous.** I wrote them, they passed, and
I nearly committed. Then I mutation-tested and *both mutations passed* — the
tests could not fail:

- One hardcoded `producible = {"simulated", "production"}` and compared it to
  the schema. That tests my own assumption, not the code. Reverting
  `"production"` → `"live"` did not touch it.
- The other ran `/v1/models` through a normal `TestClient`, which runs
  lifespan — so `app.state.router` always existed and the guard was never
  exercised.

Rewritten so they drive the real thing: the first calls `_make_runner` with
two genuine `VllmHttpBackend`s and asserts whatever it *actually* emits is in
the schema's `Literal`; the second builds an app **without lifespan**, which is
the condition a normal `TestClient` hides. Reverting either fix now fails.

I have spent this whole loop telling you that a test which cannot fail is
documentation, and I wrote two of them in a row. **Mutation-test every
regression test, including — especially — your own.** The first version always
looks right.

## 5. Open for your next round

1. **`RESEARCH_ASCEND_LOGPROBS.md` is still uncommitted.** It has been through
   two rounds now. Commit it and signal a SHA.
2. **R4** (keystore `hash_scheme`) — descoped by Boss: nothing is deployed, so
   a migration runner would migrate zero rows. The valuable half is the
   *refusal* on an unknown scheme; fold it in cheaply when convenient.
3. **C1 remainder** — the CPU-bound comparison still runs on the event loop.
   Real, still open, deliberately excluded from R7 so the diff stayed
   auditable.
4. **The API is still simulated-only in practice** until `cuda_base_url` /
   `ascend_base_url` are configured. R7 made real backends *reachable*; it did
   not make them *present*. Worth a line in `DEPLOYMENT.md` saying which env
   vars flip it, since `validation_level` now silently tells the difference.
