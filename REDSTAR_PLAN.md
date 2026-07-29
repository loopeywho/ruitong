# RedStar work plan — Ruitong Bridge (瑞通)

For **Kimi (kimi-k3)** as primary coder on the RedStar profile.
Maintained by Claude (audit + direction). Boss gates all spend and launches.

Open items only. History lives in `DECISIONS.md`, `LESSONS.md`, `CALIBRATION.md`
— read those once, not every round.

**State:** code HEAD post-D12 (2026-07-28) · 258 tests · mypy clean on 26 files.
Verify with `uv run --extra dev pytest -q` and `uv run mypy src/ruitong`, and
quote the real output. Never report a verdict you did not run.

---

## 1. What this product actually sells

A customer moves an LLM from NVIDIA to Huawei Ascend (or AMD, Gaudi, Trainium).
The outputs will not be identical — floating-point addition is not associative,
so different kernels sum in a different order. **We sell the evidence that the
difference is ordinary numerical noise and not a broken port.**

The harness only ever sees logprobs over an OpenAI-compatible HTTP endpoint, so
it has no idea what silicon is behind it. That is deliberate: one codebase
serves the China market (Ascend) and the West (AMD/NVIDIA).

## 2. The gate — read `DECISIONS.md` D12 before touching it

**UPDATED 2026-07-28 (D12). The earlier "the gate is known-wrong, do not
touch it" instruction is superseded — it has now been fixed properly.**

The gate is **compound**: three metrics, ANY failing → overall FAIL.

| metric | threshold | catches |
|---|---|---|
| `top1_agreement` | ≥ 0.99 | position shifts |
| `probability_mass_delta` | ≤ 0.01 | scaling, corruption, argmax collapse |
| `token_matched_prob_diff` | ≤ **0.4402** | value-only corruption at a fixed token position |

**Why all three, and why you must not drop any of them.** The obvious
simplification — gate on `top1_agreement` + `probability_mass_delta`, both of
which have a perfect record on real hardware — was tested against every fault
before being implemented, and **has a hole**: `swap top-2` (a transposed-
operator bug — values scrambled, token identity unchanged) moves both of them
by *exactly zero*. Permuting values within a row changes neither which token
sits at rank 0 nor the row's sum. `token_matched_prob_diff` is the only metric
that catches it, because it matches probability by *token identity* rather
than by position.

Its threshold changed from `0.0022` to `0.4402` — **recalibrated, not
loosened arbitrarily**. The old value came from *simulated* bf16 noise, which
D10 showed is 95× too optimistic. `0.4402` is the geometric mean of measured
real cross-hardware noise (0.195, D11) and the weakest fault in its tier
(0.993): 2.26× clear on each side.

**Rules that still apply:**
- **Do not widen any threshold to make a failing comparison pass.** If a
  correct port fails, the fix is a better metric or a better measurement —
  never a wider number. Every threshold in this repo traces to a measurement
  you can re-run; a PR that changes one without a new measurement will be
  rejected.
- **Do not drop a metric from the gate because it "seems redundant."** Three
  have already been retired *by measurement* (cosine similarity, full-vocab
  max-abs-diff, top-k max-abs-diff) and a fourth in D12
  (`top5_set_agreement` — zero unique coverage, and it sits at 0.9167 on
  every real cross-hardware run regardless of correctness). Retirement
  requires showing the metric catches nothing the others catch.
- **`scale ×1.01` (a 1% temperature error) is a known, disclosed gap** — no
  metric catches it once thresholds tolerate real cross-hardware noise. This
  is asserted deliberately in `TestKnownDetectionGap`. If you close it, say
  so loudly; don't let it change silently.

## 3. Your unique value — do R1 first

Boss's reason for using Chinese models is access this project cannot get
otherwise: **the Chinese-language Ascend ecosystem behind the firewall.**
Claude cannot reach 昇腾社区, Gitee issue threads, or Huawei developer forums.
That makes R1 worth more than any code task in this file.

---

## R1 — Does `vllm-ascend` return usable logprobs? 🔴 blocking

**Why it matters:** the entire product is built on `logprobs` + `top_logprobs`
from an OpenAI-compatible endpoint. If `vllm-ascend` does not return them
correctly, **there is no product on Ascend** and we need to know now, not after
renting 910B time. Precedent: vLLM on AMD ROCm returned `-9999` sentinel values
for logprobs (vllm-project/vllm#19305) — plausible-looking numbers that were
pure garbage.

**Research, in Chinese, using sources Claude cannot reach:**
- `vllm-ascend` on Gitee and GitHub — search issues/PRs for `logprobs`,
  `top_logprobs`, `采样`, `对数概率`, `精度`
- 昇腾社区 (hiascend.com) developer forums
- Any published Ascend-vs-GPU numerical comparison with actual numbers

**Answer these specific questions:**
1. Does `vllm-ascend` populate `logprobs.content[].top_logprobs` at all?
2. Are the values real, or sentinels/placeholders like the ROCm bug?
3. Known precision issues on 910B for softmax / log-softmax specifically?
4. Does `--enforce-eager` behave the same? Is prefix caching supported?
   (We rely on warming both sides — see D9. If Ascend has no prefix cache,
   the warm-up protocol needs rethinking.)

**Deliverable:** `RESEARCH_ASCEND_LOGPROBS.md` — every claim with a URL and a
quoted snippet. **A link with no quote is not evidence.** If the answer is
"nobody has documented this", say exactly that; a confident guess here would
cost Boss real money on 910B rental.

**Acceptance:** Claude can read your file and decide whether to recommend
renting Ascend time, without opening a single link.

---

## R2 — Cross-tenant read on jobs 🔴 security

Logged as P1.5 in the Phase 5 audit (`QA_FINDINGS.md`), still open. The `jobs` table has
no owner column. Now that multi-key auth exists (`auth/keystore.py`), **any
authenticated customer can read any other customer's job** by guessing or
enumerating an id.

**Fix:** add an owner column keyed to `key_id`, scope every read/list/delete to
the authenticated principal, and migrate existing rows to a sentinel owner that
no live key matches (never to "everyone").

**Acceptance — a test that fails before your fix and passes after:** create two
keys, create a job under key A, then assert key B gets **404** (not 403 — do
not confirm the id exists) for get, and that it does not appear in B's list.

## R3 — `POST /v1/port` certifies a self-comparison 🔴

P1.6 from the same audit, still open. Real surface (read the router first —
an earlier draft cited a `/v1/port/comparison` route that does not exist):
`POST /v1/port` and `/v1/port/preview` take `target ∈ {cuda, ascend, auto}`;
for `cuda` or `ascend`, `_make_runner` (`src/ruitong/api/router.py`) builds
`EquivalenceRunner(backend, backend)` — the *same instance* both sides — and
returns `passed: true`. The CLI refuses exactly this (exit 2, D8); the API
certifies it. **A false pass is the worst defect this product can ship.**

**Fix:** reject `target="cuda"`/`"ascend"` with 422 on both routes; never
return `passed: true` from a comparison that made no real comparison.
(`target="auto"` compares two distinct backends and stays; the fakes behind
it are R6/P2.13's problem.)

**Acceptance:** the existing tests asserting a single-target port returns a
passing report are *replaced* by tests asserting 422 and that no report with
`passed: true` is produced (`QA_FINDINGS.md` T1 #9).

## R4 — Keystore hash migration note 🟠 small but sharp

`d668f58` changed key hashing from `hmac.new(...)` to `hashlib.sha256(...)`.
Correct change — but it **invalidates every hash in any existing
`ruitong-keys.db`**. Harmless today (nothing deployed); fatal the day after
launch.

**Fix:** add a `hash_scheme` column defaulting to `sha256`, and refuse to
authenticate rows with an unknown scheme rather than silently failing them.
Document in `DEPLOYMENT.md` that any pre-`d668f58` keystore must be recreated.

**Acceptance:** a test with a row written under the old HMAC scheme asserts a
clear refusal, not a confusing auth failure.

## R5 — Widen the prompt corpus ✅ DONE (Claude, 2026-07-28, Boss reassigned)

The headline finding — 19% of prompts diverge across two NVIDIA GPUs — rests on
**3 divergences in 16 prompts**. That confidence interval is far too wide for
the number now published on the site.

**Task:** extend `PROMPTS` in `tools/capture_corpus.py` to ~60, keeping the
existing 16 unchanged so past corpora stay comparable. Spread across:
short factual, long generation, code, reasoning, 中文, and translation. Bias
toward prompts with genuine near-ties (that is where divergence happens).

**Done:** 61 prompts; first-16 sha256 verified byte-identical
(`7a2045b3…`); `--help` runs; `corpora/` untouched. Kimi: nothing to do here —
the next capture run will use the wider list automatically.

---

## R6 — remaining P2s, in this order 🟡 after R1–R4

The Phase 5 audit's open P2 list (`QA_FINDINGS.md` tail), re-ranked by what
actually protects the product. Do them top-down; stop and report rather than
rushing the tail:

1. **P2.7 — global `"anonymous"` bucket.** With auth disabled, every client
   shares one rate-limit bucket: one noisy user exhausts it for everyone, and
   the fallback masks misconfigured auth. When `api_key` is unset, key the
   bucket on client IP — and log a warning at startup that auth is off.
2. **P2.9 — tautological test.** A test that cannot fail is worse than no test:
   it manufactures confidence. Find it, make it assert the real behaviour, then
   mutation-test it (break the code, confirm the test fails, restore).
3. **P2.5 — validate `create_key` body.** Pydantic model, `name` length-capped
   (e.g. 1–128 chars), reject unknown fields. An admin endpoint is still an
   input surface.
4. **P2.10 — exercise `lifespan` in tests.** Use the real app via
   `TestClient(app)` as a context manager so startup wiring runs. The replica
   app (`_make_test_app`) already drifted from production once; this is the
   same failure class.
5. **P2.11 — pricing auth test.** Assert unauthenticated requests to pricing
   endpoints are rejected when auth is configured.
6. **P2.6 — pricing config validation.** Malformed `RUITONG_PRICING` JSON must
   fail loudly at startup, not serve a half-parsed price list. Wrong prices are
   a monetary bug, silently.
7. **P2.8 — audit logging.** Log admin-API actions (key created/revoked, by
   which principal, when) to stderr/structured log. No new dependencies.
8. **P3.6, promoted — the API report quotes only retired metrics.**
   `_report_to_response` emits `cosine_similarity` and `max_absolute_difference`
   — both retired by D9/D10 — and omits the metrics that actually gate, so a
   consumer cannot see why `passed` is what it is. Per D3/D5 the report *is*
   the product; a report built from dead numbers is a trust bug, not polish.
   Sync the response models to `report.metrics` and `Thresholds` as defined in
   `equivalence/` — mirror them exactly, do not redefine them (my lane).
9. **P2.13 — carryovers.** Fake backends still registered in `main.py`
   lifespan; job concurrency cap; orphaned tasks. Take the fake-backend one
   first: production serving fabricated responses is the oldest open finding
   in this repo (Phase 2 audit).

P3.1–13 (polish): **leave until every P1/P2 above is closed.** Polish on top of
an open security finding is misordered work.

## Lane separation — avoid collisions

| Mine (Claude) | Yours (RedStar) |
|---|---|
| `src/ruitong/equivalence/` | `src/ruitong/api/`, `auth/`, `pricing/`, `jobs/` |
| `tools/`, `corpora/`, `reports/` | `src/ruitong/main.py` |
| `CALIBRATION.md`, `DECISIONS.md` | `QA_FINDINGS.md`, `DEPLOYMENT.md` |
| thresholds and metric definitions | `RESEARCH_ASCEND_LOGPROBS.md` |

Exception: the `PROMPTS` list in `tools/capture_corpus.py` is yours for R5;
the capture logic around it stays mine. Any other file in my lane: say so in
your report instead of editing it.

## Non-negotiable rules

1. **Commit before requesting an audit, and cite the SHA.** Three false
   findings came from reviewing a tree being written to.
2. **Read the file before citing it.** A hallucinated `POST /v1/port` endpoint
   once got *built* because an agent described a plan it had not opened.
3. **Verdicts come from execution.** Run `pytest` and `mypy`, quote real
   output. Every serious bug in this repo came from *running* code, never from
   reading it.
4. **A fix is a code change and needs its own adversarial pass.** A previous
   `hmac.compare_digest` fix introduced an unauthenticated 500.
5. **Never edit `corpora/*.json` or `reports/*`.** Measurements, not source.
6. **Boss gates all spend.** Never rent hardware. If a task seems to need a
   GPU, say so and stop.

## The habit that would have caught nearly every bug here

Four separate bugs, one root cause: **a fixture returning a shape no real
server produces.** `logprobs: list[float]` vs the real object; `/v1/models` as a
bare list vs `{"data": […]}`; nine distinct token ids all decoding to `""`;
ISO-8601-with-`T` vs SQLite's space separator. Every one passed its tests,
because the test used the same wrong format as the code.

So, three habits:

1. **Pin at least one test to a payload captured verbatim from a real system**,
   with a comment saying where and when it came from. See
   `tests/test_vllm_http.py::TestRealModelsEnvelope`.
2. **Assert reflexivity on every comparison: `metric(x, x)` must be exactly 0.**
   One line; it caught a metric scoring 0.67 against itself.
3. **Mutation-test your suite** — break the thing on purpose, confirm the tests
   fail, restore. A test that cannot fail is documentation.

**Log your own mistakes to `LESSONS.md` in the existing format.** That file
exists so the next model does not repeat what the last one learned, and the
habit transfers to every project you work on — not just this one.

## Reporting

Append to `QA_FINDINGS.md` (never overwrite — it was clobbered once, losing a
whole audit). Per task: what changed, the SHA, the real `pytest`/`mypy` output,
and anything you could not verify. **"I could not confirm X" is a valid and
valuable report.** A confident wrong answer costs more than an honest gap.

---

## R7 — Wire `/v1/port` to the registry (the remaining half of P2.13) 🔴

**Boss dispatched 2026-07-29, ahead of R4.** R4 (keystore `hash_scheme`) is
deferred: nothing is deployed, so it would migrate zero rows.

**Do not redo these — already done in `667a808`:** the job concurrency cap is
wired (`router.py:160`), and orphaned background tasks are tracked and drained
on shutdown. `main.py::lifespan` already registers a real `VllmHttpBackend`
when `config.cuda_base_url` / `ascend_base_url` are set, falling back to fakes
otherwise. **That part is correct — leave it alone.**

### The actual gap

`_make_runner` (`src/ruitong/api/router.py:27`) **ignores the registry
entirely** and hardcodes `FakeCuda()` / `FakeAscend()` on every call. So the
config plumbing exists and is simply never used: the whole `/v1/port` surface
returns reports built from fixtures, no matter how the server is configured.
The CLI is currently the only path that can reach real hardware.

### Scope

1. **`_make_runner` pulls from the registry** (`request.app.state.router` /
   `BackendRegistry`) instead of constructing fakes. It needs access to the
   request, so its signature changes.
2. **`GET /v1/models`** (`main.py:107`) does the same — it currently
   instantiates `FakeCuda`/`FakeAscend` per request unconditionally.
3. **`validation_level` must tell the truth.** It is currently hardcoded to
   `"simulated"` at both call sites. It must be derived from what actually ran:
   `"simulated"` when either side is a fake, something else when both are real.
   **This is the most important item in R7** — a report that says `simulated`
   while using real hardware is merely unhelpful, but one that omits the label
   while using fakes is a false trust claim, which is the worst defect this
   product can ship (D5).

### Acceptance

- With no endpoints configured: `/v1/port` still works, still returns
  `validation_level="simulated"`, and `TestPortEndpoint` passes unchanged.
- With both endpoints configured (point them at a stub HTTP server, not real
  hardware — no spend): the report's `validation_level` is **not**
  `"simulated"`, and the fakes are never constructed. Assert the second part by
  patching `FakeCuda.__init__` to raise, or by asserting on the backend names
  in the report.
- A test asserting `_make_runner` returns registry-held instances, not fresh
  fakes.

### Explicitly OUT of scope

- The C1 remainder (CPU-bound comparison on the event loop). Real, still open,
  but it is a performance concern with no correctness or trust consequence —
  and mixing it into this change would make the diff hard to audit.
- Renting hardware. Use a stub HTTP server; `tools/mock_vllm.py` already exists
  for exactly this.
