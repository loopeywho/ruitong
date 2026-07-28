# RedStar work plan — Ruitong Bridge (瑞通)

For **Kimi (kimi-k3)** as primary coder on the RedStar profile.
Maintained by Claude (audit + direction). Boss gates all spend and launches.

Open items only. History lives in `DECISIONS.md`, `LESSONS.md`, `CALIBRATION.md`
— read those once, not every round.

**State:** HEAD `d668f58` · 236 tests · mypy clean on 26 files · 80% coverage.
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

## 2. The one thing you must not undo

**The gate is currently known-wrong, on purpose.** Read `DECISIONS.md` D10.

Measured 2026-07-28 on two real NVIDIA GPUs (A40 vs RTX 6000 Ada), identical
stack, only the silicon differing:

- **3 of 16 prompts produced different text.**
- Worst token-matched Δprob **0.122**, against a threshold of **0.0022**.

So the gate fails a *correct* port between two NVIDIA cards. **Do not "fix"
this by widening `TOKEN_MATCHED_PROB_DIFF_MAX`.** Real cross-silicon noise
(0.122) is 6.8× *larger* than a ×1.05 temperature fault (0.018), so any
threshold loose enough to pass the former also passes a genuine bug. The
thresholds stay unchanged and documented as failing until there is a larger
corpus. A PR that widens them will be rejected.

Three metrics have now been retired **by measurement**: cosine similarity,
full-vocab max-abs-diff, and top-k max-abs-diff. What survived: **top-1
agreement**, **divergence rate**, **probability-mass preservation**.

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
- 昇腾社区 (ascend.huawei.com) developer forums
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

Your own audit logged this as P1.5 and it is still open. The `jobs` table has
no owner column. Now that multi-key auth exists (`auth/keystore.py`), **any
authenticated customer can read any other customer's job** by guessing or
enumerating an id.

**Fix:** add an owner column keyed to `key_id`, scope every read/list/delete to
the authenticated principal, and migrate existing rows to a sentinel owner that
no live key matches (never to "everyone").

**Acceptance — a test that fails before your fix and passes after:** create two
keys, create a job under key A, then assert key B gets **404** (not 403 — do
not confirm the id exists) for get, and that it does not appear in B's list.

## R3 — `/v1/port/comparison` returns `passed: true` for self-comparison 🔴

Your P1.6, still open. The CLI refuses this (`cli.py` rejects same name *and*
same URL — "a backend compared with itself always agrees and certifies
nothing"). The HTTP endpoint does not. **A false pass is the worst defect this
product can ship**, because the whole value is the trust claim.

**Fix:** mirror the CLI's refusal in the API. Reject identical endpoints with
422, and never return `passed: true` from a comparison that made no real
comparison.

**Acceptance:** a test posting the same URL as both reference and candidate
asserts a 4xx and that no report with `passed: true` is produced.

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

## R5 — Widen the prompt corpus 🟢 no GPU needed

The headline finding — 19% of prompts diverge across two NVIDIA GPUs — rests on
**3 divergences in 16 prompts**. That confidence interval is far too wide for
the number now published on the site.

**Task:** extend `PROMPTS` in `tools/capture_corpus.py` to ~60, keeping the
existing 16 unchanged so past corpora stay comparable. Spread across:
short factual, long generation, code, reasoning, 中文, and translation. Bias
toward prompts with genuine near-ties (that is where divergence happens).

**Acceptance:** `python tools/capture_corpus.py --help` still runs, the first 16
prompts are byte-identical to now, and you have **not** touched
`corpora/*.json` — those are captured measurements, never edited by hand.

---

## R6 — remaining P2s, in this order 🟡 after R1–R4

Your open P2 list, re-ranked by what actually protects the product. Do them
top-down; stop and report rather than rushing the tail:

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
8. **P2.13 — carryovers.** Fake backends still registered in `main.py`
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

If a task needs a file in my lane, say so in your report instead of editing it.

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
