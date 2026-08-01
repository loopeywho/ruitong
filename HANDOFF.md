# Handoff — Claude → Kimi (RedStar) · updated 2026-08-01 ~02:35, HEAD `13473ff`

**Division of labour (Boss):** Claude audits and sets direction. Kimi builds (primary coder on
RedStar since 2026-07-28). Kimi's edge is what sits behind the firewall — Chinese-language CANN
documentation, Huawei developer forums, Gitee, and mainland resources Claude cannot reach. Route
anything needing those to Kimi.

---

## 🔴 READ THIS FIRST (2026-08-01) — first real cross-vendor data point: NVIDIA A40 vs AMD MI300X

**The actual product thesis, measured for the first time.** `corpora/mi300x.json` (61 prompts,
61/61 bit-exact on its own warm repeat — MI300X is deterministic-when-warm same as the NVIDIA
cards). Compared against the 61-prompt A40 reference: 43/61 identical text (70.5%), worst
token-matched Δprob 0.152 (well inside the 0.4402 gate), top-1 agreement 1.000. **VERDICT: PASS.**
Full report: `reports/cross_vendor_a40_vs_mi300x_61prompt.txt`.

Interesting, not yet a claim: the 29.5% divergence rate is in the same range as both same-vendor
NVIDIA pairs this week (A40-vs-Ada 26%, A40-vs-H100 25%). One data point, one model — no error
bars across pairs — but worth keeping in mind before assuming cross-vendor is categorically worse.

**Took two failed attempts (~$2.40 wasted) to get here — `tools/grab_mi300x.sh` had two real
bugs, both now fixed (`dffef0b`, `fd65ee4`):** (1) no `dockerStartCmd` — vLLM never bound port
8000 in 30 min on env-vars alone; (2) even with the args fixed, still nothing — turned out
`rocm/vllm` (AMD's own image) is **deprecated**, found by reading vLLM's own docs instead of
guessing a third time. Current image: `vllm/vllm-openai-rocm:latest`. If you touch this script
again and it stops working, check whether that tag has moved before re-guessing docker flags.

---

## READ THIS SECOND (2026-07-31) — `compare_corpora.py` was gating on a retired metric

**Third real cross-hardware data point captured: A40 vs H100 (`corpora/h100.json`,
`reports/cross_hardware_a40_vs_h100.txt`).** It printed `VERDICT: FAIL`, but the only failing
line was `top5_set_agreement` at 0.916666667 — the *exact* same number D10/D11 got on A40-vs-Ada,
which D12 (below) already documented as expected-and-not-gated. The tool's `Thresholds` class
had been updated for D12; `tools/compare_corpora.py`'s own gate list had not. Fixed in `464aa8c`
(moved `top5_set_agreement` to reported-only, matching `runner.py`). Re-run: `VERDICT: PASS`.
305/305 tests still pass — nothing asserted the old gate list, which is itself worth noting next
time you add a metric: **add a `TestRunnerGateIntegration`-style test that would fail if a tool
script drifts from `Thresholds`, not just one that checks `runner.py` in isolation.**

RunPod status while we're at it: the old pod (`vjqkdwuzls8hnf`) is gone — fully terminated, not
"stopped" as previously assumed; check pod existence by ID before assuming a stopped pod is still
there. MI300X: still zero stock. A40/H100 both available on demand now if you need more corpus.

---

## READ THIS THIRD — the gate changed AGAIN (D12), and it affects R1

**If you are mid-write on `RESEARCH_ASCEND_LOGPROBS.md`: stop and read this section before
finishing your severity analysis.** I can see it uncommitted at 252 lines. Your conclusions about
what an Ascend `-inf` bug *does to our gate* depend on a gate that changed 20 minutes ago.

**`git pull` / rebase onto `9a8bc4c` before continuing.**

### What changed (DECISIONS.md D12)

The gate is now **compound** — three metrics, ANY failing → FAIL:

| metric | threshold | catches |
|---|---|---|
| `top1_agreement` | ≥ 0.99 | position shifts |
| `probability_mass_delta` | ≤ 0.01 | scaling, corruption, argmax collapse |
| `token_matched_prob_diff` | ≤ **0.4402** (was 0.0022) | value-only corruption at a fixed token position |

`top5_set_agreement` is **demoted to reported-only** — the fourth metric retired by measurement.

### Why this specifically matters to R1

Your committed R1 draft said an Ascend `-inf` bug is dangerous because it would "silently inflate
divergence" against our comparison. Two corrections, and the second one is mine to own:

1. **It cited `cosine_similarity` and `max_absolute_difference`** as what `EquivalenceRunner`
   computes. Both were already retired (D7/D9) before you wrote that. Neither gates anything.
2. **I then told you `-inf` is "not silent" — and I was partly wrong.** I tested that claim on a
   hand-written row and it scored loudly. Measured against the *real* corpus, detectability spans
   ~10⁴×: 0.500 at rank 1 on the worst prompt, but **3.5e-05** on a confident prompt like "What is
   the capital of France?" — genuinely invisible. You were closer to right than my correction was.

**The upshot for your write-up:** `-inf` is neither reliably caught nor reliably missed — it is
*prompt-dependent*. That is precisely why the runner now **refuses** rather than grades: any
prompt with a non-finite logprob is marked `mode="unusable"` and excluded from the verdict
(`count_non_finite`, added in `0d2073e`). Frame the Ascend risk as **"we cannot certify against
this server"**, not as "it inflates our divergence number".

### The one rule that did NOT change

**Do not widen a threshold to make a failing comparison pass.** `0.0022 → 0.4402` looks like
loosening; it is not. The old number was anchored to *simulated* bf16 noise, which D10 measured as
95× too optimistic. The new one is the geometric mean of *measured* real cross-hardware noise
(0.195) and the weakest fault it must catch (0.993) — 2.26× clear on each side. Every threshold in
this repo traces to a measurement you can re-run. A PR that changes one without a new measurement
will be rejected.

### A worked example of the trap, since it nearly caught me

The obvious simplification — drop `token_matched_prob_diff`, gate on the two metrics with perfect
real-hardware records — **has a hole.** A transposed-operator bug (values scrambled, token
identity unchanged) moves `top1_agreement` and `probability_mass_delta` by *exactly zero*:
permuting values within a row changes neither which token sits at rank 0 nor the row's sum. I only
found it because I tested the simplification against every fault *before* implementing it, on the
real corpus rather than by reasoning.

Then a mutation test found a second layer: removing `token_matched_prob_diff` from the *real*
runner's gate broke **no tests at all**, because the sensitivity suite reimplemented the gate
formula instead of exercising `runner.py`. The suite was testing my arithmetic, not the product.
Fixed by `TestRunnerGateIntegration`, which drives the actual `EquivalenceRunner` end-to-end.

**Both lessons are the same one:** verify against the real thing, and check that your test would
actually fail if the code were wrong.

---

## Older notice — superseded by D12 above, kept for context

We rented a real GPU (NVIDIA A40, `Qwen/Qwen3-8B`) for the first time. **It invalidated the
D7 gate.** If you are holding work that touches `equivalence/`, rebase onto `a8722f0` before
continuing — thresholds, the primary metric, and the runner's request pattern all changed.

**What broke, and why it matters to how you write code:**

The old primary metric scored **1.25 between two *correct* executions** against its own **0.05**
threshold. It rejected the reference model compared with itself. 202 tests passed and mypy was
clean the entire time.

The single shared cause of that — and of three other bugs found the same evening — was
**fixtures returning a shape no real server produces**:

| what the fixture returned | what a real vLLM server returns |
|---|---|
| `logprobs: list[float]` | `{"content": [{"token":…, "top_logprobs":[…]}]}` |
| `/v1/models` → bare list | `{"object":"list","data":[{"id":…}]}` |
| unique token strings per row | **nine distinct ids all decoding to `""`** |
| top-10 spread over a narrow range | top-10 spanning logprob 0 to −29 |

Test count, coverage % and a clean type-check **cannot** detect this, because all three measure
agreement with the fixture. Only contact with a real system detects it.

**The three habits that would have caught all four bugs — adopt them:**

1. **Pin at least one test to a payload captured verbatim from a real server**, with a comment
   saying where and when it came from. See `tests/test_vllm_http.py::TestRealModelsEnvelope`.
   If you cannot capture one, say the code is unverified rather than shipping green tests.
2. **Test every comparison metric for reflexivity: `metric(x, x)` must be exactly 0.** One line.
   It would have caught the token-string collision immediately.
3. **Mutation-test your suite.** Break the thing on purpose, confirm tests fail, restore. A test
   that cannot fail is documentation. Worked examples are in this commit's message.

Full write-up: `CALIBRATION.md` (numbers + method), `DECISIONS.md` D9 (what changed and why),
`LESSONS.md` Round 5 (six mistakes, mine, with the rule each one produced).

**Please append your own mistakes to `LESSONS.md` in the same format.** The file exists so the
next model does not repeat what the last one learned — that is worth more than any single fix,
and it is a skill that transfers to every project you work on, not just this one.

### New tools you can use without a GPU

- `tools/capture_corpus.py` — pull a real logprob corpus from any OpenAI-compatible endpoint.
- `corpora/cuda_a40_qwen3_8b.json` — **already captured, committed, 333 KB.** Real Qwen3-8B
  output. Use it instead of writing new fixtures.
- `tools/calibrate_from_corpus.py` — replay fault injection offline; prints the separation
  analysis. Run it after any change to `metrics.py` or `Thresholds`.
- `tools/measure_noise_floor.py` — needs a live endpoint.

**Rule going forward: GPU time is the scarce resource, analysis is free.** Capture once to
`corpora/`, then iterate offline. Never burn GPU hours on something a saved corpus can answer.

---

**Read next:** `LESSONS.md`, `DECISIONS.md`, `PLAN.md` rules 7–10.

**State at handoff:** HEAD `a8722f0`, **212 passed, mypy clean, 80% coverage**. Run
`uv run --extra dev pytest -q` and `uv run mypy src/ruitong` before starting and quote the output.

---

## Ground rules — non-negotiable

1. **Commit before requesting an audit, and cite the SHA.** Three false findings this session came
   from reviewing a tree that was being written to.
2. **Read the file before citing it.** A hallucinated `POST /v1/port` got *built* because an agent
   described a plan it had not opened.
3. **Verdicts come from execution.** Run `pytest` and `mypy`; quote real output. Every serious bug
   found tonight came from *running* code, never from reading it.
4. **A security fix is a code change and needs its own adversarial pass.** Claude's `hmac` fix
   introduced an unauthenticated 500 (`SECURITY_AUDIT.md` M1). Fixes are not exempt.
5. **The CLI is the product** (`DECISIONS.md` D3–D5). No new API surface without a decision entry.

---

## Work queue — ordered

### P0 · finish the CLI test suite (in progress, Qwen)

3 failures remain in `tests/test_cli.py`. They assert the **old, broken** contract — chiefly that
`--target cuda` returns 0. That path used to compare a backend to itself and always pass; it is gone.

New contract to assert:
- exit `0` pass · `1` gate failed · `2` could not run
- no endpoints → `synthetic: true`, `passed: false`, **cannot** exit 0
- report is written **even when the gate fails** (previously only on pass, so every archived
  artifact was a passing one)
- `reference_backend != target_backend`, always

**Also:** make the tests call `main(argv) -> int` **in-process**. `cli.py` currently reports 0%
coverage because subprocess execution cannot be instrumented — the product's primary surface has no
coverage signal. Keep only two subprocess tests to prove packaging.

### P1 · security — availability, all exploitable today from one laptop

Full detail with executed exploits in `SECURITY_AUDIT.md`.

- **H2 · rate limiting runs before auth.** Starlette's `add_middleware` inserts at index 0, so the
  *last* registered runs *first*. Five unauthenticated requests consumed the whole budget and locked
  out the paying caller. Register the limiter **before** `auth_middleware` so it runs after it, and
  key the bucket on the authenticated principal, not the IP.
- **C1 remainder · move the runner off the event loop.** Input is now capped, but the comparison
  loop still never awaits — `run_in_executor` / `anyio.to_thread`, and move `JobStore` calls too.
- **H3 · wire the dead cap.** `count_active()` is defined and called from nowhere. No `DELETE`, no
  TTL, no vacuum; measured **29× disk amplification** (500 KB in → 14.5 MB on disk).
- **S2 · payload cap bypass.** Enforce while reading the body; `Transfer-Encoding: chunked` carries
  no `Content-Length` and skips the check entirely.
- **S3 · unbounded rate-limiter buckets** — evict stale keys.
- **S4 · behind Cloudflare** `request.client.host` is a Cloudflare IP, so all customers share one
  bucket. Read `CF-Connecting-IP` **only** when the peer is a trusted proxy.
- **M2 · must land BEFORE `VllmHttpBackend` is wired to a route.** Backend errors currently relay
  200 bytes of the upstream body to the client — a demonstrated payload contained an internal
  traceback, another customer's model path, and an `hf_token=`.
- **M3/M4/M5** — orphaned background tasks, in-memory job store with multiple workers (18 of 20
  jobs 404'd), Dockerfile runs as root and ignores `uv.lock`.

### P2 · the equivalence gate is mis-calibrated — this is the product

`CALIBRATION.md` and `DECISIONS.md` D7. All measured, no hardware needed.

- **Drop `cosine_similarity`.** Scale-invariant by definition — ×1.01 through ×2.0 all score exactly
  `1.0000000000`. A softmax/temperature bug is mathematically invisible to it. It moved for one of
  five injected faults.
- **Replace full-vocab `max_abs_diff` with top-k (k=10).** The current metric is dominated by the
  vocabulary tail (logprobs −130 to −180, probability ~1e-57). BF16 rounding alone measures
  **0.4929** against a threshold of 0.05 — **every correct port fails**. Restricted to the top 10
  tokens it measures 0.0152, and separation from the weakest fault improves from **1.02× to 15×**.
- **Rename `max_absolute_difference`** — it computes a mean of per-row maxima, then means again
  across prompts. The worst case is the one number a customer cares about and the one the harness
  never computes.
- **Gate on coverage.** 99 of 100 prompts erroring still yields `passed: true`; one catastrophically
  broken prompt in 100 is averaged into silence. Adding prompts currently makes the gate *weaker*.
- **`top_k_agreement` compares position indices, not tokens.** `top_logprobs` — the actual
  `{token_id: logprob}` map — is generated by the fixtures, carried through the schema, and thrown
  away. Same tokens with reordered confidence reports total disagreement; different tokens with
  matching confidence reports perfect agreement.
- **Raise `max_tokens` above 1.** At the runner's own `max_tokens=1`, logprob vectors have length 1,
  which makes cosine, top-1 and top-5 **constant 1.0**. Three of four gates are decoration.
- **Wire `equivalence/faults.py`** into a CI sensitivity suite. It is currently dead code. A fault
  the gate stops detecting is a false PASS the product would ship.

### P3 · needs Chinese-language sources — your edge, not Claude's

- **Confirm `vllm-ascend` + Qwen3-8B on a real Atlas A2 (910B).** Huawei forums, Gitee issues, CSDN
  and Zhihu writeups are where the real install friction is documented. Which CANN / torch_npu /
  vLLM row actually works together in practice, not just on paper?
- **MindIE vs vllm-ascend** — benchmarks suggest MindIE is faster, but its open-source presence is
  thin (16 GitHub stars, development on Gitee). Is it viable for a small team?
- **Ascend access from Hong Kong.** Huawei Cloud International's CN-Hong Kong region has Snt9B
  (`RESEARCH.md`) — confirm real pricing and whether Ascend software packages are permission-gated
  in practice. Verified mainland comparison: China Telecom 天翼云 at **¥38.45/hr** per 910B2 card.
  Treat aggregator pricing as fabricated unless sourced (the widely-quoted "¥19.8/hr 星宇智算" is
  fake — that platform has no Ascend at all).
- **信创 / domestic-substitution procurement.** Is there a mandate with teeth that forces CUDA→Ascend
  migration, and on what timeline? That determines whether this market is pulled or pushed.

---

## What Claude changed this session

Repo placed under git (`4cd7835`) — there was no version control through Phases 0–4 despite ~190
tests, which is how a fabricated citation and a false "7 failing tests" report both survived.

`DECISIONS.md` D1–D7 · `LESSONS.md` · `SECURITY_AUDIT.md` (2 rounds) · `CALIBRATION.md` ·
`QA_PHASE2.md` / `QA_PHASE4.md` · `PHASE_4_5_PROPOSAL.md` · `equivalence/faults.py`.

Fixed: constant-time key comparison (then fixed *that* fix for non-ASCII) · eager per-request config
rebuild · `.gitignore` missing `.env` and `*.db` · CLI self-comparison · no real backend wiring ·
report not written on failure · exit-code contract · unbounded `prompts`.

## Open for Boss — not for you to decide

1. **Export-control counsel** (`RESEARCH.md`). Blocks *commercialisation*, not building. Three
   specific questions; a customer-run CLI is the safer posture (D5).
2. **Per-customer API keys** (`SECURITY_AUDIT.md` H1) — a design decision, and the one that matters
   most commercially. The report *is* the product and the system cannot currently tell one customer
   from another.
3. **Phase 4.5** — calibrate against a real model on CPU before renting Ascend (`PHASE_4_5_PROPOSAL.md`).

---

# ADDENDUM — 2026-07-27 01:30 · the monetising track

Boss's directive is a **monetising pain point on both sides of the globe**. Two things now exist to
serve that, and they are the priority above everything in the queue below.

## The harness is vendor-neutral as of `399861a` (D8)

```
ruitong port <model> --reference nvidia-a100=http://ref:8000 \
                     --candidate amd-mi300x=http://cand:8000
```

Any OpenAI-compatible endpoint. One codebase, both markets — Ascend on the China side, AMD ROCm /
Intel Gaudi / AWS Trainium on the Western side. `cuda`/`ascend` shortcuts still work.

## P0-NEW · Build the public accuracy delta table — `BENCHMARK_SPEC.md`

This outranks the P0/P1/P2 items below. It is the artifact that turns the tool into revenue.

Nobody publishes end-to-end cross-accelerator accuracy deltas — not Huawei, not vllm-ascend (their
CI accepts 5% drift against a hardcoded value with **no GPU baseline column**), not CAICT, not
MLCommons. Publishing it first defines the category.

**v1 is one weekend:** Qwen3-8B, NVIDIA vs AMD, three rows (precision noise floor / cross-vendor /
batch-size effect), 128 prompts, rented by the hour. Legally clean, self-verifiable, transactable
from Hong Kong today. The Ascend column is **v3 and gated on counsel** — everything transfers
unchanged because the harness never sees the silicon.

Read `BENCHMARK_SPEC.md` for method, hard rules and acceptance. The hard rules matter more than the
code: never claim bitwise equivalence, publish results that make us look wrong, ≥3 runs with
variance, worst-case not just mean, `max_tokens ≥ 128` (at 1, three of four metrics are constant
by construction).

## Where Qwen/Kimi are uniquely needed

`STRATEGY.md` lists what only sources behind the firewall can settle. Two are now revenue-critical:

- **PBoC `JR/T 0221-2021《人工智能算法金融应用评价规范》`** — nobody has read it. On its title it is the
  most likely Chinese regulatory hook for compelled equivalence testing. If it mandates anything
  about model evaluation on infrastructure change, the China side gains a *compelled* buyer, which
  is the single thing the market research found missing.
- **信创 procurement acceptance criteria** — does any 国产化替代 standard require 精度对齐 evidence at
  acceptance? Same question, different door.

A compelled buyer changes the business from "persuade someone to care" to "they must buy". Three
Western companies selling *voluntary* AI testing pivoted out in the last two years. That is the
difference these two documents decide.
