# 瑞通 Ruitong — Decision Record

Append-only. Newest last. Short by design — this is read every round.
Rationale lives here; `PLAN.md` carries the instructions that follow from it.

---

## D1 · 2026-07-26 — Serving stack: `vllm-ascend`, not ONNX Runtime

Both backends are vLLM over OpenAI-compatible HTTP. The ONNX/CANN-EP path was investigated and
rejected — no dynamic shapes, no generation loop for CANN, zero public precedent. Evidence in
`RESEARCH.md`. **Consequence:** the Ascend backend is a second instance of the CUDA client, not a
research project.

## D2 · 2026-07-26 — Reference model: `Qwen3-8B`

Not Qwen2.5-7B, which vllm-ascend deprecated and stripped from CI in Apr 2026 and which carries an
open crash bug plus prefix-cache corruption. Qwen3-8B is Core-tier with live CI.

## D3 · 2026-07-26 (Boss) — CLI-first, not API-first

`ruitong port <model> --target ascend` → artifacts + equivalence report on disk. The runner is a
library; the CLI wraps it. The report on disk **is** the deliverable — it's what a customer is
handed as evidence.

## D4 · 2026-07-27 — `/v1/port` was built anyway. Ruling: keep the code, hold the gate.

**What happened.** `POST /v1/port` and `POST /v1/port/preview` were implemented with SQLite job
persistence, API keys, rate limiting and payload caps — the API-first path D3 declined. Worth
recording *how*: `POST /v1/port` was an endpoint an agent **hallucinated** into a description of
PLAN.md hours earlier. It never existed in the plan. It has now been built. A fabricated detail
became a build target.

**Ruling — not a revert.** The code exists, is tested, and D3 said "CLI-first", not "CLI-only".
Deleting working code is waste. But three constraints bind:

1. **The CLI is the product.** It must run standalone, with no HTTP server, no job store, no auth.
   If `ruitong port` ever requires the API to be up, that is a regression.
2. **The API is not deployed.** `PLAN.md` Phase 6 is unapproved and unchanged. Building it is fine;
   exposing it is gated on the security review *and* D5 below.
3. **No new API surface** without a decision recorded here.

## D5 · 2026-07-27 — A customer-run CLI is the safer export-control posture

This upgrades D3 from a product preference to a compliance one, and it is the strongest argument
for the CLI that nobody had made yet.

Per `RESEARCH.md`: BIS GP10 guidance covers **"use"** and **"otherwise service"** of Ascend
910B/C/D. Those verbs attach to whoever operates the hardware and whoever services it.

- **If we host a porting service**, we operate against Ascend on a customer's behalf, continuously,
  as a commercial offering. That is the fact pattern the guidance describes most directly.
- **If the customer runs our CLI on their own hardware**, they operate; we ship a tool. Materially
  different posture — and it keeps **15 CFR §734.7** live, under which *published* software leaves
  EAR jurisdiction entirely. A hosted service cannot use that exit. An open-sourced CLI can.

**Direction:** the monetisable artifact is the **equivalence report and the porting recipe**, not
compute we rent out. Sell the proof, not the GPU-hours. This also removes the worst margin problem
in the business — we never pay ¥38.45/hr; the customer does, on hardware they already own.

Still requires counsel (see `RESEARCH.md`). This narrows what counsel must answer, it does not
replace them.

## D6 · 2026-07-27 — Repo is under git as of `4cd7835`

No version control existed through Phases 0–4 despite ~190 tests. Every prior audit reviewed whole
files with no way to pin a revision, which is how a fabricated citation and a false "7 failing
tests" report both survived. **Every audit from here cites a SHA.**

## D7 · 2026-07-27 — The equivalence gate is mis-calibrated; measured replacement adopted

Calibrated against injected faults on CPU (`CALIBRATION.md`, `equivalence/faults.py`). Three defects,
all measured:

1. **`max_abs_diff ≤ 0.05` rejects correct ports.** BF16 rounding alone measures **0.4929** — 10×
   the threshold. Every correct port would fail.
2. **Root cause: the metric measures the wrong region.** Largest BF16 differences fall at logprobs
   of −130 to −180 (probability ~1e-57, never sampled). Across the top-10 tokens the difference is
   **0.0152**. Separation between correct (0.4929) and a real fault (0.5038) is **2.2%** — no
   threshold can work.
3. **`cosine_similarity` cannot detect scaling errors.** It is scale-invariant by definition:
   ×1.01 through ×2.0 all score exactly 1.0000000000. It moved for one of five injected faults.

**Adopted gate** — top-k (k=10) max_abs_diff ≤ 0.05 · probability mass |Σp−1| ≤ 0.01 · top-1 ≥ 0.99
· top-5 set ≥ 0.95. **Drop cosine similarity and full-vocab max_abs_diff.** Separation improves from
1.02× to **15×**.

Thresholds are measured, not assumed. Re-measure against a real model (Qwen2.5-0.5B, FP32 vs BF16,
CPU) before shipping — `PHASE_4_5_PROPOSAL.md` §4.5a.

**Why it matters:** per D5 the report *is* the product. A gate that fails correct ports destroys
trust on contact; one blind to a fault class ships a silent false PASS the customer acts on.

## D8 · 2026-07-27 — One harness, both markets (vendor-neutral backends)

**Boss's directive: a monetising pain point on BOTH sides of the globe.** That is now implemented
rather than argued about.

The harness only ever sees logprobs over an OpenAI-compatible endpoint. It has no knowledge of the
silicon behind it. So the same tool serves:

- **China side** — Ascend via `vllm-ascend` (`--ascend-endpoint`)
- **Western side** — AMD ROCm, Intel Gaudi, AWS Trainium, and any other OpenAI-compatible server
  (`--reference NAME=URL --candidate NAME=URL`)

No second codebase, no fork. `cuda`/`ascend` remain as convenience shortcuts; the neutral form takes
arbitrary backend names.

**Why this matters commercially, not just technically:**

1. **Two revenue sides from one build.** The China side has the market (documented pain: vllm-ascend
   issue #31 open 17 months, 7 correctness issues last month). The Western side has *reachability* —
   AMD and NVIDIA are rentable by the hour from Hong Kong, customers can pay a HK company, and no
   export-control question applies.
2. **The Western side is where we can self-verify.** Boss has no independent way to QC Ascend
   numerics and no legal path to the hardware. On AMD/NVIDIA we can run it ourselves, which is the
   only way to earn the right to make an equivalence claim at all.
3. **It de-risks the Ascend side rather than abandoning it.** Everything proven on
   NVIDIA↔AMD — the metrics, the tolerances, the report format, the fault-detection sensitivity —
   transfers unchanged to Ascend the moment counsel and hardware access clear.

Self-comparison is refused on this path too: identical reference and candidate names exit 2.

Verified: 194 passed, mypy clean.

---

## D9 — Real hardware retires the D7 gate; cache state becomes a controlled variable

**2026-07-27 · first measurement on a real GPU** (NVIDIA A40, `Qwen/Qwen3-8B`,
RunPod, ~$0.44/hr). Full numbers and method in `CALIBRATION.md`.

D7 chose `top_k_max_abs_diff` as the primary gate, calibrated against synthetic
logprobs because no hardware was available. Real model output invalidated that
choice in three ways at once. Every finding below came from *running* code
against a real server; none was visible from reading it, and 202 green tests
agreed with the synthetic fixtures throughout.

### 1. The gate rejected the reference model compared with itself

On real data, two **correct** executions of the same prompt on the same GPU —
differing only in prefix-cache state — scored up to **1.25**, against a
threshold of **0.05**. The weakest genuine fault scores **0.406**. Noise and
faults overlap, so no threshold exists that separates them.

Two independent causes, both invisible in synthetic data:

- **Ranks, not tokens.** Tail tokens are near-tied, so a negligible
  perturbation swaps their order and the metric compares two unrelated tokens.
  Observed: rank 3 held `'高'` on one side and `' like'` on the other.
- **Log space, not probability space.** The top-10 of an 8B model spans
  logprob 0 to −29. A 0.5 shift at −26 (p ≈ 3e-12) scores the same as 0.5 at
  the argmax, which flips the output.

**Decision:** `top_k_max_abs_diff` is demoted to reported-only, joining
`cosine_similarity` and full-vocab `max_absolute_difference`. Three metrics
have now been retired by measurement rather than opinion.

### 2. The replacement: `token_matched_prob_diff`

Matches tokens **by identity**, compares **probabilities**. Calibrated at
**2.2e-03** — the geometric mean of the measured noise ceiling (bf16 rounding,
1.293e-03) and the weakest fault (1% temperature error, 3.660e-03).

The 1.7× margin is thin, and is published rather than hidden: a temperature
error below ~1% is at or past the detection limit.

Implementation trap worth remembering: **token strings are not unique within a
row.** One position in the corpus held *nine distinct token ids all decoding to
`""`*. Keying a dict on the string kept only the last, and the metric scored
**0.67 comparing a tensor with itself**. Repeated strings are paired by order
of appearance.

### 3. Prefix-cache state is now controlled, not tolerated

A prompt's first execution takes **zero** cache hits and returns different
logprobs from every later one. Proven from vLLM's own counters: cold call +0
hits, warm call +16. The resulting difference reaches **8.65e-02** — 24× the
weakest fault — while leaving the output text identical.

A gate wide enough to tolerate that would pass a demoted argmax. So the runner
now warms **both** backends on every prompt and discards the result before
measuring (`EquivalenceRunner._warm`).

**This is a methodological differentiator, not just a bug fix.** A naive
CUDA-vs-Ascend benchmark that does not control cache state measures the cache,
not the silicon — and whichever side happens to be colder is reported as
broken. Once warm, the reference contributes *zero* noise (bit-exact, 16/16
prompts), which is what makes an attribution claim possible at all.

### Corpus over live calls

`corpora/cuda_a40_qwen3_8b.json` (333 KB, committed) is captured once and
replayed offline. GPU time is the scarce resource; analysis is free. The
sensitivity suite now runs against real model output instead of fixtures — the
previous suite passed while the gate it guarded was rejecting correct ports.

### What this does not establish

One GPU, one model, one vendor. bf16 rounding is a *proxy* for two hardware
kernels disagreeing. No Ascend hardware has been touched, so the CUDA↔Ascend
claim remains unevidenced.

Verified: 212 passed, mypy clean, sensitivity suite mutation-tested (removing
the warm-up pass fails 2 tests; widening the threshold to 0.1 fails 4).

---

## D10 — Real cross-silicon measurement invalidates the D9 threshold

**2026-07-28 · first two-GPU comparison.** NVIDIA A40 (Ampere) vs NVIDIA RTX
6000 Ada, both 48 GB, both `vllm/vllm-openai:latest` from the same RunPod
template, same `Qwen/Qwen3-8B`, `temperature=0, seed=1234`, both captured warm
and both bit-exact on their own warm repeat (16/16 each). The only variable is
the silicon.

### Result

| | |
|---|---|
| Identical output text | **13/16** |
| Diverged token stream | **3/16** |
| Worst token-matched Δprob (pre-divergence) | **0.1224** |
| Top-1 agreement (pre-divergence) | **1.000** |
| Top-5 set agreement | 0.917 |
| Probability-mass delta | 0.0000298 |

**Two NVIDIA GPUs running an identical stack produce different text on 19% of
prompts.** Not different distributions — different sentences.

### What this does to the D9 gate

| | value | ratio to real noise |
|---|---|---|
| simulated bf16 noise (D9 calibration basis) | 1.29e-03 | 95× too small |
| `TOKEN_MATCHED_PROB_DIFF_MAX` threshold | 2.20e-03 | 56× too small |
| weakest injected fault (1% temperature error) | 3.66e-03 | 33× too small |
| scale ×1.05 fault | 1.80e-02 | 6.8× too small |
| **measured cross-silicon noise** | **1.22e-01** | — |

Real cross-silicon noise is **6.8× larger than a ×1.05 scaling fault.** So
`token_matched_prob_diff` cannot separate a *correct* port onto different
silicon from a genuine temperature bug. Its noise and fault distributions
overlap, exactly as `top_k_max_abs_diff`'s did in D9 — one level up.

**bf16 rounding is not a valid proxy for two hardware kernels disagreeing.**
It understates the real effect by ~95×. Every threshold derived from it —
D7's and D9's — was derived from the wrong distribution. This is the third
metric retired by measurement, and the second threshold invalidated by
getting closer to real hardware.

### What survived

- **Top-1 agreement = 1.000.** The argmax agreed at *every* compared position
  on *every* prompt. Where the models diverge, they diverge by picking a
  different token at a near-tie — not by disagreeing about what is likely.
- **Probability-mass delta = 3e-05**, four orders below its 0.01 tolerance,
  and still moves ~0.023 under a ×1.05 scaling fault. It remains a working
  scaling-fault detector.

### Decision

**Do not re-tune `TOKEN_MATCHED_PROB_DIFF_MAX` to 0.15 and call it fixed.**
That would buy a gate that passes a ×1.05 temperature bug. The honest reading
is that top-10 distribution distance is *not the discriminating signal across
hardware* — legitimate silicon variation swamps it.

The claim the product can defend, on this evidence, is:

1. **Top-1 agreement at every position before divergence** — held at 1.000.
2. **Divergence rate** — 3/16 here; the number a customer actually feels.
3. **Probability-mass preservation** — catches the scaling class independently.

Thresholds are therefore left **unchanged and known-wrong** rather than
silently widened, and the gate is documented as failing this comparison. A
threshold change needs a larger corpus than 16 prompts: 3 divergences gives a
confidence interval far too wide to set a limit on.

### Cost and honesty note

Two pods, ~25 minutes, well under $1. The finding that the whole D9
calibration rested on a 95×-optimistic proxy was worth more than every
synthetic experiment before it. Boss pushed for this after noticing the RunPod
balance had barely moved — the spend was low precisely because only one GPU
had ever been rented, and one GPU cannot measure a port.

Reproduce: `python tools/compare_corpora.py corpora/a40_v3.json corpora/rtx6000ada.json`
