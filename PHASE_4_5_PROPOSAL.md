# Proposal — Phase 4.5: validate the harness on real numerics, before any hardware

**Author:** Claude (Sonnet 5) · 2026-07-27 · **Status:** proposed, awaiting Boss
**Cost:** zero. No GPU, no NPU, no cloud account, no export-control exposure.

---

## The problem this closes

Every metric, threshold and gate in the equivalence harness has only ever been exercised against
`FakeCuda` and `FakeAscend` — deterministic fixtures that return canned numbers. The harness has
**never seen output from a real model.**

That matters because the thresholds in `PLAN.md` (cosine ≥ 0.99, max-abs-diff ≤ 0.05, top-1 ≥ 99%,
top-5 set ≥ 95%) were chosen analytically, not calibrated. Nobody knows:

- What cosine similarity two *genuinely equivalent* runs actually produce.
- What a *genuinely broken* port looks like on these metrics.
- Whether the gate has any discriminating power at all between those two cases.

Right now the plan is to answer those questions for the first time on **rented Ascend hardware at
¥38.45/hr**, after a legal review, on the one run that is supposed to prove the product works. That
is the most expensive and least forgiving place to discover the methodology is mis-calibrated.

## The insight

**Validating the *harness* does not require Ascend.** It requires two implementations that produce
genuinely different floating-point results for the same model — which is exactly what you get from
any two different inference paths, including two on CPU.

The harness cannot tell the difference between "CUDA vs Ascend" and "implementation A vs
implementation B". It only sees logprobs over HTTP. So we can calibrate it now.

## Proposed work

**Reference model:** something tiny — `Qwen2.5-0.5B-Instruct` or smaller. Nothing here depends on
model quality; it depends on numerics being real.

### 4.5a — Establish the noise floor (what "equivalent" looks like)

Run the same model twice through paths that *should* agree, and record what the metrics actually say:

- same backend, same prompts, two separate runs (pure run-to-run noise)
- FP32 vs BF16 on the same backend (precision-induced difference — the closest analogue to a
  cross-kernel port)
- batch size 1 vs batch size 8 (batching changes reduction order, so results genuinely differ)

**Output:** the real distribution of cosine / max-abs-diff / top-1 / top-5 for *equivalent* runs.
That is the empirical basis for the thresholds. If BF16-vs-FP32 on identical hardware already
violates `max_abs_diff ≤ 0.05`, the threshold is wrong and would have failed every real port.

### 4.5b — Establish the detection floor (what "broken" looks like)

Inject faults that mimic real porting failures and confirm the gate **fails**:

| Injected fault | Mimics |
|---|---|
| Swap two tokens' logprobs | operator returning transposed output |
| Truncate the vocabulary tail | incorrect vocab-size handling |
| Scale all logprobs by 1.05 | temperature/softmax scaling bug |
| Shift by one position | off-by-one in KV-cache indexing |
| Replace 1% of positions with noise | intermittent kernel fault |

**Output:** a sensitivity table — which faults the gate catches, and the smallest magnitude of each
that it still detects. **Any fault the gate does not catch is a false-PASS the product would ship.**

### 4.5c — Calibrate and record

Set thresholds from 4.5a/4.5b so that the equivalent-run distribution passes and every injected
fault fails, with margin. Record the measured numbers and the reasoning in `DECISIONS.md`.

## Acceptance

- A committed calibration report with real measured numbers, not analytic guesses.
- Every threshold in `PLAN.md` traceable to a measurement.
- A fault-injection suite in CI that fails the build if the gate stops detecting a known-bad port.
- Runs on CPU in minutes; no network, no accelerator.

## Why this is worth doing before Phase 5

1. **It de-risks the expensive step.** By the time we rent Ascend, the only unknown left is Ascend
   itself — not whether our own measurement works.
2. **The fault-injection suite is the real regression test for the product.** Test count and
   coverage say nothing about whether a gate *discriminates*. This does.
3. **Zero legal exposure.** Nothing here touches Ascend hardware, so it proceeds while the GP10
   question sits with counsel. It converts blocked time into progress.
4. **It is a sellable artifact by itself.** "Here is our fault-detection sensitivity table" is
   exactly the evidence a buyer needs to trust an equivalence report — and per `DECISIONS.md` D5
   the report *is* the product.

## Risk if skipped

The harness passes 194 tests and could still be measuring nothing. Fakes agree with fakes by
construction. The first real disagreement it ever sees would be on paid hardware, and if the
thresholds are wrong in the permissive direction the failure is **silent** — a green report on a
broken port. That is the single worst outcome for this business, because the report is what the
customer is buying.
