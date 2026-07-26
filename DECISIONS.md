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
