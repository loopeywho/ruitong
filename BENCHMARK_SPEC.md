# 瑞通 Ruitong — Public Accuracy Delta Table · build spec

**Purpose: the monetising wedge.** This is the artifact that turns a tool nobody knows about into
a reference everyone cites. It is the cheapest credible entry into *both* markets.

**Implementer: Qwen / Kimi.** **Reviewer: Claude.** Read `LESSONS.md` and `PLAN.md` rules 7–10 first.

---

## Why this is the wedge, in one paragraph

Everyone in this space tells you to run an accuracy test, and **nobody publishes a number to compare
against.** Huawei's `msprobe` gives you thresholds at the *operator* level and nothing end-to-end.
`vllm-ascend`'s own accuracy CI accepts **5% drift against a hardcoded YAML value with no GPU
baseline column at all**. MLCommons publishes 99%/99.9%-of-FP32 for a handful of benchmark models
and nothing for yours. CAICT's national programme scores five dimensions and equivalence is not one
of them.

So the table below does not exist anywhere. Publishing it first defines the category, and the
category is the product.

## What to publish

> **"The same model, the same prompts, different accelerators. Here is exactly how much the answers
> differ — with the method, the tolerances, and the raw data."**

Dated, versioned, reproducible by anyone with the command line in the appendix.

### The table

| Model | Reference | Candidate | top-k max Δlogprob | prob mass | top-1 agree | top-5 set | verdict |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | A100 fp16 | A100 bf16 | | | | | |
| Qwen3-8B | A100 bf16 | MI300X bf16 | | | | | |
| Qwen3-8B | A100 bf16 bs=1 | A100 bf16 bs=8 | | | | | |
| Llama-3.1-8B | … | … | | | | | |
| Qwen3-8B W8A8 | A100 bf16 | A100 W8A8 | | | | | |

**The three rows that matter most, and why:**

1. **Same hardware, different precision** (fp16 vs bf16) — establishes the *noise floor*. Everything
   else must be read against it. Without this row the table means nothing.
2. **Different vendor, same precision** (NVIDIA vs AMD) — the actual question a migrating customer
   has.
3. **Same hardware, different batch size** — proves the number moves for reasons that have nothing
   to do with vendors. Published research shows bf16 accuracy varying **up to 9%** from GPU count
   and batch size alone. Omitting this row would make the vendor row look damning when it isn't,
   and someone would catch it.

## Method — non-negotiable, this is what makes it citable

1. **Use the harness**, vendor-neutral form:
   ```
   ruitong port <model> \
     --reference nvidia-a100=http://ref:8000 \
     --candidate amd-mi300x=http://cand:8000 \
     --output results/<model>__<ref>__<cand>.json
   ```
2. **Pin and publish everything**: model weights **SHA256**, vLLM version, ROCm/CUDA/driver
   versions, GPU model, batch size, `max_tokens`, temperature (0), seed, prompt set with its own
   hash. A number without provenance is not evidence.
3. **Set `max_tokens` well above 1.** At `max_tokens=1` three of the four metrics are constant 1.0
   by construction — the vectors have length 1. Use ≥ 128.
4. **Publish the raw JSON reports**, not just the table. The table is the headline; the JSON is why
   anyone believes it.
5. **State the tolerance and where it came from.** Ours is measured — see `CALIBRATION.md`:
   bf16 rounding gives top-k max Δ of **0.0152**; the weakest injected fault gives **0.2341**.
   Cite MLPerf's 99%/99.9%-of-FP32 as the only existing public precedent.
6. **Report worst case, not just mean.** A mean over prompts hides exactly the single bad position a
   customer cares about. Report p50 / p99 / max.

## Hard rules — breaking any of these destroys the artifact's value

- **Never claim bitwise equivalence.** It is not achievable across vendors and claiming it would be
  fraud. Everything is statistical, tolerance-banded, and says so.
- **Publish results that make us look wrong.** If AMD and NVIDIA diverge more than expected, publish
  it. The table's entire value is that it is disinterested. One massaged number and it is marketing.
- **No vendor is the villain.** Report differences; do not rank vendors "better" or "worse".
- **Run each configuration ≥ 3 times** and report variance. A single run is an anecdote.
- **If a run fails, publish the failure row.** The harness exits 2 for "could not run" precisely so
  this is distinguishable from a fault.

## Scope — start small, ship, then widen

**v1 (target: one weekend):** one model (Qwen3-8B), NVIDIA vs AMD, three rows above, 128 prompts.
Rent by the hour — no purchase, no commitment.

**v2:** add Llama-3.1-8B and a quantised variant (W8A8), widen the prompt set.

**v3 (gated):** add the Ascend column. **Only after counsel clears GP10** — see `RESEARCH.md` and
`STRATEGY.md`. Everything proven in v1/v2 transfers unchanged, because the harness never sees the
silicon.

**This ordering is deliberate.** v1 and v2 are legally clean, self-verifiable, and rentable from
Hong Kong. They earn the credibility that makes the Ascend column worth having, instead of the
Ascend column being the thing that blocks all publication.

## What "monetising" looks like from here

The table is not the product; it is the **distribution**. It generates:

1. **Inbound from the exact people in pain.** The vllm-ascend and vLLM issue threads are full of
   engineers with this problem and no reference number. A citable table is what you post there.
2. **A credential.** "We publish the reference accuracy deltas" is the sentence that makes an
   equivalence report worth paying for. Nobody buys assurance from an unknown party — they buy it
   from whoever wrote the reference.
3. **Two live markets from one artifact.** Western teams migrating NVIDIA→AMD/Gaudi/Trainium can
   transact with a HK company today. Chinese teams migrating to Ascend become reachable once v3 and
   the entity question are resolved.
4. **The paid tier writes itself:** the public table covers open models on rented hardware; the
   paid engagement runs it on **the customer's model, on the customer's hardware**, and produces a
   signed, provenance-bound report. That report is the artifact nobody currently sells — every
   existing tool emits a developer CSV, not something a procurement officer or auditor accepts.

## Acceptance

- `results/` contains raw JSON per configuration, committed.
- `BENCHMARK.md` renders the table with full provenance and a stated tolerance.
- Every number reproducible from the published command line by a third party.
- Variance across ≥3 runs reported.
- A reviewer who dislikes us cannot find a hole in the method.
