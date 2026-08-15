# 瑞通 Ruitong — Cross-Accelerator Equivalence Reference

**Auspicious flow between hardware ecosystems.**  
Domain: [ruitong.io](https://ruitong.io)  
Chinese: 瑞通 (Ruitong) — 瑞 (propitious omen) + 通 (flow)

---

## The Problem

The same LLM, the same weights, the same prompt — different answers on different hardware.

When you move inference from NVIDIA CUDA to Huawei Ascend, AMD MI300X, or even change precision (fp16 → bf16) on the same GPU, the output distribution shifts. The industry has good benchmarks for throughput and latency, but **nobody publishes a reference number for accuracy equivalence**.

Huawei's `msprobe` tells you operator-level tensor diffs. `vllm-ascend`'s CI accepts 5% drift against a hardcoded YAML value with no GPU baseline. CAICT's national adaptation programme scores five dimensions — accuracy equivalence is not one of them. MLCommons publishes 99%/99.9%-of-FP32 for a handful of benchmark models.

**The table below does not exist anywhere. Publishing it first defines the category.**

---

## What Ruitong Measures

A dated, versioned, reproducible accuracy delta table:

| Model | Reference | Candidate | metric | value | verdict |
|-------|-----------|-----------|--------|-------|---------|
| Qwen3-8B | A100 fp16 | A100 bf16 | top-k max Δlogprob | tbd | tbd |
| Qwen3-8B | A100 bf16 | Ascend 910B bf16 | top-1 agree | tbd | tbd |
| Llama-3.1-8B | A100 bf16 | MI300X bf16 | prob mass | tbd | tbd |

### The three rows that matter most:

1. **Same hardware, different precision** (fp16 vs bf16) — establishes the *noise floor*. Everything else is read against it.
2. **Different vendor, same precision** (NVIDIA vs Ascend/AMD) — the actual question a migrating customer has.
3. **Same hardware, different batch size** — published research shows bf16 accuracy varying up to 9% from GPU count and batch size alone. Omitting this row would make vendor deltas look damning when they aren't.

### Key metric: behavioral equivalence gates

Beyond raw logprob deltas, we run **calibrated equivalence gates** — end-to-end tests that check whether the candidate output is *functionally equivalent* to the reference (same classification, same choice, same numeric output within tolerance). Our finding: even when per-token metrics show drift, calibrated gates often pass. This has implications for how inference benchmarks should be designed.

---

## Methodology

- **Models**: Qwen3-8B (primary), Llama-3.1-8B, DeepSeek variants
- **Hardware**: NVIDIA A100 80GB (reference), Ascend 910B 64GB, Ascend 910C 128GB, AMD MI300X
- **Precisions**: fp16, bf16, W8A8
- **Metrics**: top-k max Δlogprob, prob mass shift, top-1 agreement, top-5 set agreement, calibrated equivalence gates
- **Tooling**: Open-source Python pipeline, reproducible via `pip install ruitong` (planned)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Benchmark spec | ✅ Complete | Published in repo |
| Calibration harness | ✅ Complete | Equivalence gate framework |
| A100 fp16/bf16 baseline | ✅ Collected | 18/61 prompts showed text differences |
| Ascend data collection | 🔄 In progress | Requires Ascend hardware rental |
| Public accuracy table | 📝 Draft | First version pending Ascend runs |
| ruitong.io | ✅ Live | Static site on Vercel |
| CNIPA trademark | ✅ Clear | Classes 9 & 42 |

### Key finding so far

On A100 across fp16 vs bf16 (same hardware, same vendor, different precision): **18 out of 61 prompts produced different output text.** All calibrated equivalence gates still passed. This establishes the noise floor — the baseline against which cross-vendor deltas must be evaluated.

---

## Why This Matters

Every company migrating inference from CUDA to Ascend (or any accelerator) needs to answer one question: *"Does the model still work correctly?"* Without a published reference, every team independently discovers the same drift — and independently decides what tolerance is acceptable. Ruitong publishes the reference so the industry has a single number to cite, argue with, and improve against.

---

## Repository

Source code and data: [github.com/loopeywho/ruitong](https://github.com/loopeywho/ruitong)  
Project site: [ruitong.io](https://ruitong.io)  
License: Apache 2.0

---

*瑞通 — auspicious flow between ecosystems.*