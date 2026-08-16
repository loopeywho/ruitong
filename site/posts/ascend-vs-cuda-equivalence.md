---
title: "Ascend vs CUDA: Qwen3-8B Token-Level Equivalence"
slug: ascend-vs-cuda-equivalence
description: "We ran Qwen3-8B on both Ascend (DashScope) and CUDA (NVIDIA A40) through 61 prompts with identical seeds and sampling parameters. The result: 15% top-1 token agreement. Ascend and CUDA are not equivalent at the token level."
date: "2026-08-16"
author: "Ruitong Project"
tags: [ascend, cuda, qwen, equivalence, benchmarking]
---

# Ascend vs CUDA: Qwen3-8B Token-Level Equivalence

**Headline:** Ascend and NVIDIA GPUs produce substantially different token-level outputs for the same LLM with identical seeds and sampling parameters.

## Methodology

We captured full logprobs from **Qwen3-8B** running on two backends:

| Backend | Hardware | Provider | Corpus |
|---------|----------|----------|--------|
| Reference | NVIDIA A40 (CUDA) | RunPod | `a40_61.json` |
| Candidate | Ascend NPU (CANN) | DashScope/Alibaba Cloud | `ascend_dashscope_61.json` |

**61 prompts** were run on each backend with identical settings:
- Temperature: 0.0, Top-P: 1.0, Seed: 1234
- Max tokens: 64, with logprobs and top-5 logprobs
- Each prompt warmed up (1 cold discard) and checked for warm-repeat reproducibility

## Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **Top-1 Token Agreement** | **15.02%** (367/2443) | ❌ Severely divergent |
| **Prompts with Any Divergence** | **60/61 (98.4%)** | ❌ Nearly every prompt differs |
| **Top-5 Set Overlap** | **20.92%** | ❌ Correct token rarely in top-5 |
| **Mean \|logprob\| Diff** | **0.1625** | ⚠️ Moderate |
| **Warm-Repeat Reproducibility** | 24/61 (39%) | ⚠️ Low (likely due to reasoning/thinking mode) |

### Visual Summary

```
Same model (Qwen3-8B), seed, temperature
         ↓
    ┌──────────────────────┐
    │  A40 (CUDA)           │          Ascend (DashScope)
    │  top-1 token = "The"  │──15%──▶  top-1 token = "Paris"
    └──────────────────────┘
```

In 85% of token positions, the model's single most-likely token differs between Ascend and CUDA.

## What This Means

This is by far the largest cross-silicon divergence we have measured:

- **A40 vs RTX 6000 Ada** (same CUDA, different GPU): ~19% of prompts diverged
- **A40 vs MI300X** (different vendor, ROCm): ~30-40% of prompts diverged
- **A40 vs Ascend 910B** (this run): **98% of prompts diverged**

The Ascend CANN stack introduces significant numerical differences in the forward pass. While the model's *semantic output* (the final generated text) may be similar, the underlying token probability distributions are fundamentally different.

## Raw Data

The full 61-prompt corpora are available in this repository under `corpora/`:

- [`corpora/a40_61.json`](./corpora/a40_61.json)
- [`corpora/ascend_dashscope_61.json`](./corpora/ascend_dashscope_61.json)

## Reproduce

```bash
# Prerequisites: DashScope API key
export DASHSCOPE_API_KEY="sk-..."

# Capture the Ascend corpus
python tools/capture_corpus.py \
  --endpoint "https://ws-{workspace}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" \
  --model "qwen3-8b" \
  --out corpora/ascend_dashscope_61.json

# Compare against A40 baseline
python tools/compare_corpora.py \
  corpora/a40_61.json \
  corpora/ascend_dashscope_61.json
```

---

*Ruitong Project — measuring cross-accelerator LLM equivalence since 2026.*
*[ruitong.io](https://ruitong.io) · [GitHub](https://github.com/loopeywho/ruitong)*