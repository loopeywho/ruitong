# r/LocalLLaMA Post — Ascend vs CUDA Benchmark

**Title:** We ran Qwen3-8B on Ascend (DashScope) vs A40 (CUDA) — 61 prompts, identical seed. Top-1 token agreement: 15%. Here's what we found.

**Post:**

We've been building [ruitong.io](https://ruitong.io) — an open project that measures how LLM outputs differ across hardware (CUDA, Ascend, AMD).

Today's finding: **Ascend 910B vs NVIDIA A40 produce fundamentally different token-level outputs for the same model.**

## Methodology

- Model: Qwen3-8B
- 61 prompts, temperature=0, seed=1234, top_logprobs=5
- Warm-repeat checked (each prompt run 3x, discard first/cold)
- Full corpora open-sourced

## Results

| Metric | Value |
|--------|-------|
| Top-1 token agreement | **15.02%** |
| Prompts with divergence | **60/61 (98.4%)** |
| Top-5 set overlap | **20.92%** |

For comparison, A40 vs RTX 6000 Ada (same CUDA stack) diverged on ~19% of prompts. A40 vs Ascend? **98% diverge.**

The Ascend CANN stack introduces major numerical differences in the forward pass. The semantic output is similar, but the underlying probability distributions are fundamentally different.

This matters if you're doing logit-based alignment, RLHF, speculative decoding, or any technique that depends on token-level distributions matching across hardware.

📊 Full write-up: [ruitong.io](https://ruitong.io)  
🐙 Code + corpora: [github.com/loopeywho/ruitong](https://github.com/loopeywho/ruitong)

Would love to hear if others have seen similar results on Ascend. Our next target is AMD MI350 vs CUDA.