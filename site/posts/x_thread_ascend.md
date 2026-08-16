# X/Twitter Thread — Ascend vs CUDA

1/ 🚨 **Big finding from Ruitong:**
Ascend 910B and NVIDIA A40 are NOT token-level equivalent.

We ran Qwen3-8B on both — same model, same seed, same 61 prompts.

Top-1 token agreement: only **15%**.

2/ For context:
• A40 vs RTX 6000 Ada (same CUDA): ~19% of prompts diverged
• A40 vs MI300X (ROCm): ~30-40%
• **A40 vs Ascend 910B: 98% of prompts diverged**

3/ We checked carefully:
✅ Same sampling params (temp=0, seed=1234)
✅ Warm-repeat tested (each prompt 3x)
✅ Full logprobs captured with top-5

This isn't noise. The Ascend CANN stack produces fundamentally different token distributions.

4/ What this means for ML:
If you're doing logit-based alignment, RLHF, speculative decoding, or any technique relying on token-level distributional equivalence — **your results may not transfer between CUDA and Ascend.**

5/ We're open-sourcing everything:
📊 Full analysis: ruitong.io
🐙 Corpora & code: github.com/loopeywho/ruitong

Next up: AMD MI350 vs CUDA. Follow for results.

#Ascend910B #LLM #AI #Benchmarking #Ruitong