# R1: Ascend Logprob Research — Does vllm-ascend Return Usable Logprobs?

**Status:** Completed 2026-07-28
**Sources queried:** GitHub (vllm-project/vllm-ascend, vllm-project/vllm), hiascend.com, blog posts, CN vLLM docs
**Caveat:** Web-extract unavailable (DuckDuckGo search-only backend); findings based on search-result descriptions and snippet evidence. Cross-reference after obtaining 910B access.

---

## 1. Verdict: USABLE-BUT-FRAGILE — Do NOT Ship on Ascend Logprobs Alone

There IS a known Ascend-specific logprob defect. Unlike the ROCm precedent (`-9999` sentinels) which makes logprobs entirely unusable, Ascend has a subtler `-inf` overflow bug. This means:
- **Logprobs will pass basic "does it return numbers?" checks.**
- **The numbers will be wrong for a material fraction of tokens.**
- **Equivalence tests against GPU will show inflated divergence, masking real port issues.**

**Do not ship an Ascend equivalence product until this is resolved upstream.**

---

## 2. Direct Evidence: Ascend-Specific Logprob Bug

### Issue #2934 — "vllm-ascend 0.9.1 模型输出logprob包含过多-inf"
- **Model:** Qwen3-8B on vllm-ascend 0.9.1
- **Bug:** Output logprobs contain excessive `-inf` values
- **GPU comparison:** The same model/payload on GPU produces valid logprobs
- **Severity:** Since `EquivalenceRunner` computes cosine similarity and max-abs-difference over logprobs, even a single `-inf` entry per position corrupts the entire comparison vector

**Root cause (inferred, not yet confirmed in vendor docs):** The Ascend NPU's softmax/log_softmax implementation has a numerical stability edge on certain hardware/dtype combinations. CUDA has `torch.log_softmax(x, dim=-1)` which handles extreme values gracefully; the CANN equivalent (`torch_npu.npu_log_softmax`) may not apply the same `max(x) - x` numerical trick in all code paths.

### Issue #19305 (ROCm Precedent) — "Most top_logprobs returned are -9999, but only on amd/rocm"
- **Model:** Google Gemma 3-4B-IT on AMD ROCm
- **Bug:** Returns `-9999` for all tokens except first 1-2
- **Status:** Open upstream vLLM, unfixed
- **Why it matters:** This is the exact failure mode the REDSTAR_PLAN.md warned about. ROCm's issue is a sentinel-value bug (always -9999), which is *easier to detect* than Ascend's -inf bug (which corrupts some tokens silently while others look valid).

**The Ascend -inf bug is WORSE than the ROCm -9999 bug for our use case**, because:
- -9999 is detectable: you can filter it, flag it, refuse to compare.
- -inf is not: valid logprobs range from ~0 (high confidence) to ~-20 (low confidence). An -inf value from a numerical bug looks like a valid low-confidence prediction. `EquivalenceRunner` will compare `-inf vs -12.3` and return a large but not obviously-corrupted divergence value, silently inflating the failure rate.

---

## 3. Active Mitigation Work (Upstream)

### PR #9399 — Propagates logprobs_mode into TopKTopPSampler
- Constructs `AscendTopKTopPSampler(logprobs_mode=logprobs_mode)`
- Enables four modes: `raw_logprobs`, `processed_logprobs`, `raw_logits`, `processed_logits`
- **Status:** Merged into vllm-ascend
- **Significance:** The community is actively working on logprob correctness. The `logprobs_mode` flag gives us a path to request raw logits (pre-softmax) and compute our own logprobs on the client side, bypassing the Ascend softmax entirely.

### PR #2654 — Support prompt logprobs to fix ceval accuracy in V1
- `[0.9.1][PromptLogprobs][V1]`
- **Status:** Merged
- **Significance:** Prompt logprobs are now supported on Ascend. If the -inf bug is confined to the *sampler's* logprobs (generation), prompt logprobs might be valid — partial usability.

### GLM-4.7-FP8-vLLM-Ascend Deployment
- Gitcode-hosted deployment of a 130B FP8 model
- API response shows `"logprobs": null` — suggests logprobs were intentionally disabled in that deployment
- Could be a workaround for the -inf bug, not a limitation

---

## 4. What We Need to Verify (Spike S1 Checklist)

If/when we rent 910B time, the verification protocol:

```
1. Start vllm-ascend with logprobs=true, top_logprobs=5
2. Send 20 diverse prompts (from the R5 corpus)
3. For each output token position, log:
   - token_logprob
   - top_logprobs[0:5]
   - proportion of -inf values
   - proportion of -9999 values
4. Repeat with --logprobs-mode=raw_logits
   - If raw_logits are clean, compute log_softmax client-side
5. Compare against CUDA output for same model+prompt+seed
   - If divergence matches within hardware tolerance -> usable
   - If systematic -inf/-9999 -> blocked
```

**Pass criterion:** Fewer than 2% of token positions have `-inf` or `-9999` logprobs after switching to `raw_logits` mode. Above 2% → logprob-based equivalence cannot ship on Ascend without upstream fix.

---

## 5. Implications for Ruitong Product

### If logprobs pass (clean after raw_logits):
- `POST /v1/port` with `target="ascend"` can use the same `EquivalenceRunner(ascend_backend, cuda_backend)`
- Logprob metrics (cosine_similarity, max_abs_difference, top_k_agreement) are valid
- **Gate keeps its current design.** Pass/fail line: distribution divergence across 60+ prompts.

### If logprobs fail (persistent -inf):
- **No logprob-based product on Ascend.** Full stop.
- Possible fallbacks (in priority order):
  1. **Output-string equivalence only** — compare terminated text, no distribution check. Weaker but still useful.
  2. **Prompt logprobs only** — if the bug is in generation sampling, prompt logprobs may work.
  3. **Hidden-state comparison** — requires CANN-side hook, not supported by vllm-ascend today.
  4. **Custom CANN kernel** — compute log_softmax with numerical guard on Ascend. Months of work.
- If all fallbacks fail, **Ruitong on Ascend is a router (load-balance to CUDA/Ascend backends) without equivalence validation.** The router product is still sellable, but the moat narrows to "operational equivalence" rather than "mathematical equivalence."

---

## 6. Summary Table

| Source | Finding | Severity | Status |
|--------|---------|----------|--------|
| Issue #2934 | vllm-ascend 0.9.1 emits -inf logprobs on Qwen3-8B | **HIGH** — corrupts comparison | Open bug |
| Issue #19305 (ROCm) | ROCm emits -9999 sentinels (precedent) | Reference | Open upstream |
| PR #9399 | logprobs_mode flag for AscendTopKTopPSampler | **LOW** — mitigation path | Merged |
| PR #2654 | Prompt logprobs support for V1 | **MEDIUM** — partial usability | Merged |
| hiascend.com | Llama 3.1-8B deployment tutorial | Info | — |
| Issue #11711 | Qwen3.5-35B prefix caching slowdown on 910B | Low (perf, not correctness) | Open |
| GLM-4.7-FP8 deploy | `"logprobs": null` in API response | Signal (possible workaround) | — |

**Bottom line:** The -inf bug must be verified and mitigated before any Ascend equivalence product ships. The `raw_logits` mode via PR #9399 is our best mitigation path — compute log_softmax client-side avoids the Ascend softmax defect. Spike S1 should target this verification first.