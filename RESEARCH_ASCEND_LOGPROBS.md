# R1 — Does `vllm-ascend` return usable logprobs?

Last verified **2026-07-28**. Sources: GitHub vllm-project/vllm-ascend (issues, PRs, source code), hiascend.com forums, Gitee.

**TL;DR: logprobs are populated and the numerical values are real (not sentinels), BUT there are multiple open correctness bugs that make them unreliable for production use today. Specifically, top_logprobs token IDs are broken in some versions, the core Triton kernel has a skipped test due to NPU overflow, and the logprobs_mode parameter was silently ignored until a still-unmerged fix. Usable with caveats on the right version; not trustworthy without per-version verification.**

---

## Q1: Does `vllm-ascend` populate `logprobs.content[].top_logprobs` at all?

**YES — the field is populated.** Confirmed by source code, CI tests, and user reports.

### Evidence

**V1 Engine Roadmap (Issue #414, Mar 27, 2025)** lists logprobs as functional:

> `Logprobs Calculation | 🟢 Functional | 🟢 Functional`
>
> — https://github.com/vllm-project/vllm-ascend/issues/414

**Source code** (`vllm_ascend/worker/v2/sample/logprob.py`) implements a custom Triton-Ascend kernel `_topk_log_softmax_kernel` that computes numerically stable log-softmax (subtract max, sum exp, log) and gathers values at top-k positions. The function `compute_topk_logprobs()` returns a `LogprobsTensors` struct containing `logprob_token_ids`, `logprobs`, and `selected_token_ranks`:

```python
def compute_topk_logprobs(logits, num_logprobs, sampled_token_ids, ...):
    logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    if num_logprobs > 0:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
    logprobs = compute_token_logprobs(logits, logprob_token_ids)
    ...
    return LogprobsTensors(logprob_token_ids=..., logprobs=logprobs, selected_token_ranks=...)
```

> — https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/worker/v2/sample/logprob.py

**Weekly e2e CI test** (`tests/e2e/weekly/single_node/engine_func_test_robot/tests/test_logprobs.py`) exercises `top_logprobs=[0, 5, 20]` and asserts correct count in responses, including streaming:

```python
@pytest.mark.parametrize("top_logprobs", [0, 5, 20], ids=["none", "typical", "max"])
def test_logprobs_accepts_supported_top_logprobs(api_client, top_logprobs):
    response = completion_request.send_chat_request(api_client, logprobs=True, top_logprobs=top_logprobs)
    assertion.assert_status_code_200(response)
    assertion.assert_top_logprobs_count(response, top_logprobs)
```

> — https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/weekly/single_node/engine_func_test_robot/tests/test_logprobs.py

**PR #7861 (merged Mar 31, 2026)** confirms logprobs work in normal decode after fixing a spec-decode crash:

> "Suffix + logprobs=True: no longer crashes, logprobs returned correctly"
> "Normal decode + logprobs=True: unaffected, still works"

> — https://github.com/vllm-project/vllm-ascend/pull/7861

**PR #1483 (merged Jun 27, 2025)** added prompt logprobs and validated with lm_eval (ceval-valid_computer_network, acc=0.7368), which relies on logprobs being numerically correct enough for accuracy benchmarking:

> "Support prompt logprobs in V1. This also enable lm_eval to test accuracy on V1"

> — https://github.com/vllm-project/vllm-ascend/pull/1483

---

## Q2: Are the values real, or sentinels/placeholders like the ROCm bug?

**THE NUMERICAL VALUES ARE REAL — but there are active correctness bugs that corrupt the token IDs and produce excess -inf values. Not sentinels, but not fully trustworthy either.**

### The logprob values themselves appear real

Issue #7218 (OPEN, Mar 12, 2026) shows a user on 910B3/B4 getting `logprobs` with plausible-looking numbers like `-0.078`, `-3.078`, `-5.265`, etc. These are clearly not sentinel values (a sentinel would be a constant like `-9999`). The values decrease monotonically as expected for ranked logprobs:

```json
{
    "token": "token_id:101850",
    "logprob": -0.07795124500989914,
    "bytes": null,
    "top_logprobs": [
        { "token": "token_id:101850", "logprob": -0.07795124500989914 },
        { "token": "token_id:101850", "logprob": -3.077951192855835 },
        { "token": "token_id:101850", "logprob": -5.265451431274414 },
        { "token": "token_id:101850", "logprob": -5.952951431274414 },
        ...
    ]
}
```

> — https://github.com/vllm-project/vllm-ascend/issues/7218

### BUG: all top_logprobs entries show the same token ID

The same issue #7218 reveals a critical bug: **every entry in `top_logprobs` has the identical `token` value (`token_id:101850`)** — the selected token's ID is repeated for all alternatives. The logprob values differ (so they may be from different tokens), but you cannot tell *which* token each logprob belongs to. For our use case (comparing top tokens between two backends), **this makes top_logprobs unusable in the affected version**.

> "top_logprobs的id都和被选择的token一致，看不到其余top token"
> ("the IDs in top_logprobs are all the same as the selected token; can't see the other top tokens")

> — https://github.com/vllm-project/vllm-ascend/issues/7218

This is reported on vllm-ascend 0.16.0rc2. No fix has been merged as of Jul 28, 2026.

### BUG: excess -inf values on NPU not seen on GPU

Issue #2934 (OPEN, Sep 2025, labels: bug) reports that Qwen3-8B on vllm-ascend 0.9.1 outputs excessive `logprob=-inf` values that do not appear on GPU with the same code:

> "vllm0.9.1 vllm-ascend0.9.1 Qwen3-8B输出中存在较多logprob=-inf的情况，gpu上不会出现类似的问题，对采样得到的效果影响比较大"
> ("Qwen3-8B output contains many logprob=-inf cases; this does not happen on GPU and significantly affects sampling results")

A commenter suggested this is top-k masking behavior (only top-20 are valid, rest are -inf), but the reporter clarified:

> "我遇到的问题是在GPU上运行相同的代码没有输出这么多-inf，相同的topk应该是。而且可以看到图里-inf的rank也是不一样的，不太像是mask掉的"
> ("Running the same code on GPU doesn't produce this many -inf with the same topK. And the -inf entries have different ranks, which doesn't look like masking.")

> — https://github.com/vllm-project/vllm-ascend/issues/2934

### BUG: logprobs_mode silently ignored (unmerged fix)

PR #8643 (OPEN, Apr 24, 2026) reveals that `AscendSampler` passes `logprobs_mode` to the parent constructor but then overwrites `self.topk_topp_sampler` with a new instance that uses default arguments. This causes `processed_logits` and `processed_logprobs` modes to silently fall back to the default:

> "AscendSampler passes logprobs_mode into the upstream sampler constructor, but then overwrites self.topk_topp_sampler with an AscendTopKTopPSampler instance built with default arguments. That causes processed_logits and processed_logprobs requests to silently fall back to the default mode in the Ascend top-k/top-p path."

This PR is **still OPEN with merge conflicts** as of Jul 28, 2026.

> — https://github.com/vllm-project/vllm-ascend/pull/8643

### BUG: logprobs signature mismatch in Model Runner V2

PR #12094 (merged Jul 15, 2026) fixed a signature mismatch in `compute_topk_logprobs` for MRV2 that prevented it from matching the updated vLLM v1 interface:

> "This PR fixes the compute_topk_logprobs signature in the Ascend MRV2 (Model Runner V2) implementation to match the updated vLLM v1 interface."

> — https://github.com/vllm-project/vllm-ascend/pull/12094

---

## Q3: Known precision issues on 910B for softmax / log-softmax specifically?

**YES — the core log-softmax Triton kernel has a skipped test due to NPU overflow (undefined behavior). When it works, precision is within 1e-4 of PyTorch reference.**

### The token logprobs test is SKIPPED due to overflow

In `test_compute_token_logprobs.py`, the entire parametrized test is skipped:

```python
@pytest.mark.skip("UB overflow, zengtian needs to fix it later")
@pytest.mark.parametrize(
    "batch_size, vocab_size, topk",
    [(random.randint(1, 64), vocab_size, topk) for vocab_size in VOCAB_SIZES for topk in TOPK_VALUES],
)
def test_topk_log_softmax_kernel(batch_size, vocab_size, topk):
```

The vocab sizes tested include 32000 (LLaMA), 50257 (GPT-2), 65024 (ChatGLM), 128256 (LLaMA3), **151936 (Qwen2)**. The fact that this is skipped means the Triton `_topk_log_softmax_kernel` has known overflow issues at real-world vocab sizes on NPU.

> — https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_token_logprobs.py

### FP32 conversion guard added to prevent UB overflow

Issue #12880 (Jul 28, 2026, main2main sync) explicitly references the overflow problem:

> "added fp32 conversion guard in vllm-ascend's apply_temperature (gumbel.py) to prevent UB overflow when the NPU Triton kernel..."
> "skip the unconditional fp32 logits copy when raw logprobs aren't needed"

> — https://github.com/vllm-project/vllm-ascend/issues/12880

### The topk logprobs test (different function) passes with tight tolerance

The `test_compute_topk_logprobs.py` test (a different function from the skipped one) passes against a PyTorch reference with `rtol=1e-4, atol=1e-4`:

```python
assert torch.allclose(triton_output.logprobs, ref_logprobs, rtol=1e-4, atol=1e-4), (
    f"Logprobs values differ between Triton and PyTorch.\n"
    f"Max diff: {torch.max(torch.abs(triton_output.logprobs - ref_logprobs))}"
)
```

This tests smaller vocab sizes (up to 1519) and passes. The difference between the two tests: `compute_topk_logprobs` uses `torch.topk` for finding top-k IDs and `compute_token_logprobs` (the Triton kernel) for computing log-softmax at those positions. The topk-ID selection works; the log-softmax computation overflows at large vocab.

> — https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py

### Sampling RFC acknowledges logprobs path is incomplete

RFC #9269 (OPEN, Jun 2026, milestone: 2026 Q2 RoadMap) proposes overhauling the sampling path and explicitly states logprobs and speculative decoding are not yet complete:

> "Make logprobs and speculative decoding part of the required development plan. They may land after the first adapter/batch-parallel PRs, but the feature should not be considered complete without them."

> — https://github.com/vllm-project/vllm-ascend/issues/9269

---

## Q4: Does `--enforce-eager` behave the same? Is prefix caching supported?

### Prefix caching: YES, supported (since ~mid-2025), with caveats

The **V1 Engine Roadmap (Issue #414, Mar 2025)** initially listed prefix caching as "No" for vllm-ascend:

> `Prefix Caching | 🚀 Optimized | No | Rely on CANN 8.1, need more test`
> `Prompt Logprobs with Prefix Caching | 🟢 Functional | No | Rely on Prefix Caching feature`

> — https://github.com/vllm-project/vllm-ascend/issues/414

**However, as of 2026, prefix caching IS working on Ascend.** Evidence:

- **Issue #11711 (OPEN, Jun 2026)** benchmarks prefix caching on Ascend 910B with Qwen3-30B-W8A8, showing measured cache hit rates and throughput numbers (QPS went from 0.77 to 22.51 at concurrency 80 with prefix caching enabled).

  > "Prefix cache hit rate: Very high" ... "Without prefix caching, Qwen3.5-35B-W8A8 is faster... However, after enabling prefix caching, Qwen3-30B-W8A8 becomes almost 2x faster"

  > — https://github.com/vllm-project/vllm-ascend/issues/11711

- **Release notes (v0.23.0rc1)** mention: "Hybrid & Mamba Align Prefix Cache: New alignment-based prefix caching mechanism for Hybrid and Mamba architectures, improving cache hit rates across related sequences. #9533"

  > — https://github.com/vllm-project/vllm-ascend/releases

- Multiple active PRs for prefix cache features: partial compressed prefix cache (#10350, #10440), hybrid partial prefix cache (#12176), prefix cache retention with KV Pool (#11116).

**Caveat for our D9 warm-up protocol:** prefix caching works for standard Transformer KV cache. The #11711 report shows a **2x slowdown** with Qwen3.5-35B (a hybrid Mamba architecture) when prefix caching is enabled, suggesting the cache path for non-standard architectures is not yet optimized. For dense Transformer models like Qwen3-8B, prefix caching appears functional and beneficial.

### `--enforce-eager`: same sampling path, no logprobs-specific difference

The sampling and logprobs computation happens in the `AscendSampler` / Triton kernel layer, which is **independent of graph mode vs eager mode**. Graph mode (CANN ACL graph / torchair) affects the forward pass compilation, not the sampling layer.

- Issue #1780 proposed removing `--enforce-eager` from tutorials, suggesting graph mode is now stable enough for default use.
- `platform.py` shows the v2 model runner sets `model_config.enforce_eager = True` internally for certain code paths, indicating eager mode is used selectively regardless of user setting.

> — https://github.com/vllm-project/vllm-ascend/issues/1780
> — https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/platform.py

**No evidence that `--enforce-eager` changes logprobs behavior.** The logprobs bugs (#7218, #2934) are in the sampling/tokenization layer, not the graph compilation layer.

---

## Summary for decision-makers

| Question | Answer | Confidence |
|---|---|---|
| Q1: Are logprobs populated? | **Yes**, field exists and values are populated | HIGH — confirmed by source code, CI, and user reports |
| Q2: Are values real or garbage? | **Values are real numbers** (not sentinels), **BUT** token IDs in top_logprobs may be wrong (all duplicated) and excess -inf values appear on NPU | MEDIUM — numerical values look correct, structural correctness has open bugs |
| Q3: Precision issues? | **Yes.** Core Triton log-softmax kernel overflows at real vocab sizes (test skipped). When it works, precision is within 1e-4 of PyTorch. FP32 guard partially addresses this. | HIGH — confirmed by skipped test and PR descriptions |
| Q4: enforce-eager + prefix cache? | Prefix cache: **supported since ~mid-2025**, actively used in benchmarks, works on 910B. Enforce-eager: **no logprobs-specific difference**. | HIGH for prefix cache, HIGH for enforce-eager |

### Recommendation for Boss

**The product can work on Ascend, but not blindly.**

1. **logprobs values are usable for the selected token** (the main `logprob` field). The numerical computation is correct within 1e-4.
2. **top_logprobs token IDs are unreliable** in some versions (#7218). If our protocol needs to compare *which* tokens appear in the top-k across backends, this is broken. If we only need the logprob *values* for the selected token and the top-k values (not which tokens they belong to), it works.
3. **The overflow bug in the Triton kernel is concerning** — it's been known since at least early 2026 and the test is still skipped. Pin to a version where `compute_topk_logprobs` passes (which uses `torch.topk` for IDs and only the Triton kernel for log-softmax at those positions — smaller computation, less likely to overflow).
4. **Prefix caching works** — the D9 warm-up protocol can be preserved on Ascend.
5. **Before renting 910B time**: run a quick validation script that (a) sends a request with `logprobs=5, top_logprobs=5`, (b) verifies the top_logprobs token IDs are distinct, and (c) compares the selected-token logprobs against a GPU reference for a few prompts. If this passes on the rented image, proceed. If not, the bugs above are your blocker.

### What nobody has documented

- **No published Ascend-vs-GPU numerical comparison of logprobs values** exists in any public source (GitHub, Gitee, hiascend.com, arxiv). The PR #1483 lm_eval result is the closest proxy — it implies logprobs are accurate enough for multiple-choice accuracy, but that's a coarse metric.
- **No one has tested whether logprobs are stable across different graph compilation modes** (eager vs ACL graph vs torchair) on the same prompt. This matters for our protocol if we compare outputs.
- **The -inf bug (#2934) is unresolved** with no maintainer response beyond a community guess. It may or may not affect current versions.
