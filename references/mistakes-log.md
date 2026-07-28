# Mistakes Log — Ruitong Bridge

> Qwen Memory Store. Read before every build. Do NOT repeat listed mistakes.
> Logged by KIMI 3 reviews and Opus 5 audits.

## R1 — vllm-ascend logprobs (KIMI 3, 2026-07-28)

**[RULE] Never assume an unverified upstream feature works correctly.** vllm-ascend populates logprobs values (real numbers, not sentinels), but has 3+ open correctness bugs: token IDs duplicated in top_logprobs (#7218), excess -inf on NPU (#2934), logprobs_mode silently ignored (#8643, unmerged). Each must be version-pinned and validation-script-gated before trusting.

**[RULE] A skipped test with `@pytest.mark.skip("UB overflow")` means the kernel is broken at real vocab sizes.** The Triton `_topk_log_softmax_kernel` overflows at Qwen2's 151K vocab. The test has been skipped for 6+ months. An fp32 guard (Jul 2026) partially mitigates but the root cause remains. Do not ship a threshold calibrated against a kernel whose correctness test is disabled.

**[INFO] Prefix caching on Ascend works since ~mid-2025** — confirmed by benchmarks, active PRs, and release notes. The D9 warm-up protocol can be preserved.

**[INFO] No published Ascend-vs-GPU logprobs comparison exists anywhere** — lm_eval accuracy results are the closest proxy. A validation script on rented hardware is the first time anyone has done a direct numerical comparison.

## R2 — Cross-tenant security fix (KIMI 3, 2026-07-28)

**[RULE] LIST and DELETE endpoints must also be owner-scoped, not just GET.** The owner column and GET scoping already existed in the jobs table, but LIST returned all jobs and DELETE had no owner check. Every CRUD endpoint must filter by authenticated principal — not just the read path.

**[INFO] Return 404 (not 403) for cross-tenant access.** 404 confirms the existence of the resource. 403 confirms the existence but denies access. The spec is correct: 404 for cross-tenant reads, to prevent resource enumeration.

**[FIX] Degenerate top_logprobs guard added** — `bdfe1eb` refuses top_logprobs where all token IDs are identical (vllm-ascend bug #7218 workaround). The gate now rejects inputs where logprobs mode would produce structurally wrong data.