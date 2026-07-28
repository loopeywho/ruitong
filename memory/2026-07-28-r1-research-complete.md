# Ruitong — R1 Research Complete

**Date:** 2026-07-28
**State:** HEAD `9a8bc4c` (D12 gate landed), 258 tests, mypy clean

## R1 — vllm-ascend logprobs viability

### Deliverables
- `RESEARCH_ASCEND_LOGPROBS.md` — KIMI 3 (252 lines, 20 URL-cited sources)
- `reports/R1-AUDIT.md` — Audit verdict: **CONDITIONAL RECOMMEND RENT**
- `references/mistakes-log.md` — 4 lessons logged

### Key findings
| Question | Answer | Status |
|---|---|---|
| Q1: logprobs populated? | Yes — source code, CI, user reports confirm | ✅ |
| Q2: Real values or sentinels? | **Values real** (not -9999). BUT #7218: top_logprobs token IDs all same, #2934: excess -inf, #8643: logprobs_mode ignored (unmerged) | ⚠️ |
| Q3: Precision issues? | Triton kernel overflow at 151K vocab — test skipped (`@pytest.mark.skip("UB overflow")`). FP32 guard added Jul 2026 | 🔴 |
| Q4: enforce-eager + prefix cache? | Prefix caching works (20x QPS). enforce-eager: no logprobs difference | ✅ |

### Recommendation
**Rent 910B time, but mandate a validation script first.** If selected-token `logprob` only is needed (no `top_logprobs` token identity), Ascend works today.

### Citation weakness
- `platform.py` cited for enforce-eager independence without a quoted snippet — weakest link in an otherwise strong deliverable.
- Audit was run by Qwen (model fallback), not Opus 5 — Finn Loop violation.

## Next
R2 — Cross-tenant read on jobs (P1.5 security, open)