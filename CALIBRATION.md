# 瑞通 Ruitong — Equivalence Gate Calibration

**Measured on real hardware, 2026-07-27.** This replaces the synthetic
calibration that preceded it (see git history for the superseded version — its
numbers are retired and must not be quoted). The standing caveat in
`PHASE_4_5_PROPOSAL.md` §4.5a — *"thresholds are calibrated on synthetic
logprobs and must be re-measured against a real model before publishing"* — is
discharged by this document.

## Rig

| | |
|---|---|
| Hardware | 1× NVIDIA A40 (46 GB), RunPod pod `vjqkdwuzls8hnf` |
| Server | vLLM · `--dtype auto --enforce-eager --gpu-memory-utilization 0.95 --max-model-len 8128` |
| Model | `Qwen/Qwen3-8B` |
| Sampling | `temperature=0, top_p=1.0, seed=1234, max_tokens=64, top_logprobs=20`, thinking disabled |
| Corpus | `corpora/cuda_a40_qwen3_8b.json` — 16 prompts, EN + 中文, 333 KB |
| Cost | $0.44/hr, under two hours |

The corpus is committed, so every number below reproduces offline:

```bash
python tools/calibrate_from_corpus.py corpora/cuda_a40_qwen3_8b.json
```

---

## Finding 1 — the server is bit-exact once warm

Same request, same server, repeated: **every top-k logprob identical to the
last bit**, 16/16 prompts. Not "close" — equal.

This is a strong foundation. It means the reference side contributes *zero*
noise, so any difference measured against a second backend is attributable to
that backend rather than shared between them.

## Finding 2 — but a prompt's *first* execution differs

The first execution of a novel prompt returns measurably different logprobs
from every later one. Proven by reading vLLM's own counters across the two
calls, not inferred:

| call | `prefix_cache_queries_total` | `prefix_cache_hits_total` |
|---|---|---|
| first (cold) | +25 | **+0** |
| second (warm) | +25 | **+16** |

Magnitude, over six fresh prompts:

| | worst | warm-vs-warm control |
|---|---|---|
| token-matched Δprob | **8.65e-02** | 0.0 |
| top-k max abs Δlogprob | **1.25** | 0.0 |

Output text was identical every time. The difference lives in the
distribution, not the answer.

**8.65e-02 is 24× larger than the weakest fault the gate must catch.** A gate
loose enough to tolerate it would also pass a demoted argmax — a
catastrophically broken port.

So cache state is not noise to absorb with a wider threshold; it is a variable
to hold constant. `EquivalenceRunner._warm` now sends every prompt to both
backends once and discards the result before measuring (D9). Without it,
whichever backend happened to be colder is reported as broken, and the report
measures the cache instead of the silicon.

## Finding 3 — the D7 primary gate was measuring the wrong thing

`top_k_max_abs_diff` was the D7 primary gate. Real data retired it, for two
independent reasons.

**It ranks by position, so it compares different tokens.** In the tail of the
top-k many tokens are near-tied, so a negligible perturbation swaps their
order. Observed directly at position 30 of one prompt: rank 3 held `'高'` on
one side and `' like'` on the other. The metric reported the gap between two
unrelated tokens as disagreement.

**It works in log space, so it weights irrelevant tokens equally.** For an 8B
model the top-10 spans logprob 0 down to about −29. A 0.5 shift at −26 is a
probability change of ~3e-12 and cannot alter any behaviour, yet it scores the
same as a 0.5 shift at the argmax, which flips the output.

On real data its noise and fault distributions **overlap**:

| | value |
|---|---|
| noise — two *correct* executions (cold vs warm) | up to **1.25** |
| weakest fault — 1% temperature error | **0.406** |

No threshold separates overlapping distributions. The metric is not
re-tunable. Demoted to reported-only.

## Finding 4 — the replacement and its threshold

`token_matched_prob_diff` matches tokens **by identity** (not rank) and
compares **probabilities** (not logprobs). Identity matching removes the
rank-swap artifact; probability space weights each token by how much it can
actually matter.

One subtlety it must handle: **token strings are not unique within a row.**
The wire format carries the decoded string, and many distinct token ids decode
to the same one — a single position in this corpus held **nine ids all
decoding to `""`**. A naive `{token: logprob}` dict keeps only the last, which
made the metric score **0.67 comparing a tensor with itself**. Repeated strings
are paired by order of appearance instead.

| condition | token-matched Δprob | verdict |
|---|---|---|
| identical port | **0.0** exactly | correct |
| bf16 rounding | **1.293e-03** | correct — noise ceiling |
| scale ×1.01 (1% temperature error) | **3.660e-03** | fault — detection limit |
| scale ×1.05 | 1.795e-02 | fault |
| corrupt 1 position in 8 | 9.933e-01 | fault |
| swap top-2 | 1.000 | fault |
| shift positions by 1 | 1.000 | fault |
| demote argmax | 1.000 | fault |

**Threshold: `TOKEN_MATCHED_PROB_DIFF_MAX = 2.2e-03`** — the geometric mean of
the noise ceiling and the weakest fault, sitting 1.7× above noise and 1.7×
below the nearest fault.

That margin is thinner than one would like, and is stated rather than hidden:
**a temperature error below roughly 1% is at or past this gate's detection
limit.** Widening the corpus beyond 16 prompts would tighten the noise estimate
and is the obvious next improvement.

---

## What is still unproven

- Every number here comes from **one GPU, one model, one vendor**. bf16
  rounding is a *proxy* for two hardware kernels disagreeing, not the real
  thing. Until the same corpus is captured from a genuinely different backend,
  the noise ceiling remains an estimate.
- **No Ascend hardware has been touched.** The CUDA↔Ascend claim the product
  exists to make is not yet evidenced.
- Faults are injected, not organic. A real broken port may fail in ways this
  suite does not model.
- 16 prompts is a small corpus for a threshold with a 1.7× margin.
