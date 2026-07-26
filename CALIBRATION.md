# 瑞通 Ruitong — Equivalence Gate Calibration

**Author:** Claude (Sonnet 5) · 2026-07-27 · **Method:** measured, not reasoned
**Reproduce:** `uv run python -m ruitong.equivalence.faults` fixtures + the table below.
Pure stdlib, CPU, no model download, no hardware, seconds to run.

> **Verdict: the gate as specified in `PLAN.md` is mis-calibrated in both directions.**
> It rejects correct ports and is blind to at least one real class of fault.
> This was found without spending a yuan on Ascend — which is the entire argument for
> `PHASE_4_5_PROPOSAL.md`.

---

## Method

Synthetic logprobs shaped like real LLM output — sharply peaked, steep tail, 512 tokens ×
8 positions, deterministic seed. Two categories compared against the reference:

- **EQUIVALENT** — differences a *correct* port legitimately produces. Chiefly **bfloat16
  rounding**, the closest CPU-only analogue to two hardware kernels disagreeing in the last bits.
- **FAULT** — injected defects mimicking real porting failures (`equivalence/faults.py`).

A gate is only useful if every EQUIVALENT case passes and every FAULT case fails, **with margin.**

## Result 1 — the gate rejects correct ports 🔴

| scenario | cosine | max_abs_diff | top-1 | top-5 | gate |
|---|---|---|---|---|---|
| EQUIVALENT identical | 1.00000 | 0.0000 | 1.000 | 1.000 | PASS |
| **EQUIVALENT bfloat16** | 1.00000 | **0.4929** | 1.000 | 1.000 | **FAIL** ← wrong |
| FAULT swap top-2 | 1.00000 | 0.3478 | 0.000 | 1.000 | FAIL |
| FAULT scale ×1.05 | 1.00000 | 9.0067 | 1.000 | 1.000 | FAIL |
| FAULT truncate vocab | 0.76417 | 150.1347 | 1.000 | 1.000 | FAIL |
| FAULT shift positions | 1.00000 | 0.5038 | 1.000 | 0.917 | FAIL |
| FAULT corrupt 12.5% | 0.99990 | 0.6250 | 1.000 | 0.906 | FAIL |

`PLAN.md` requires `max_abs_diff ≤ 0.05`. **BF16 rounding alone produces 0.4929 — ten times the
threshold.** A perfectly correct BF16 port fails the gate. Shipped as-is, the product would reject
every real port it was ever pointed at.

## Result 2 — `max_abs_diff` measures the wrong part of the distribution 🔴 (root cause)

The largest per-token differences under BF16 occur at vocabulary ranks **366, 509, 427**, whose
logprobs are **−129.5, −179.5, −150.5**. Those correspond to probabilities around `1e-57` — tokens
that will never be sampled in the lifetime of the universe.

Meanwhile the max difference across the **top-10 tokens is 0.0152** — 32× smaller.

BF16 has 7 mantissa bits, so absolute error scales with magnitude: at logprob ≈ −130 the ULP is
≈ 0.5. The metric is dominated by precision noise in numerically irrelevant tail values, and is
almost blind to the region that determines model behaviour.

**Consequence — there is no threshold that works:**

```
bfloat16 rounding (EQUIVALENT) : 0.4929
position shift    (FAULT)      : 0.5038
margin: +2.2%
```

A 2.2% gap between "correct" and "broken". No threshold can reliably separate them.

## Result 3 — `cosine_similarity` cannot detect scaling errors at all 🔴

| scaling applied | cosine |
|---|---|
| ×1.01 | 1.0000000000 |
| ×1.05 | 1.0000000000 |
| ×1.5 | 1.0000000000 |
| ×2.0 | 1.0000000000 |

Cosine similarity is **scale-invariant by definition** — multiplying a vector changes its magnitude,
never its direction. A temperature or softmax-scaling bug, one of the most common porting failures,
is *mathematically invisible* to this metric. Doubling every logprob scores a perfect 1.0.

Cosine also returned 1.00000 for swapped tokens, shifted positions, and 0.99990 for 12.5% corrupted
positions. **It moved for exactly one of five faults.** It contributes almost no signal and should
not be a gate.

## Result 4 — the fix, measured

Restrict `max_abs_diff` to the **top-k tokens of the reference run**, and add a **probability-mass**
check (`Σ exp(logprob)` should be ≈ 1.0; catches unnormalised or rescaled output).

| scenario | top-10 max_abs_diff | prob mass | detected |
|---|---|---|---|
| EQUIVALENT bfloat16 | **0.0152** | 0.999877 | — (correctly passes) |
| FAULT swap top-2 | 0.6472 | 1.000000 | top-k |
| FAULT scale ×1.05 | 0.2341 | 0.904455 | top-k, mass |
| FAULT shift positions | 0.4476 | 1.000000 | top-k |
| FAULT corrupt 12.5% | 5.0000 | 9.472036 | top-k, mass |
| FAULT truncate vocab | 0.0000 | 1.000000 | **neither** |

**Separation improves from 1.02× to 15×** (0.0152 equivalent vs 0.2341 weakest fault). That is a
gate you can actually set a threshold on.

### On the one case still "missed"

Truncating the tail to −30 is detected by neither metric — and on inspection **it is not a real
equivalence failure.** A token at logprob −30 and one at −130 are both never sampled; the model's
behaviour is identical. My fault injector was modelling "wrong vocab handling" with a floor too
generous to matter.

The *genuine* vocab fault — a ported model with a different vocabulary **size** — is already caught
structurally: `_ensure_lists` raises on a length mismatch before any metric runs. So this is a gap
in my test fixture, not in the gate. Recorded rather than quietly dropped.

---

## Recommended gate

| metric | threshold | rationale |
|---|---|---|
| **top-k max_abs_diff** (k=10) | **≤ 0.05** | bf16 measures 0.0152; weakest fault 0.2341 → ~3× headroom above noise, ~5× below the nearest fault |
| **probability mass** | \|Σp − 1\| ≤ 0.01 | catches unnormalised/rescaled output |
| **top-1 agreement** | ≥ 0.99 | keep — the only metric that caught the token swap |
| **top-5 set agreement** | ≥ 0.95 | keep — caught shift and corruption |
| ~~cosine similarity~~ | **drop** | scale-invariant; blind to scaling bugs; moved for 1 of 5 faults |
| ~~full-vocab max_abs_diff~~ | **drop** | dominated by irrelevant tail; 2.2% separation |

**Thresholds above are measured, not assumed.** Every number traces to a run in this document.

## Caveat — stated plainly

These are **synthetic** logprobs with a plausible shape, not output from a real model. The
*structural* findings are robust and do not depend on the fixture:

- cosine's scale-invariance is a mathematical identity, true for any input;
- BF16 absolute error scaling with magnitude is a property of the format;
- therefore tail-dominated `max_abs_diff` is unavoidable on real logprobs too.

The **exact numeric thresholds** should be re-measured against a real model before shipping —
`Qwen2.5-0.5B-Instruct` on CPU, FP32 vs BF16, is sufficient and still needs no accelerator. That is
`PHASE_4_5_PROPOSAL.md` §4.5a and remains the next step.

## Why this matters commercially

Per `DECISIONS.md` D5 the **equivalence report is the product**. A gate that fails correct ports
destroys trust on first contact; a gate blind to a fault class ships a false PASS, which is worse —
it is silent, and the customer acts on it. The sensitivity table above is also the artifact a buyer
needs in order to believe a report at all: *"here is what our gate catches, and the smallest fault
of each kind it still detects."*
