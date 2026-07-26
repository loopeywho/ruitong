# 瑞通 (Ruitong) — Phase 4 Audit (supplementary)

**Reviewer:** Claude (Sonnet 5, interactive session) · **Date:** 2026-07-26 20:25
**Verified against:** working tree, quiescent 3+ min before reading. Every claim below was executed, not inferred.

> Supplements the autonomous audit in `QA_FINDINGS.md`. Separate file so nothing is overwritten.

---

## 🔴 F1 · `top_k_set_agreement` fails 100% of real comparisons — BLOCKER

**This is the most important finding in Phase 4 and the autonomous audit did not catch it.**

`metrics.py:103` compares the top-k **float values**, not the top-k **token indices**:

```python
top_a = set(v for _, v in sorted(enumerate(a_row), key=lambda x: -x[1])[:k])
#           ^ the value                                                    
```

Executed, not reasoned about:

```
A = [-1.0,    -2.0,    -3.0   ]
B = [-1.0001, -2.0001, -3.0001]   # differs by 1e-4 — obviously equivalent

cosine_similarity    = 1.0
top_k_agreement(k=1) = 1.0
top_k_set_agreement  = 0.0        # PLAN.md gate requires >= 0.95
```

**Why this is fatal rather than cosmetic.** Ruitong's entire premise is comparing *different hardware running different kernels*. CUDA and Ascend will **never** emit bit-identical floats — if they did, there'd be no product. So the value sets are disjoint on every real comparison, Jaccard collapses to 0, and the `top-5 set agreement ≥ 95%` gate rejects every correct port. The equivalence harness would declare a perfect port a failure, every time.

PLAN.md's intent was **token** set agreement: *do both backends rank the same 5 tokens in their top 5?* That's index-based and robust to small numeric differences.

**Fix:** compare index sets.

```python
top_a = set(i for i, _ in sorted(enumerate(a_row), key=lambda x: -x[1])[:k])
top_b = set(i for i, _ in sorted(enumerate(b_row), key=lambda x: -x[1])[:k])
```

The docstring ("Compares the sets of top-k *values* (not indices), measuring how much the actual logprob mass overlaps") documents the behaviour, so this was deliberate — but the intent doesn't survive contact with floating point. Comparing "logprob mass overlap" via exact set membership on floats is not a measurement; it's an equality check wearing a Jaccard costume. If mass overlap really is wanted, it needs a tolerance-based comparison, not `set()`.

## 🔴 F2 · Every metric test is vacuous — this is *why* F1 survived

`tests/test_equivalence.py::TestTopKSetAgreement` — all five cases use **exactly equal floats**:

```python
a = [10.0, 9.0, 8.0, 7.0, 6.0]
b = [10.0, 9.0, 8.0, 7.0, 6.0]   # identical
```

No test ever supplies *slightly different* values — which is the only input that occurs in production. The suite proves the implementation agrees with itself, never that it measures what it claims. Same failure mode as Phase 1's `test_invalid_role_rejected`, in new clothes.

**Fix:** every metric needs a *near-miss* case. Add, at minimum:

```python
def test_near_identical_values_still_agree(self) -> None:
    """Different kernels never produce bit-identical floats — the whole point."""
    a = [[-1.0, -2.0, -3.0]]
    b = [[-1.0001, -2.0001, -3.0001]]
    assert top_k_set_agreement(a, b, k=3) == pytest.approx(1.0)
```

That test fails today. It should be written first, then F1 fixed to make it pass.

**Standing rule, restated:** a passing test on identical inputs tells you nothing about a comparison function. Perturb the input or the test is decorative.

## 🟠 F3 · `max_absolute_difference` crashes on empty inner lists — confirmed

```
>>> max_absolute_difference([[]], [[]])
ValueError: max() arg is an empty sequence
```

`metrics.py:78` calls `max()` on an empty generator. `_ensure_lists` guards the *outer* list (`len(a) == 0`) but an inner `[]` passes validation cleanly — `inner_len` becomes 0 and every row matches.

Reachable whenever a backend returns `logprobs: []` for a position, which real servers do.

The autonomous audit flagged this area (its P1.3) but misdiagnosed the cause as "inconsistent validation for subsequent rows" — validation is in fact consistent; the failure is a downstream crash. Right neighbourhood, wrong house.

**Fix:** reject empty inner lists in `_ensure_lists` with a clear message, or return `0.0` for empty rows. Rejecting is better — silently scoring an empty comparison as perfect agreement is how a fake equivalence report gets written.

---

## Process notes on the autonomous audit

**Symbols were real this time — a genuine improvement.** No repeat of the fabricated `ChatRequest.__backend_type`. All eight cited functions exist.

**But every line number is wrong**, systematically:

| Cited | Actual |
|---|---|
| `cli.py:67` `_print_error` | `cli.py:21` |
| `cli.py:75-81` `_make_runner` | `cli.py:55` |
| `metrics.py:31-42` `_ensure_lists` | `metrics.py:12` |
| `metrics.py:67-77` `cosine_similarity` | `metrics.py:45` |
| `metrics.py:95-105` `top_k_agreement` | `metrics.py:83` |
| `runner.py:52-65` `PerPromptResult` | `runner.py:36` |
| `runner.py:180-195` `to_dict` | `runner.py:64` — off by 116 lines |

The auditor knows the code's *shape* but is citing positions it never read. Line numbers in that report should not be trusted for navigation.

**The audit ran on the wrong model.** `QA_FINDINGS.md` header records `anthropic/claude-sonnet-4`, while `_phase4_audit.py` now reads `MODEL = "anthropic/claude-opus-5"` — the script was updated at 20:19, five minutes *after* the 20:14 audit. So that report is sonnet-4 output, which plausibly explains both the line-number drift and missing F1. **Re-run the audit now that the model is corrected**, and treat the existing `QA_FINDINGS.md` as provisional.

**Recommendation:** have `_phase4_audit.py` run `pytest` and `mypy` and paste real output into the prompt, and require the auditor to quote the line it cites. An auditor that cannot see line numbers should not emit them.

---

## Priority order (original — superseded below)

1. **F1** — fix `top_k_set_agreement` to compare indices. Nothing downstream is trustworthy until this is right; it is the gate the product sells.
2. **F2** — add near-miss tests for *every* metric, before fixing F1.
3. **F3** — reject empty inner lists.
4. Re-run the autonomous audit on opus-5.
5. Carryover, still open: `git init`; `main.py` still registers `FakeCuda`/`FakeAscend` (see `QA_PHASE2.md` F1); `main.py` still at 0% coverage.

---

# ──── APPENDED: Claude Opus 5 Autonomous Audit (2026-07-26 19:23 UTC) ────

**Model:** `anthropic/claude-opus-5` via OpenRouter API
**Input tokens:** 18,600 · **Output tokens:** 19,700
**Thinking mode:** handled (fallback to `reasoning` field — `content: null` on thinking-mode responses)

## Verdict: ❌ FAIL

The harness is well-structured and reads cleanly, but it does not currently do what it claims. Four independent defects cause the harness to report `passed: YES` / exit `0` in situations where nothing was actually compared or where everything failed; two more make the top-k metrics semantically meaningless; and the "deterministic" fixture is not deterministic across processes. Because the deliverable's entire value is *trustworthy pass/fail signal*, these are blockers rather than polish.

Test count (172) and coverage (97%) are not evidence of correctness here — several of the assertions in the new tests are vacuously true, and no test asserts the *sign* of a comparison (that a divergent pair fails).

### P1 — Must Fix (deploy blockers)

**1. Exit code is always 0, even when `passed is False`**
- `main()` / `_run_port()` never propagate the verdict to the process exit status. CI is green regardless of outcome.
- **Fix:** `main()` returns `int`; `_run_port` returns 0 if `report.passed` else 1 (suggest 2 for infrastructure errors). Add tests asserting `returncode != 0` for a divergent comparison.

**2. `passed=True` when every prompt errored (silent pass on total failure)**
- `EquivalenceRunner.run` swallows backend exceptions into `warnings`. If all prompts fail, `all_cosines` is empty, every metric is `None`, all thresholds skip, `report.passed` retains its default `True`.
- **Fix:** default `passed=False` and promote to `True` only after at least one metric was computed and all thresholds held. Add invariant: `if not per_prompt or any(r.mode == "error" for r in per_prompt): passed = False`.

**3. `--target cuda` / `--target ascend` compare a backend against itself → tautological PASS**
- `_make_runner` returns `EquivalenceRunner(cuda, cuda)`. Same object invoked twice → cosine = 1.0, diff = 0.0, always `Passed: YES`.
- **Fix:** either (a) single-target mode = baseline capture (no comparison), or (b) compare against a stored golden baseline. Or reject single-target with a clear error and ship only `auto`.

**4. `top_k_set_agreement` compares float *values*, so it is 0.0 for any non-bit-identical pair** (matches F1 above)
- Two backends differing by 1e-7 produce disjoint value sets → Jaccard 0.0, below `TOP5_MIN = 0.95`. This metric can never pass for real hardware.
- **Fix:** compare top-k **token index sets** (`|A∩B| / |A∪B|` over indices). Add a test with values perturbed by 1e-9 asserting agreement ≈ 1.0.

**5. Top-1 agreement is computed over per-position scalars, not over token distributions**
- `Choice.logprobs` is a flat `list[float]` of per-position chosen-token logprobs. `top_k_agreement(logs_a, logs_b, k=1)` wraps that into one row and asks "which *position* has the highest logprob" — ranking positions against each other. Meanwhile `Choice.top_logprobs: list[dict[int, float]]` — the actual per-position token→logprob distribution — is read by nothing.
- **Fix:** compute top-1/top-5 agreement from `top_logprobs` (per position: argmax token id equality; top-5 index-set Jaccard). Assert `len(top_logprobs) == len(logprobs)` on ingest.

**6. Metric computation outside try/except → uncaught crash**
- `_ensure_lists` raises `ValueError` on length mismatch, but the metric calls are outside the per-prompt `try`. Two real backends with different token counts produce an unhandled traceback, no report, no partial results.
- **Fix:** wrap metric computation in its own `try/except (ValueError, ZeroDivisionError)`, record `mode="mismatch"` + warning, and force `passed=False`.

**7. Fake backends are non-deterministic across processes (`hash()` in the seed)**
- `_fake_logprobs(f"{req.model}-cuda-{hash(req.model) % 1000}")`. `hash()` on `str` is salted per interpreter by `PYTHONHASHSEED`, so every CLI invocation produces different logprobs and a different `report.json`.
- **Fix:** remove `hash()`; seed from stable content only (e.g. `f"{req.model}-cuda"` plus a digest of the rendered prompt). Add a subprocess test asserting byte-identical reports.

**8. Mode 2 ("task-level parity") is documented but not implemented**
- When logprobs are missing, the runner sets `mode="task_parity"`, appends a warning, and performs **no comparison at all** — responses are captured but never compared.
- **Fix:** implement it (exact-match rate + edit distance over responses, with its own threshold), or delete the claim from docstrings and hard-fail (`passed=False`) with a clear "logprobs unavailable" warning.

### P2 — Should Fix

- **9.** `-inf` / NaN logprobs silently pass every threshold and emit invalid JSON (`nan`/`Infinity` literals)
- **10.** `max_absolute_difference` aggregates by *mean of row maxima*, and the report averages again across prompts — a single catastrophic position is averaged down
- **11.** Thresholds are not actually configurable — `Thresholds` holds class attributes, constructing `EquivalenceReport(thresholds_used=CustomThresholds())` changes JSON but not the verdict
- **12.** Prompts, `max_tokens`, and `seed` are hardcoded and unsurfaced — no `--prompts-file`, no `--max-tokens`, no `--seed` passed to `ChatRequest`
- **13.** Fake logprobs ignore the prompt — all three prompts yield identical logprobs, and the cuda/ascend divergence is a uniform additive shift that preserves ranking
- **14.** `_fake_logprobs` slice-wrap is still off-by-one; comment claims otherwise
- **15.** `open()` without `encoding="utf-8"` in a China-market tool
- **16.** Several new tests assert nothing — `assert len(report.warnings) >= 0` (always true), no test asserts `passed is False` anywhere, no test asserts exit code != 0
- **17.** Coverage figure for `cli.py` is likely not real — subprocess tests don't record coverage without `COVERAGE_PROCESS_START`
- **18.** `PerPromptResult` hardcodes `cuda_*`/`ascend_*` field names on a generic A/B runner
- **19.** Degenerate-shape handling in `_ensure_lists` is inconsistent — `[[]]` vs `[[]]` passes validation, `cosine_similarity` returns `0.0`
- **20.** CLI-constructed fakes always contain the requested model, so `ModelNotFound` can never fire from the CLI

### P3 — Polish / Future
- `_ensure_lists` launders types through `Any` — mypy validates nothing inside metric functions
- `print(f"=== Ruitong Equivalence Report ===")` — f-string with no placeholders (ruff F541)
- No `--version` / `--quiet` / `--json-only` — `ruitong port … | jq` isn't possible
- Full logprob arrays embedded per prompt in JSON report — add `--include-tensors` (default off)
- `Choice.logprobs: list[float] | None` is commented "OpenAI-compatible" but OpenAI's shape is different
- Backends are awaited strictly sequentially per prompt — deliberate, worth a comment

### What's Good
- Clean layering: `metrics.py` is pure and side-effect free, `runner.py` orchestrates, `cli.py` only does I/O
- Explicit threshold constants and a `thresholds_used` block in the artifact — right instinct
- Per-prompt results retained alongside aggregates
- Graceful zero-vector handling in cosine, and the metric unit tests with known-angle fixtures
- Subprocess CLI tests catch packaging/entry-point breakage

### Root-Cause Observations
1. **`passed` defaults to `True` and is only ever downgraded.** In a verification tool the default must be "not verified". A single change — `passed: bool = False`, promoted only by an explicit `_evaluate()` — eliminates four of the eight blockers.
2. **The fixture's divergence model (uniform additive shift) is orthogonal to what the metrics measure**, and the fixture never varies by prompt. That's why 172 passing tests coexist with a broken top-1 and a broken top-5 metric.

---

# ──── EXECUTION ORDER — PROOF-BASED FIX SEQUENCE ────

**Instruction:** Fix the near-miss tests first, then the metric. This ordering proves the fix — write the failing test, watch it fail, then make the code pass it.

## Step 1: Near-miss tests (F2 / P2 #16)
*Write failing tests first — they prove the bug exists and the fix works.*

- **1a.** Add `test_near_identical_values_still_agree` — top_k_set_agreement with values differing by 1e-4 must return ≈ 1.0
- **1b.** Add `test_top1_from_logprobs_not_position` — top-1 agreement computed from `top_logprobs` token ids, not position scalars
- **1c.** Add `test_passed_is_false_on_divergent` — a pair that diverges must produce `report.passed is False`
- **1d.** Add `test_exit_code_nonzero_on_failure` — `ruitong port` with divergent fakes must exit 1
- **1e.** Add `test_length_mismatch_caught_gracefully` — different-length logprobs produce `mode="mismatch"` not a crash
- **1f.** Remove all vacuous assertions: replace `assert len(report.warnings) >= 0` with concrete assertions

**Proof gate:** `pytest tests/` — Step 1 tests must fail (RED). All previously passing tests must still pass.

## Step 2: Fix the metric (F1 / P1 #4, #5)
*Now make the failing tests pass.*

- **2a.** `top_k_set_agreement` — compare token index sets (`|A∩B| / |A∪B|` over indices), not float values
- **2b.** `top_k_agreement` — compute from `top_logprobs` (per-position argmax token id equality for top-1; index-set Jaccard for top-5)
- **2c.** Default `passed=False` — promote to `True` only after metric computation and threshold checks
- **2d.** Wrap metric computation in `try/except (ValueError, ZeroDivisionError)` — produce `mode="mismatch"` + warning

**Proof gate:** `pytest tests/` — all tests pass (GREEN), including the new Step 1 tests. Coverage ≥ 97%.

## Step 3: Fix the remaining P1s

- **3a.** Exit code propagation — `main()` returns `int`, `_run_port` returns 0/1/2
- **3b.** `--target` single-backend mode — reject with clear error, or implement baseline capture
- **3c.** Fibonacci seed for fakes — replace `hash()` with stable content-based seed
- **3d.** Mode 2 implementation — exact-match rate + edit distance, or delete the claim

## Step 4: P2 cleanup

- **4a.** `-inf`/NaN guard in `_ensure_lists` + `allow_nan=False` in `json.dump`
- **4b.** True max aggregation (not mean of row maxima)
- **4c.** Instance-level `Thresholds` dataclass + `--thresholds` CLI flag
- **4d.** `--prompts-file`, `--max-tokens`, `--seed` CLI flags
- **4e.** Perturbation-mode fakes (identical, noise, rank_flip, truncated, length_mismatch)
- **4f.** `open(..., encoding="utf-8")` in `cli.py`
- **4g.** Subprocess coverage setup or in-process `main(argv=[...])` tests
- **4h.** Rename `cuda_*`/`ascend_*` to `a_*`/`b_*` in `PerPromptResult`

## Step 5: Re-audit

- Run Opus 5 audit with the fixed code — verify CONDITIONAL PASS or better
- Upload the persistent audit script to the repo so the loop is self-contained

---

**Carryover (unchanged):** `git init`; `main.py` still registers `FakeCuda`/`FakeAscend` (see QA_PHASE2.md F1); `main.py` still at 0% coverage.