# Ruitong Bridge — Phase 4 Audit (Claude Opus 5)

_Audited: 2026-07-26 19:23 UTC | Model: anthropic/claude-opus-5_

# Ruitong Bridge — Phase 4 Audit (Equivalence Harness CLI)

**Status: ❌ FAIL**

The harness is well-structured and reads cleanly, but it does not currently do what it claims. Four independent defects cause the harness to report `passed: YES` / exit `0` in situations where nothing was actually compared or where everything failed; two more make the top-k metrics semantically meaningless; and the "deterministic" fixture is not deterministic across processes. Because the deliverable's entire value is *trustworthy pass/fail signal*, these are blockers rather than polish.

Test count (172) and coverage (97%) are not evidence of correctness here — several of the assertions in the new tests are vacuously true, and no test asserts the *sign* of a comparison (that a divergent pair fails).

---

## P1 — Must Fix (deploy blockers)

**1. Exit code is always 0, even when `passed is False`**
- Finding: `main()` / `_run_port()` never propagate the verdict to the process exit status. A CI job running `ruitong port model --target ascend` is green regardless of outcome. This defeats the primary use of an equivalence harness.
- File: `src/ruitong/cli.py` (`main`, `_run_port` — end of function)
- Fix: `main()` returns `int`; `_run_port` returns `0` if `report.passed` else `1` (suggest `2` for infrastructure/error-mode reports so CI can distinguish "diverged" from "couldn't run"). Update `pyproject` entry point to `sys.exit(main())`. Add tests asserting `returncode != 0` for a divergent comparison.
- Severity: HIGH

**2. `passed=True` when every prompt errored (silent pass on total failure)**
- Finding: in `EquivalenceRunner.run`, backend exceptions are swallowed into `warnings` and a `mode="error"` result. If *all* prompts fail, `all_cosines` is empty, every metric is `None`, all four threshold checks are skipped, and `report.passed` retains its dataclass default `True`. Report says `mode: task_parity`, `Passed: YES`. `test_model_not_found` covers this path and asserts `len(report.warnings) >= 0` — which is always true — so the defect is tested *in* rather than tested out.
- File: `src/ruitong/equivalence/runner.py` (`run`, error handler + threshold block); `tests/test_equivalence.py::test_model_not_found`
- Fix: default `passed=False` and set it `True` only after at least one metric was computed and all applicable thresholds held. Add an explicit invariant: `if not per_prompt or any(r.mode == "error" for r in per_prompt): passed = False`. Add a `status` field (`ok` / `diverged` / `incomplete`). Rewrite the test to assert `report.passed is False`, `report.warnings != []`, and `per_prompt_results[0].mode == "error"`.
- Severity: HIGH

**3. `--target cuda` / `--target ascend` compare a backend against itself → tautological PASS**
- Finding: `_make_runner` returns `EquivalenceRunner(cuda, cuda)`. The same object is invoked twice per prompt, so cosine is exactly 1.0, diff 0.0, and the report always says `Passed: YES`. `test_target_cuda_produces_report` asserts `data["passed"] is True`, locking the tautology into the test suite. An operator running `ruitong port Qwen3-8B --target ascend` will reasonably read the output as "Ascend matches CUDA".
- File: `src/ruitong/cli.py` (`_make_runner`); `tests/test_cli.py::test_target_cuda_produces_report`
- Fix: decide and implement one semantic: (a) single-target mode = *baseline capture* — run one backend, emit logprobs/responses, no comparison and no `passed` field at all; or (b) single-target mode = compare that backend against a stored golden baseline file (`--baseline report.json`). Either way, never emit `passed: true` from a self-comparison. If neither is in scope for Phase 4, reject `--target cuda|ascend` with a clear error and ship only `auto`.
- Severity: HIGH

**4. `top_k_set_agreement` compares float *values*, so it is 0.0 for any non-bit-identical pair**
- Finding: the function Jaccards the *set of top-k values*. Two backends differing by 1e-7 produce completely disjoint value sets → Jaccard 0.0, versus `TOP5_MIN = 0.95`. This metric can therefore never pass for real hardware, and in the current auto-mode fixture it is the *only* failing metric — i.e. 100% of the harness's "detected divergence" is an artifact of a broken metric. The docstring rationalises this as "measuring how much the actual logprob mass overlaps", which set-intersection of floats does not measure. Set-ification also silently dedups repeated values before/after truncation.
- File: `src/ruitong/equivalence/metrics.py` (`top_k_set_agreement`)
- Fix: compare top-k **token index sets** (`|A∩B| / |A∪B|` over indices), which is the standard top-k agreement metric. If value-mass overlap is genuinely wanted, implement it as tolerance-based (sum of `min(p_a, p_b)` over aligned tokens, or L1 distance over the top-k probability mass) — never as exact float set membership. Add a test with values perturbed by 1e-9 asserting agreement ≈ 1.0.
- Severity: HIGH

**5. Top-1 agreement is computed over per-position scalars, not over token distributions; `top_logprobs` is never used**
- Finding: `Choice.logprobs` is a flat `list[float]` of per-position chosen-token logprobs (3 entries in the fakes). `top_k_agreement(logs_a, logs_b, k=1)` wraps that into one row and asks "which *position* has the highest logprob" — it ranks positions against each other, not candidate tokens at a position. That is not top-1 token agreement and carries no equivalence meaning. Meanwhile `Choice.top_logprobs: list[dict[int, float]]` — the actual per-position token→logprob distribution the fakes carefully generate — is read by nothing in the runner.
- File: `src/ruitong/equivalence/runner.py` (`run`, Mode 1 block); `src/ruitong/equivalence/metrics.py` (`top_k_agreement`)
- Fix: compute top-1/top-5 agreement from `top_logprobs` (per position: argmax token id equality; top-5 index-set Jaccard), and compute cosine/max-abs-diff over the aligned per-position `logprobs` vector. Assert `len(top_logprobs) == len(logprobs)` on ingest. If `top_logprobs` is absent, report top-k as `None` rather than fabricating a number from the wrong tensor.
- Severity: HIGH

**6. Metric computation sits outside the try/except → uncaught `ValueError` crashes the run**
- Finding: `_ensure_lists` raises `ValueError` on length mismatch, but the calls to `cosine_similarity` et al. are outside the per-prompt `try`. Two real backends returning different token counts (different tokenizer, different stop handling, one truncated) — the expected case — produce an unhandled traceback out of `runner.run()` through `asyncio.run()` to the user, with no report and no partial results.
- File: `src/ruitong/equivalence/runner.py` (`run`, Mode 1 block, lines after `logs_a`/`logs_b` extraction)
- Fix: wrap metric computation in its own `try/except (ValueError, ZeroDivisionError)`, record `mode="mismatch"` + warning, and force `passed=False`. Add a test with two fakes returning different-length logprobs.
- Severity: HIGH

**7. Fake backends are non-deterministic across processes (`hash()` in the seed)**
- Finding: `_fake_logprobs(f"{req.model}-cuda-{hash(req.model) % 1000}")`. `hash()` on `str` is salted per interpreter by `PYTHONHASHSEED`, so every CLI invocation produces different logprob values and a different `report.json`. The module docstring and the Phase 4 charter both promise *deterministic* comparison; the fixture that underpins every test violates it. In-process tests don't catch it because both fakes share the same salt.
- File: `src/ruitong/backends/fake.py` (`FakeCuda.chat`, `FakeAscend.chat`)
- Fix: remove `hash()`; seed from stable content only (e.g. `f"{req.model}-cuda"` plus a digest of the rendered prompt). Add a regression test that runs the CLI twice via subprocess and asserts the two `report.json` files' `metrics` and `per_prompt_results` are byte-identical.
- Severity: HIGH

**8. Mode 2 ("task-level parity") is documented but not implemented**
- Finding: when logprobs are missing, the runner sets `mode="task_parity"`, appends a warning, and performs **no comparison at all** — `cuda_response` and `ascend_response` are captured but never compared, no metric is produced, no threshold applied. Combined with #2, a run where neither backend returns logprobs reports `Passed: YES`. `EquivalenceRunner.run`'s docstring explicitly claims "Falls back to Mode 2 (task-level parity)".
- File: `src/ruitong/equivalence/runner.py` (`run`, `else` branch)
- Fix: implement it (exact-match rate and/or normalised edit distance over responses, with its own `TASK_PARITY_MIN` threshold and aggregate metric key), or delete the claim from the docstrings and hard-fail (`passed=False`) with a clear "logprobs unavailable — cannot establish equivalence" warning. Add a fake variant returning `logprobs=None` and test both the metric and the verdict.
- Severity: HIGH

---

## P2 — Should Fix

**9. `-inf` / NaN logprobs silently pass every threshold and emit invalid JSON**
- Finding: real logprobs contain `-inf`. Cosine and max-abs-diff then yield `nan`; `nan < 0.99` is `False`, so all four threshold checks pass and `passed` stays `True`. `json.dump` writes bare `NaN`/`Infinity` literals, which are invalid JSON and rejected by strict parsers (Go, Java, `json.loads(..., parse_constant=...)`-strict consumers).
- File: `src/ruitong/equivalence/metrics.py`; `src/ruitong/equivalence/runner.py` (threshold block); `src/ruitong/cli.py` (`json.dump`)
- Fix: validate inputs with `math.isfinite` in `_ensure_lists` (either reject, or clamp `-inf` to a documented floor like `-100.0` and record a warning). Guard threshold checks with `math.isnan(...) → passed = False`. Pass `allow_nan=False` to `json.dump` and handle the resulting `ValueError`.
- Severity: MEDIUM

**10. `max_absolute_difference` aggregates by *mean of row maxima*, and the report averages again across prompts**
- Finding: a single catastrophic position (diff 10.0) among 100 clean ones is averaged down to 0.1 and passes `MAX_ABS_DIFF_MAX = 0.05`… or hides entirely. For a gate metric named "max", mean aggregation is the wrong operator and the name actively misleads.
- File: `src/ruitong/equivalence/metrics.py` (`max_absolute_difference`); `src/ruitong/equivalence/runner.py` (aggregate `metrics` dict)
- Fix: return the true max for batched input; aggregate across prompts with `max`, not `mean`. If a mean is wanted too, expose it as a separate `mean_absolute_difference` key. Aggregate cosine/top-k with both mean **and** min so worst-case is visible.
- Severity: MEDIUM

**11. Thresholds are not actually configurable, and `thresholds_used` is decorative**
- Finding: `Thresholds` holds class attributes; the runner reads `Thresholds.COSINE_MIN` directly rather than `self`/`report.thresholds_used`. Constructing `EquivalenceReport(thresholds_used=CustomThresholds())` changes the JSON output but not the verdict — a trap. No CLI flags to override. The provenance of 0.99/0.05/0.99/0.95 is undocumented.
- File: `src/ruitong/equivalence/runner.py` (`Thresholds`, `run` threshold block)
- Fix: make `Thresholds` a `@dataclass(frozen=True)` with instance fields, accept it in `EquivalenceRunner.__init__`, and evaluate against the instance. Add `--cosine-min/--max-abs-diff/--top1-min/--top5-min` (or `--thresholds thresholds.json`). Document where the numbers come from.
- Severity: MEDIUM

**12. Prompts, `max_tokens`, and `seed` are hardcoded and unsurfaced**
- Finding: `_default_prompts` hardcodes three English prompts; `max_tokens=1` is buried in the runner; `seed` is never set on `ChatRequest` (so a real backend is free to be nondeterministic); `temperature` relies on the schema default. An equivalence harness whose comparison set cannot be changed is not usable for the real porting workflow, and `max_tokens=1` makes the (unimplemented) task-parity mode compare single tokens.
- File: `src/ruitong/cli.py` (`_default_prompts`, `_run_port`); `src/ruitong/equivalence/runner.py` (`run`)
- Fix: add `--prompts-file` (one prompt per line or JSONL) and `--max-tokens`; pass an explicit `--seed` (default 0) into `ChatRequest`; record prompts/max_tokens/seed/temperature in the report so a run is reproducible from its own artifact.
- Severity: MEDIUM

**13. Fake logprobs ignore the prompt, and the injected divergence is the least-detectable perturbation**
- Finding: `_fake_logprobs` is seeded only by the model name, so all three prompts yield identical logprobs — the per-prompt table is three copies of one row, and the harness cannot demonstrate detection of prompt-dependent divergence. Worse, the cuda/ascend divergence is a *uniform additive shift* (`-0.001` vs `-0.012`), which preserves ranking exactly and barely perturbs cosine. The fixture therefore cannot exercise the failure modes that matter (rank flips, tail truncation, `-inf` mismatch, length mismatch), which is why every P1 above survived 172 tests.
- File: `src/ruitong/backends/fake.py` (`_fake_logprobs`, `chat`, `_CUDA_OFFSET`/`_ASCEND_OFFSET`)
- Fix: seed from `(model, prompt)`. Add configurable perturbation modes to the fakes — `identical`, `noise(eps)`, `rank_flip(n)`, `truncated`, `neg_inf_mismatch`, `length_mismatch` — and add tests asserting the expected verdict for each. Golden-value test for the auto comparison (exact metric values + `passed is False`).
- Severity: MEDIUM

**14. `_fake_logprobs` slice-wrap is still off-by-one; comment claims otherwise**
- Finding: the guard is `if idx >= raw_len` *after* incrementing, so at `idx == 58` a perfectly valid 6-char pair is discarded and the counter re-reads from offset 0, colliding with position 0's bytes (duplicate token ids collapse in the `top5` dict, silently yielding <5 entries). The 12-char `chunk` read has **no** wrap handling at all, so `count > 3` can produce a short or empty slice and `int("", 16)` raises. The comment "Single linear counter avoids slice-wrap bugs" is false.
- File: `src/ruitong/backends/fake.py` (`_fake_logprobs`)
- Fix: replace the ad-hoc hex-window arithmetic with a seeded `random.Random(seed)` (deterministic, unbounded, no wrap logic), or use `hashlib.shake_256(seed).digest(n_bytes_needed)` sized from `count`. Assert `len(top5) == 5`. Add a test at `count=1, 3, 8, 32`.
- Severity: MEDIUM

**15. `open()` without `encoding="utf-8"` in a China-market tool**
- Finding: `open(args.output, "w")` uses the platform default encoding — `cp936/GBK` on Chinese Windows. `ensure_ascii=True` masks it today, but any future `ensure_ascii=False` or any prompt/response echoed to stdout in Chinese will raise `UnicodeEncodeError` on the primary target platform.
- File: `src/ruitong/cli.py` (`_run_port`, report write)
- Fix: `open(args.output, "w", encoding="utf-8")`; consider `ensure_ascii=False` for readable Chinese in reports and reconfigure stdout/stderr to UTF-8 at entry.
- Severity: MEDIUM

**16. Several new tests assert nothing**
- Finding: `assert len(report.warnings) >= 0` (always true); `assert "cosine_sim" in pp or "cuda_logprobs" in pp` (both keys always present); `test_auto_compare` has its actual expectation as a comment ("likely fail thresholds"); no test asserts `passed is False` anywhere; no test asserts an exit code other than 0; `--target` invalid value, unwritable `--output` (the `OSError` branch), and unknown-model-via-CLI are untested.
- File: `tests/test_equivalence.py` (`test_model_not_found`, `test_auto_compare`); `tests/test_cli.py` (`test_per_prompt_results_in_report`, `test_warnings_list_present`)
- Fix: replace with concrete assertions; add negative-path tests: `--target bogus` → rc 2, `--output /nonexistent/dir/x.json` → rc 1 + "Cannot write report" on stderr, divergent pair → rc 1.
- Severity: MEDIUM

**17. Coverage figure for `cli.py` is likely not real**
- Finding: all CLI tests run the code in a child process (`subprocess.run([sys.executable, "-m", "ruitong.cli"])`). Unless `coverage` is configured with `--parallel-mode` + `COVERAGE_PROCESS_START` and a `sitecustomize` hook, none of those lines are recorded — so the reported 97% either excludes `cli.py` or reflects import-only coverage, and the "97% at equivalence/" claim needs re-verification for the branches flagged in #2/#6/#8.
- File: test harness / `pyproject.toml` coverage config
- Fix: enable subprocess coverage, or add in-process tests that call `main(argv=[...])` directly (it already accepts `argv`) and assert on `capsys` + `SystemExit.code`. Keep one subprocess smoke test for the console-script wiring.
- Severity: MEDIUM

**18. `PerPromptResult` hardcodes `cuda_*` / `ascend_*` field names on a generic A/B runner**
- Finding: the runner is deliberately generic (`backend_a`/`backend_b`, with `backend_a_name` properties that are then never used), but the result and JSON schema say `cuda_logprobs`/`ascend_logprobs`. In `--target ascend` mode, `cuda_logprobs` contains Ascend data. Report consumers will be misled.
- File: `src/ruitong/equivalence/runner.py` (`PerPromptResult`, `to_dict`)
- Fix: rename to `a_*`/`b_*` (or `left_*`/`right_*`) and emit `backend_a: "cuda"`, `backend_b: "ascend"` once at report level, using the existing name properties.
- Severity: MEDIUM

**19. Degenerate-shape handling in `_ensure_lists` is inconsistent**
- Finding: `[[]]` vs `[[]]` passes validation (`inner_len == 0`); `cosine_similarity` then silently returns `0.0` (indistinguishable from "orthogonal"), while `max_absolute_difference` raises a bare `max() arg is an empty sequence` — an internal error message leaking to the user. `isinstance(a[0], (int, float))` also accepts `bool`. Mixed nesting (`[1.0, 2.0]` vs `[[1.0],[2.0]]`) is only caught incidentally by the outer length check.
- File: `src/ruitong/equivalence/metrics.py` (`_ensure_lists`)
- Fix: reject `inner_len == 0` with `ValueError("Inner vectors must not be empty")`; validate that all elements of every row are `int|float` and finite; exclude `bool`.
- Severity: MEDIUM

**20. CLI-constructed fakes always contain the requested model**
- Finding: `FakeCuda(model_ids=[model])` guarantees `ModelNotFound` can never fire from the CLI, so the CLI's most likely real-world error (model not served by one side) is unreachable and unexercised. It also silently discards the fakes' default catalogues.
- File: `src/ruitong/cli.py` (`_make_runner`)
- Fix: use the fakes' default model lists (or a `--fake-models` flag), let `ModelNotFound` surface, and add a CLI test for the not-found path with a nonzero exit code and an actionable message listing available models.
- Severity: MEDIUM

---

## P3 — Polish / Future

- **`_ensure_lists` launders types through `Any`**, so mypy validates nothing inside the metric functions despite the precise public signatures. Consider overloads or a narrow `TypeGuard`/normalisation helper returning a distinct `Matrix = list[list[float]]` type. *(`metrics.py`)* — LOW
- **`print(f"=== Ruitong Equivalence Report ===")`** — f-string with no placeholders (ruff F541). *(`cli.py`, `_run_port`)* — LOW
- **`print(f"Prompt: {report.total_prompts}")`** — singular label on a count reads like it's about to print the prompt text. Use `Prompts:`. Also align column widths with the metric block. *(`cli.py`)* — LOW
- **`report.mode` is `"logprob"` if *any* prompt produced cosines**, so a run where 1/50 prompts had logprobs is labelled `logprob`. Add `"mixed"`, or report per-mode counts. *(`runner.py`)* — LOW
- **`main()` silently no-ops on an unrecognised command** (`if args.command == "port":` with no `else`). Add `else: _print_error(...)` so future subcommands can't be forgotten into a 0-exit no-op. *(`cli.py`)* — LOW
- **No `--version` / `--quiet` / `--json-only`**; the human summary and the "Report written to …" line both go to stdout, so `ruitong port … | jq` isn't possible. Consider `--format json` writing to stdout. *(`cli.py`)* — LOW
- **Full logprob arrays are embedded per prompt in the JSON report.** Fine for 3 floats; at real vocab/sequence sizes this is megabytes per prompt. Add `--include-tensors` (default off) or write tensors to a sidecar `.npz`. *(`runner.py::to_dict`)* — LOW
- **`tests/test_cli.py` docstring claims it "tests the actual CLI entry point configured in pyproject.toml"** but invokes `python -m ruitong.cli`, which exercises the `__main__` guard, not the console script. Either add one test that shells out to `ruitong` (installed) or fix the docstring. *(`tests/test_cli.py`)* — LOW
- **`Choice.logprobs: list[float] | None` is commented "OpenAI-compatible"** but OpenAI's shape is an object with a `content` array of token/logprob/top_logprobs entries. Reword the comment as a documented Ruitong simplification, or align the schema before a real backend lands. Also note `dict[int, float]` keys become strings after a JSON round-trip. *(`schemas.py`)* — LOW
- **`top_k_agreement`'s docstring says "top-k token ranks agree"** but it compares unordered index sets, ignoring order within k. Tie-breaking is index-order (stable sort) — deterministic, worth stating explicitly. *(`metrics.py`)* — LOW
- **Backends are awaited strictly sequentially** per prompt. Deliberate and correct for determinism/hardware contention — worth a comment so a future contributor doesn't "optimise" it with `asyncio.gather`. *(`runner.py::run`)* — LOW
- **`backend_a_name`/`backend_b_name` properties are dead code** (only exercised by a test). Wire them into the report per #18 or remove. *(`runner.py`)* — LOW

---

## Notes

**What's good and should be preserved:**
- Clean layering: `metrics.py` is pure and side-effect free, `runner.py` orchestrates, `cli.py` only does I/O. That separation is why every finding above is a small, local fix rather than a rewrite.
- Explicit threshold constants and a `thresholds_used` block in the artifact — right instinct for auditability (just needs to be load-bearing, #11).
- Per-prompt results retained alongside aggregates — essential for debugging real divergence.
- Graceful zero-vector handling in cosine, and the metric unit tests with known-angle fixtures (60°, orthogonal, opposite) are genuinely good.
- Subprocess CLI tests catch packaging/entry-point breakage that in-process tests miss; keep one, but move assertion-heavy cases in-process (#17).

**Root-cause observation.** The recurring pattern across P1s #2, #3, #8 and P2 #16 is that **`passed` defaults to `True` and is only ever downgraded.** In a verification tool the default must be "not verified". A single change — `passed: bool = False`, promoted only by an explicit `_evaluate()` that requires evidence — plus a `status` enum distinguishing `ok` / `diverged` / `incomplete` / `error`, eliminates four of the eight blockers and makes the remaining ones visible in tests.

**Second root cause.** The fixture's divergence model (uniform additive shift) is orthogonal to what the metrics measure, and the fixture never varies by prompt. That's why 172 passing tests coexist with a top-1 metric that ranks positions instead of tokens and a top-5 metric that can never pass. Perturbation-mode fakes (#13) with golden expected verdicts per mode is the highest-leverage test investment for Phase 5.

**Questions for the team:**
1. What is `--target <single>` *meant* to do — baseline capture, golden-file regression, or smoke test? The current self-comparison suggests the semantics were never settled. This decision blocks #3.
2. Where do 0.99 / 0.05 / 0.99 / 0.95 come from? Vendor guidance, empirical fp16-vs-bf16 measurement, or placeholder? These numbers will be quoted in customer-facing porting reports, so provenance needs to be in the docstring.
3. Tokenizer parity is unaddressed: if CUDA and Ascend tokenize differently, positional logprob alignment is meaningless before any metric runs. Is a tokenizer-hash equality precondition in scope for Phase 5, or assumed?
4. Is `-inf` in logprobs expected from either real backend? The answer determines whether #9 is "clamp and warn" or "reject".
5. Does the Phase 4 exit contract require `ruitong port` to be CI-gateable? If yes, #1 is a hard blocker; if Phase 4 is explicitly "human-readable report only", say so in the CLI docstring and I'll re-grade #1 as P2.

**Re-audit scope:** P1 #1–#8 all need fixes plus tests that would fail without them (particularly a determinism test for #7 and a `passed is False` assertion for #2/#3). P2 #9, #10, #11, #16, #17 should land in the same cycle; the rest can be scheduled.
