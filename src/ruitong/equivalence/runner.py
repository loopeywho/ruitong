"""Equivalence runner — orchestrates comparison between two backends.

Runs a set of prompts through both backends and computes equivalence metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.base import Backend
from ..backends.fake import FakeAscend, FakeCuda
from ..schemas import ChatRequest, Message
from .metrics import (
    cosine_similarity,
    max_absolute_difference,
    top_k_agreement,
    top_k_set_agreement,
    top1_token_agreement,
    topk_token_set_agreement,
    top_k_max_abs_diff,
    probability_mass,
    count_non_finite,
    token_matched_prob_diff,
)


# ── Thresholds ──────────────────────────────────────────────────────

# Comparison must generate enough tokens for the metrics to mean anything:
# at max_tokens=1 the logprob vector has length 1, and cosine / top-1 / top-5
# are then constant 1.0 by construction.
DEFAULT_COMPARISON_TOKENS = 128
DEFAULT_TOP_LOGPROBS = 20


class Thresholds:
    """Pass/fail thresholds for equivalence."""

    # ── Primary gate, tier 1 (DECISIONS.md D12 — recalibrated on TWO
    # real GPUs, not one) ─────────────────────────────────────────────
    #
    # top1_agreement and probability_mass_delta together answer "did the
    # model's most-likely token ever change, and was probability mass
    # conserved". Both have zero exceptions across three independent real
    # cross-hardware runs (D9 warm/warm, D10 A40-vs-Ada, D11 A40-vs-Ada
    # again with 61 prompts). Kept exactly as calibrated.
    PROB_MASS_TOLERANCE: float = 0.01
    TOP1_MIN: float = 0.99

    # ── Primary gate, tier 2 — token-matched probability difference ──
    #
    # tier 1 alone has a real, measured blind spot: a fault that mutates
    # LOGPROB VALUES while leaving the TOKEN ARRAY untouched (a transposed-
    # operator bug is exactly this shape) moves top1_agreement and
    # probability_mass_delta by ZERO — permuting values within a row changes
    # neither which token nominally sits at rank 0 nor the row's sum. Only a
    # metric that matches probability BY TOKEN IDENTITY catches it.
    #
    # D9 used this metric as the sole primary gate at 2.2e-3, calibrated on
    # SIMULATED bf16 rounding as a noise proxy. D10/D11 measured REAL
    # cross-silicon noise (NVIDIA A40 vs RTX 6000 Ada) at up to 0.195 —
    # 89x the simulated estimate — which made that threshold reject every
    # correct cross-hardware port. The metric itself was never the problem;
    # the noise estimate it was calibrated against was.
    #
    #   real cross-hardware noise (D11, 61 prompts, max)  0.195   <- ceiling
    #   swap top-2 / shift / demote argmax                ~1.0
    #   corrupt 1-in-8 (weakest of this tier)              0.993  <- nearest fault
    #
    # 0.4402 is the geometric mean: 2.26x above measured noise, 2.26x below
    # the nearest fault in this tier. Deliberately NOT calibrated to catch
    # `scale x1.01` (a 1% temperature error, scores ~0.004-0.02 on this
    # metric) — no threshold that tolerates real cross-hardware noise (0.195)
    # can also catch a fault that small; that gap is inherent, not a
    # regression, and was already flagged in D9. It IS caught independently
    # by tier 1 above 1.05x scaling (probability_mass_delta) whenever the
    # scaling factor is large enough to move mass past 0.01.
    TOKEN_MATCHED_PROB_DIFF_MAX: float = 0.4402

    # ── Demoted to reported-only (D9 / D12) ─────────────────────────
    # `top_k_max_abs_diff` (D9): ranks by position, so tail near-ties compare
    # two *different* tokens; works in log space, so a shift at logprob -26
    # (p~5e-12) scores the same as one at the argmax. Two CORRECT executions
    # differing only in prefix-cache state scored up to 1.25 on the A40 —
    # above the weakest fault at 0.406. Noise and fault distributions
    # overlap on real data; no threshold separates them.
    #
    # `top5_set_agreement` (D12): checked against every fault in the
    # sensitivity suite — it catches nothing that top1_agreement does not
    # already catch (both only see shift_positions; both are blind to
    # value-only corruption, since neither reads a logprob). Meanwhile it
    # sits at 0.9167 on EVERY real cross-hardware measurement so far
    # (D10 and D11, identical to four decimal places), just under its own
    # 0.95 gate, for reasons unrelated to correctness. Zero unique coverage,
    # one more way to reject a correct port — retired.
    TOPK_MAX_ABS_DIFF_MAX: float = 0.05
    TOP5_MIN: float = 0.95

    # ── Reported but NOT gated (D9) ──────────────────────────────────
    # Both were measured unusable as gates and are kept for continuity of the
    # report only:
    #   cosine_similarity — scale-invariant by definition, so x1.01 through
    #       x2.0 all score exactly 1.0. Blind to scaling faults.
    #   max_absolute_difference — full-vocab, dominated by tail logprobs near
    #       -130 (p ~ 1e-57, never sampled). bf16 alone scores 0.4929 against
    #       a 0.05 threshold, so it would reject every correct port.
    COSINE_MIN: float = 0.99
    MAX_ABS_DIFF_MAX: float = 0.05


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class PerPromptResult:
    """Result for a single prompt comparison."""

    prompt: str
    mode: str
    cuda_logprobs: list[list[float]] | None
    ascend_logprobs: list[list[float]] | None
    cuda_response: str
    ascend_response: str
    cosine_sim: float | None = None
    max_abs_diff: float | None = None
    top1_agreement: float | None = None
    top5_set_agreement: float | None = None
    response_parity: float | None = None
    # Calibrated metrics — these are the ones that gate (D7).
    topk_max_abs_diff: float | None = None
    token_matched_prob_diff: float | None = None
    probability_mass_delta: float | None = None


@dataclass
class EquivalenceReport:
    """Overall equivalence comparison report."""

    model: str
    mode: str
    total_prompts: int
    # Coverage. A verdict computed from 1 of 100 prompts is not a verdict, and
    # an outage must be distinguishable from a genuine equivalence failure —
    # they demand opposite responses from whoever reads the report.
    compared_prompts: int = 0
    errored_prompts: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    per_prompt_results: list[PerPromptResult] = field(default_factory=list)
    passed: bool = False
    thresholds_used: Thresholds = field(default_factory=Thresholds)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "model": self.model,
            "mode": self.mode,
            "total_prompts": self.total_prompts,
            "compared_prompts": self.compared_prompts,
            "errored_prompts": self.errored_prompts,
            "metrics": self.metrics,
            "per_prompt_results": [
                {
                    "prompt": r.prompt,
                    "mode": r.mode,
                    "cuda_logprobs": r.cuda_logprobs,
                    "ascend_logprobs": r.ascend_logprobs,
                    "cuda_response": r.cuda_response,
                    "ascend_response": r.ascend_response,
                    "cosine_sim": r.cosine_sim,
                    "max_abs_diff": r.max_abs_diff,
                    "top1_agreement": r.top1_agreement,
                    "top5_set_agreement": r.top5_set_agreement,
                    "response_parity": r.response_parity,
                    "topk_max_abs_diff": r.topk_max_abs_diff,
                    "token_matched_prob_diff": r.token_matched_prob_diff,
                    "probability_mass_delta": r.probability_mass_delta,
                }
                for r in self.per_prompt_results
            ],
            "passed": self.passed,
            "thresholds_used": {
                "token_matched_prob_diff_max": self.thresholds_used.TOKEN_MATCHED_PROB_DIFF_MAX,
                "topk_max_abs_diff_max": self.thresholds_used.TOPK_MAX_ABS_DIFF_MAX,
                "prob_mass_tolerance": self.thresholds_used.PROB_MASS_TOLERANCE,
                "cosine_min": self.thresholds_used.COSINE_MIN,
                "max_abs_diff_max": self.thresholds_used.MAX_ABS_DIFF_MAX,
                "top1_min": self.thresholds_used.TOP1_MIN,
                "top5_min": self.thresholds_used.TOP5_MIN,
            },
            "warnings": self.warnings,
        }


# ── Runner ──────────────────────────────────────────────────────────

class EquivalenceRunner:
    """Compare two backends on a set of prompts.

    The two backends are treated as backend_a (expected cuda-like) and
    backend_b (expected ascend-like).  The ``run`` method sends each
    prompt through both and compares the outputs.
    """

    def __init__(self, backend_a: Backend, backend_b: Backend) -> None:
        self.backend_a = backend_a
        self.backend_b = backend_b

    @property
    def backend_a_name(self) -> str:
        return getattr(self.backend_a, "name", "unknown_a")

    @property
    def backend_b_name(self) -> str:
        return getattr(self.backend_b, "name", "unknown_b")

    @staticmethod
    async def _warm(backend: Backend, request: ChatRequest) -> None:
        """Prime one backend's prefix cache; the response is discarded.

        Failures are swallowed deliberately. This call exists only to put the
        backend in a known cache state — if the endpoint is genuinely broken
        the measured call that follows will say so, with a real error attached
        to a real comparison. Failing here instead would report the warm-up as
        the defect and hide what actually went wrong.
        """
        try:
            await backend.chat(request)
        except Exception:
            return

    async def run(
        self, model: str, prompts: list[str]
    ) -> EquivalenceReport:
        """Run equivalence comparison across *prompts*.

        Tries Mode 1 (teacher-forced logprob comparison) first. If both
        backends return logprobs, computes metrics and reports. Falls back
        to Mode 2 (task-level parity) if logprobs are unavailable.
        """
        per_prompt: list[PerPromptResult] = []
        all_cosines: list[float] = []
        all_tok_prob_diff: list[float] = []
        all_abs_diffs: list[float] = []
        all_top1: list[float] = []
        all_top5: list[float] = []
        all_topk_diff: list[float] = []
        all_mass_delta: list[float] = []
        all_response_parities: list[float] = []
        warnings: list[str] = []

        for prompt in prompts:
            request = ChatRequest(
                model=model,
                messages=[Message(role="user", content=prompt)],
                # max_tokens=1 makes logprob vectors length-1, which
                # forces cosine, top-1 and top-5 to a constant 1.0 —
                # three of four gate metrics become decoration.
                max_tokens=DEFAULT_COMPARISON_TOKENS,
                logprobs=True,
                top_logprobs=DEFAULT_TOP_LOGPROBS,
            )
            try:
                # ── Warm both sides before measuring (D9) ──────────────
                # A prompt's FIRST execution takes no prefix-cache hits and
                # returns measurably different logprobs from every later one.
                # Measured on an A40 serving Qwen3-8B (2026-07-27): the first
                # call scored +0 cache hits, the second +16, and the two
                # differed by up to 8.65e-02 in token-matched probability —
                # 24x the weakest fault this gate must catch. Once warm the
                # server is bit-exact across repeats (16/16 prompts).
                #
                # So cache state is not noise to be tolerated; it is a
                # variable to be held constant. Without this, whichever
                # backend happens to be colder looks broken, and the report
                # measures the cache instead of the silicon.
                await self._warm(self.backend_a, request)
                await self._warm(self.backend_b, request)

                resp_a = await self.backend_a.chat(request)
                resp_b = await self.backend_b.chat(request)
            except Exception as exc:
                warnings.append(
                    f"Failed to run prompt '{prompt[:50]}': "
                    f"{type(exc).__name__}: {exc}"
                )
                per_prompt.append(
                    PerPromptResult(
                        prompt=prompt,
                        mode="error",
                        cuda_logprobs=None,
                        ascend_logprobs=None,
                        cuda_response="",
                        ascend_response="",
                    )
                )
                continue

            # `choices[].logprobs` is an OpenAI object, not a bare list.
            # Compare the per-position top-k matrix — that is the actual
            # distribution, and it is what the calibrated metrics expect.
            lp_a = resp_a.choices[0].logprobs
            lp_b = resp_b.choices[0].logprobs
            logs_a = lp_a.top_k_matrix() if lp_a is not None else None
            logs_b = lp_b.top_k_matrix() if lp_b is not None else None
            # Token identity — what top-1/top-5 agreement is meant to compare.
            toks_a = lp_a.top_k_tokens() if lp_a is not None else None
            toks_b = lp_b.top_k_tokens() if lp_b is not None else None
            # A server can return the object with an empty content array;
            # treat that as "no logprobs" rather than comparing nothing.
            if not logs_a or not logs_b:
                logs_a = logs_b = None
            resp_a_text = resp_a.choices[0].message.content
            resp_b_text = resp_b.choices[0].message.content

            result = PerPromptResult(
                prompt=prompt,
                mode="unknown",
                cuda_logprobs=logs_a,
                ascend_logprobs=logs_b,
                cuda_response=resp_a_text,
                ascend_response=resp_b_text,
            )

            if logs_a is not None and logs_b is not None:
                # ── Refuse to grade a server that is emitting garbage ──
                # A correct log-softmax never emits -inf/+inf/NaN. When one
                # does, the fault is in the SERVER, not the port — and those
                # demand opposite responses (D8: exit 1 "the port is broken"
                # vs exit 2 "we could not tell").
                #
                # Real: vllm-ascend 0.9.1 on Qwen3-8B emits excessive -inf
                # where the same model on GPU does not (vllm-ascend #2934);
                # vLLM on ROCm returns -9999 sentinels (vllm#19305). Without
                # this, a *correct* Ascend port is condemned for an upstream
                # sampler defect — a false FAIL, and an expensive one to
                # debug because every metric looks plausibly bad.
                bad_a = count_non_finite(logs_a)
                bad_b = count_non_finite(logs_b)
                if bad_a or bad_b:
                    sides = []
                    if bad_a:
                        sides.append(f"{self.backend_a_name}={bad_a}")
                    if bad_b:
                        sides.append(f"{self.backend_b_name}={bad_b}")
                    warnings.append(
                        f"Non-finite logprobs for prompt '{prompt[:50]}' "
                        f"({', '.join(sides)}). A correct log-softmax cannot "
                        f"produce these; this is a server defect, not a port "
                        f"defect. Not graded. See vllm-ascend#2934."
                    )
                    result.mode = "unusable"
                    per_prompt.append(result)
                    continue

                # Mode 1: logprob comparison
                result.mode = "logprob"
                try:
                    # Calibrated (gating) metrics.
                    topk_diff = top_k_max_abs_diff(logs_a, logs_b, k=10)
                    # Primary gate (D9). Needs token identity, so it degrades
                    # to None rather than guessing when a backend omits it.
                    tok_prob_diff = (
                        token_matched_prob_diff(toks_a, logs_a, toks_b, logs_b, k=10)
                        if toks_a and toks_b
                        else None
                    )
                    # Compare mass BETWEEN backends, not against 1.0.
                    # An OpenAI server returns only the top-k of a ~150k
                    # vocabulary, so this mass is legitimately far below 1.0
                    # and an absolute check would be meaningless. But a
                    # scaling fault shifts one side's mass relative to the
                    # other, which is exactly what this catches — and it needs
                    # no access to the full distribution.
                    mass_delta = abs(
                        probability_mass(logs_a) - probability_mass(logs_b)
                    )
                    # Legacy metrics — reported, not gated.
                    cos = cosine_similarity(logs_a, logs_b)
                    abs_diff = max_absolute_difference(logs_a, logs_b)
                    # Compare TOKENS, not positional indices. The index-based
                    # metrics scored identical output as total disagreement and
                    # unrelated output as perfect agreement.
                    if toks_a and toks_b:
                        top1 = top1_token_agreement(toks_a, toks_b)
                        top5 = topk_token_set_agreement(toks_a, toks_b, k=5)
                    else:
                        top1 = top_k_agreement(logs_a, logs_b, k=1)
                        top5 = top_k_set_agreement(logs_a, logs_b, k=5)
                    result.topk_max_abs_diff = round(topk_diff, 6)
                    result.token_matched_prob_diff = (
                        round(tok_prob_diff, 9) if tok_prob_diff is not None else None
                    )
                    result.probability_mass_delta = round(mass_delta, 6)
                    result.cosine_sim = round(cos, 6)
                    result.max_abs_diff = round(abs_diff, 6)
                    result.top1_agreement = round(top1, 6)
                    result.top5_set_agreement = round(top5, 6)
                    all_topk_diff.append(topk_diff)
                    if tok_prob_diff is not None:
                        all_tok_prob_diff.append(tok_prob_diff)
                    all_mass_delta.append(mass_delta)
                    all_cosines.append(cos)
                    all_abs_diffs.append(abs_diff)
                    all_top1.append(top1)
                    all_top5.append(top5)
                except Exception as exc:
                    warnings.append(
                        f"Metric computation failed for prompt "
                        f"'{prompt[:50]}': {type(exc).__name__}: {exc}"
                    )
            else:
                # Mode 2: task-level parity fallback
                result.mode = "task_parity"
                if resp_a_text is not None and resp_b_text is not None:
                    parity = 1.0 if resp_a_text.strip() == resp_b_text.strip() else 0.0
                    result.response_parity = parity
                    all_response_parities.append(parity)
                warnings.append(
                    f"No logprobs for prompt '{prompt[:50]}' — using task parity"
                )

            per_prompt.append(result)

        errored = sum(1 for r in per_prompt if r.mode == "error")

        # Aggregate metrics
        report = EquivalenceReport(
            model=model,
            mode="logprob" if all_cosines else "task_parity",
            total_prompts=len(prompts),
            compared_prompts=len(prompts) - errored,
            errored_prompts=errored,
            metrics={
                # Worst case across prompts, not mean. A mean lets one
                # catastrophic prompt be averaged into silence, and makes the
                # gate *weaker* the more prompts you add.
                "token_matched_prob_diff": (
                    round(max(all_tok_prob_diff), 9) if all_tok_prob_diff else None
                ),
                "topk_max_abs_diff": (
                    round(max(all_topk_diff), 6) if all_topk_diff else None
                ),
                "probability_mass_delta": (
                    round(max(all_mass_delta), 6) if all_mass_delta else None
                ),
                "cosine_similarity": (
                    round(sum(all_cosines) / len(all_cosines), 6)
                    if all_cosines
                    else None
                ),
                "max_absolute_difference": (
                    round(sum(all_abs_diffs) / len(all_abs_diffs), 6)
                    if all_abs_diffs
                    else None
                ),
                "top1_agreement": (
                    round(sum(all_top1) / len(all_top1), 6)
                    if all_top1
                    else None
                ),
                "top5_set_agreement": (
                    round(sum(all_top5) / len(all_top5), 6)
                    if all_top5
                    else None
                ),
                "response_parity": (
                    round(sum(all_response_parities) / len(all_response_parities), 6)
                    if all_response_parities
                    else None
                ),
            },
            per_prompt_results=per_prompt,
            warnings=warnings,
        )

        # ── Gate (DECISIONS.md D12) ──────────────────────────────────
        #
        # Four are reported but NOT gated, each retired by measurement:
        #   cosine_similarity        — scale-invariant, blind to scaling faults
        #   max_absolute_difference  — full-vocab, rejects every correct port
        #   topk_max_abs_diff        — ranks by position and works in log
        #        space; noise (up to 1.25) exceeds its weakest fault (0.406)
        #   top5_set_agreement       — catches nothing top1_agreement does
        #        not already catch, while sitting at 0.9167 on every real
        #        cross-hardware run regardless of correctness (D12)
        #
        # The three that gate cover complementary failure classes: top1 and
        # mass catch whole-row/whole-distribution defects (position shifts,
        # scaling, catastrophic collapse); token_matched_prob_diff is the
        # only one that catches VALUE-ONLY corruption at a fixed token
        # position (e.g. a transposed-operator bug) — top1 and mass are both
        # structurally blind to that, since permuting values within a row
        # changes neither which token nominally sits at rank 0 nor the row's
        # sum. Its threshold was recalibrated in D12 against real two-GPU
        # noise (0.195) rather than the D9 simulated estimate (0.0013),
        # which had made it reject every correct cross-hardware port.
        gated: list[bool] = []
        m = report.metrics

        if m.get("token_matched_prob_diff") is not None:
            gated.append(
                m["token_matched_prob_diff"]
                <= Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX
            )
        if m.get("probability_mass_delta") is not None:
            gated.append(
                m["probability_mass_delta"] <= Thresholds.PROB_MASS_TOLERANCE
            )
        if m.get("top1_agreement") is not None:
            gated.append(m["top1_agreement"] >= Thresholds.TOP1_MIN)

        # No gated metric means nothing was measured. That is "could not
        # determine", never "equivalent" — a report with no evidence must not
        # certify a port.
        report.passed = bool(gated) and all(gated)

        return report
