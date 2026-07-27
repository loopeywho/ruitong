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
)


# ── Thresholds ──────────────────────────────────────────────────────

# Comparison must generate enough tokens for the metrics to mean anything:
# at max_tokens=1 the logprob vector has length 1, and cosine / top-1 / top-5
# are then constant 1.0 by construction.
DEFAULT_COMPARISON_TOKENS = 128
DEFAULT_TOP_LOGPROBS = 20


class Thresholds:
    """Pass/fail thresholds for equivalence."""

    # ── Gating thresholds (DECISIONS.md D7 — measured, not assumed) ──
    #
    # top-k max abs diff: bf16 rounding measures 0.0152; the weakest injected
    # fault measures 0.2341. 0.05 sits ~3x above the noise and ~5x below the
    # nearest fault. See CALIBRATION.md.
    TOPK_MAX_ABS_DIFF_MAX: float = 0.05
    # probability-mass DELTA between the two backends (top-k only, so an
    # absolute check against 1.0 would be meaningless). A x1.05 scaling fault
    # shifts one side by ~0.096 — the defect class top-k max-abs-diff misses.
    PROB_MASS_TOLERANCE: float = 0.01
    TOP1_MIN: float = 0.99
    TOP5_MIN: float = 0.95

    # ── Reported but NOT gated ──────────────────────────────────────
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
                    "probability_mass_delta": r.probability_mass_delta,
                }
                for r in self.per_prompt_results
            ],
            "passed": self.passed,
            "thresholds_used": {
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
        all_abs_diffs: list[float] = []
        all_top1: list[float] = []
        all_top5: list[float] = []
        all_topk_diff: list[float] = []
        all_mass_delta: list[float] = []
        all_response_parities: list[float] = []
        warnings: list[str] = []

        for prompt in prompts:
            try:
                resp_a = await self.backend_a.chat(
                    ChatRequest(
                        model=model,
                        messages=[Message(role="user", content=prompt)],
                        # max_tokens=1 makes logprob vectors length-1, which
                        # forces cosine, top-1 and top-5 to a constant 1.0 —
                        # three of four gate metrics become decoration.
                        max_tokens=DEFAULT_COMPARISON_TOKENS,
                        logprobs=True,
                        top_logprobs=DEFAULT_TOP_LOGPROBS,
                    )
                )
                resp_b = await self.backend_b.chat(
                    ChatRequest(
                        model=model,
                        messages=[Message(role="user", content=prompt)],
                        # max_tokens=1 makes logprob vectors length-1, which
                        # forces cosine, top-1 and top-5 to a constant 1.0 —
                        # three of four gate metrics become decoration.
                        max_tokens=DEFAULT_COMPARISON_TOKENS,
                        logprobs=True,
                        top_logprobs=DEFAULT_TOP_LOGPROBS,
                    )
                )
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
                # Mode 1: logprob comparison
                result.mode = "logprob"
                try:
                    # Calibrated (gating) metrics.
                    topk_diff = top_k_max_abs_diff(logs_a, logs_b, k=10)
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
                    result.probability_mass_delta = round(mass_delta, 6)
                    result.cosine_sim = round(cos, 6)
                    result.max_abs_diff = round(abs_diff, 6)
                    result.top1_agreement = round(top1, 6)
                    result.top5_set_agreement = round(top5, 6)
                    all_topk_diff.append(topk_diff)
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

        # ── Gate (DECISIONS.md D7) ──────────────────────────────────
        #
        # Only the calibrated metrics gate. cosine_similarity and full-vocab
        # max_absolute_difference are computed and reported for continuity,
        # but deliberately NOT gated: measurement showed the first is blind to
        # scaling faults and the second rejects every correct BF16 port.
        gated: list[bool] = []
        m = report.metrics

        if m.get("topk_max_abs_diff") is not None:
            gated.append(
                m["topk_max_abs_diff"] <= Thresholds.TOPK_MAX_ABS_DIFF_MAX
            )
        if m.get("probability_mass_delta") is not None:
            gated.append(
                m["probability_mass_delta"] <= Thresholds.PROB_MASS_TOLERANCE
            )
        if m.get("top1_agreement") is not None:
            gated.append(m["top1_agreement"] >= Thresholds.TOP1_MIN)
        if m.get("top5_set_agreement") is not None:
            gated.append(m["top5_set_agreement"] >= Thresholds.TOP5_MIN)

        # No gated metric means nothing was measured. That is "could not
        # determine", never "equivalent" — a report with no evidence must not
        # certify a port.
        report.passed = bool(gated) and all(gated)

        return report
