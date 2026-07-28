"""Sensitivity suite — does the gate still detect known-bad ports?

Test count and coverage say nothing about whether a gate *discriminates*.
This suite does: it injects faults that mimic real porting failures and
asserts each one still fails, while a correct port still passes.

**A fault this suite stops catching is a false PASS the product would ship.**
That is the worst possible outcome for a tool whose entire value is a trust
claim, so these tests are the real regression guard for the product — more so
than any unit test of an individual metric.

Everything here runs against `corpora/cuda_a40_qwen3_8b.json`: real Qwen3-8B
logprobs captured from an NVIDIA A40. The previous version of this suite ran
on synthetic logprobs and passed while the gate it guarded was rejecting the
reference model compared with itself — synthetic data had a convenient
distribution that real model output does not share.

Numbers here trace to CALIBRATION.md and DECISIONS.md D9/D12.

The gate is now COMPOUND (D12): token_matched_prob_diff, probability_mass_delta
and top1_agreement together, ANY failing -> overall FAIL. `_gate()` below
reimplements that exact composition — testing tmpd in isolation, as the D9
version of this file did, would have hidden the real D12 finding: a fault that
mutates logprob VALUES while leaving the TOKEN ARRAY untouched (swap-top-2)
moves top1_agreement and probability_mass_delta by *zero*, so only tmpd
catches it. Any one of these three metrics tested alone is incomplete.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ruitong.equivalence import faults as F
from ruitong.equivalence.metrics import (
    probability_mass,
    token_matched_prob_diff,
    top1_token_agreement,
)
from ruitong.equivalence.runner import EquivalenceRunner, Thresholds

CORPUS_PATH = Path(__file__).resolve().parents[1] / "corpora" / "cuda_a40_qwen3_8b.json"

# Measured cold-vs-warm spread on the A40 (DECISIONS.md D9). Not a fault —
# two correct executions — but larger than every fault below, which is why the
# runner warms both backends instead of tolerating it.
MEASURED_COLD_WARM_SPREAD = 8.65e-02


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        pytest.skip(f"real corpus not present at {CORPUS_PATH}")
    return json.loads(CORPUS_PATH.read_text())["entries"]


# ── Fault injectors, applied to real captured output ────────────────


def _scale(rows: list[list[float]], factor: float) -> list[list[float]]:
    return [[v * factor for v in row] for row in rows]


def _swap(rows: list[list[float]], i: int = 0, j: int = 1) -> list[list[float]]:
    out = []
    for row in rows:
        new = list(row)
        if len(new) > max(i, j):
            new[i], new[j] = new[j], new[i]
        out.append(new)
    return out


def _shift(rows: list, by: int = 1) -> list:
    return rows[by:] + rows[:by] if len(rows) > by else rows


def _corrupt(rows: list[list[float]], every: int = 8, delta: float = 5.0) -> list[list[float]]:
    return [
        [v - delta for v in row] if i % every == 0 else list(row)
        for i, row in enumerate(rows)
    ]


def _demote_argmax(rows: list[list[float]], floor: float = -30.0) -> list[list[float]]:
    out = []
    for row in rows:
        new = list(row)
        if new:
            new[0] = floor
        out.append(new)
    return out


def _bf16(rows: list[list[float]]) -> list[list[float]]:
    return [[F.bfloat16_round(v) for v in row] for row in rows]


def _worst(corpus: list[dict], transform) -> float:
    """Worst token-matched probability difference across the whole corpus."""
    worst = 0.0
    for entry in corpus:
        tokens, logs = entry["top_tokens"], entry["top_logprobs"]
        cand_tokens, cand_logs = transform(tokens, logs)
        n = min(len(tokens), len(cand_tokens))
        if n == 0:
            continue
        worst = max(
            worst,
            token_matched_prob_diff(
                tokens[:n], logs[:n], cand_tokens[:n], cand_logs[:n], k=10
            ),
        )
    return worst


def _compound_gate(corpus: list[dict], transform) -> bool:
    """The actual product gate (D12): tmpd AND mass AND top1, worst case
    across the corpus, exactly mirroring EquivalenceRunner's composition.
    Returns True if the candidate would be certified PASS.
    """
    worst_tmpd = 0.0
    worst_mass = 0.0
    worst_top1 = 1.0
    for entry in corpus:
        tokens, logs = entry["top_tokens"], entry["top_logprobs"]
        cand_tokens, cand_logs = transform(tokens, logs)
        n = min(len(tokens), len(cand_tokens))
        if n == 0:
            continue
        ta, la = tokens[:n], logs[:n]
        tb, lb = cand_tokens[:n], cand_logs[:n]
        worst_tmpd = max(worst_tmpd, token_matched_prob_diff(ta, la, tb, lb, k=10))
        worst_mass = max(worst_mass, abs(probability_mass(la) - probability_mass(lb)))
        worst_top1 = min(worst_top1, top1_token_agreement(ta, tb))
    return (
        worst_tmpd <= Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX
        and worst_mass <= Thresholds.PROB_MASS_TOLERANCE
        and worst_top1 >= Thresholds.TOP1_MIN
    )


def _gate(value: float) -> bool:
    """tmpd-only check — kept for the tests that isolate this one metric."""
    return value <= Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX


# ── Correct ports must pass ─────────────────────────────────────────


class TestEquivalentPortsPass:
    """A correct port must not be rejected — the failure mode that destroys
    trust on first contact."""

    def test_identical_scores_exactly_zero(self, corpus) -> None:
        """Reflexivity. The predecessor metric scored an identical tensor at
        0.67 against itself, because it keyed a dict on token strings and many
        distinct token ids decode to the same string (nine `""` in one row)."""
        assert _worst(corpus, lambda t, l: (t, l)) == 0.0

    def test_bfloat16_rounding_passes(self, corpus) -> None:
        """The closest analogue to two hardware kernels disagreeing."""
        assert _compound_gate(corpus, lambda t, l: (t, _bf16(l))) is True

    def test_real_cross_hardware_port_passes(self) -> None:
        """The actual D10/D11 measurement: NVIDIA A40 vs RTX 6000 Ada, same
        model, same code. A CORRECT port, per D10/D11's own top1_agreement
        (1.000, no exceptions) and mass evidence — must certify PASS.

        This is the test the D9 gate could never pass: it rejected the
        reference model compared with itself-on-different-silicon. If this
        regresses, the product is back to rejecting every correct port.
        """
        a40 = Path(__file__).resolve().parents[1] / "corpora" / "a40_61.json"
        ada = Path(__file__).resolve().parents[1] / "corpora" / "rtx6000ada_61.json"
        if not (a40.exists() and ada.exists()):
            pytest.skip("cross-hardware corpora not present")
        ref = {e["prompt"]: e for e in json.loads(a40.read_text())["entries"]}
        cand = {e["prompt"]: e for e in json.loads(ada.read_text())["entries"]}
        common = [ref[p] for p in ref if p in cand]

        worst_tmpd = worst_mass = 0.0
        worst_top1 = 1.0
        for e in common:
            other = cand[e["prompt"]]
            # Truncate at first divergent sampled token — see D10/D11 and
            # tools/compare_corpora.py for why comparing past a divergence
            # measures the divergence, not the hardware.
            sa, sb = e["sampled_tokens"], other["sampled_tokens"]
            n = min(len(sa), len(sb))
            div = next((i for i in range(n) if sa[i] != sb[i]), n)
            if div == 0:
                continue
            ta, la = e["top_tokens"][:div], e["top_logprobs"][:div]
            tb, lb = other["top_tokens"][:div], other["top_logprobs"][:div]
            worst_tmpd = max(worst_tmpd, token_matched_prob_diff(ta, la, tb, lb, k=10))
            worst_mass = max(worst_mass, abs(probability_mass(la) - probability_mass(lb)))
            worst_top1 = min(worst_top1, top1_token_agreement(ta, tb))

        passed = (
            worst_tmpd <= Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX
            and worst_mass <= Thresholds.PROB_MASS_TOLERANCE
            and worst_top1 >= Thresholds.TOP1_MIN
        )
        assert passed, (
            f"real A40-vs-Ada port rejected: tmpd={worst_tmpd:.4f} "
            f"mass={worst_mass:.6f} top1={worst_top1:.4f}"
        )


# ── Faults must fail ────────────────────────────────────────────────


class TestFaultsAreDetected:
    """Every injected fault must fail the compound gate. Each mimics a real
    defect. Testing the compound gate, not tmpd alone, matters here: several
    of these are caught by mass or top1 independently of tmpd, and one
    (swap-top-2) is caught ONLY by tmpd — see the module docstring."""

    def test_scale_1_05_fails(self, corpus) -> None:
        """Softmax/temperature bug. Invisible to cosine similarity (scale-
        invariant, scores 1.0) and to tmpd at the D12 ceiling (0.0179, well
        under 0.4402) — caught by probability_mass_delta (0.023 > 0.01)."""
        assert _compound_gate(corpus, lambda t, l: (t, _scale(l, 1.05))) is False

    def test_swapped_top_tokens_fail(self, corpus) -> None:
        """Transposed operator output. The one fault ONLY tmpd catches:
        permuting logprob values within a row leaves the token array and the
        row's sum unchanged, so top1_agreement and probability_mass_delta
        both score a perfect zero here (verified empirically before this
        threshold was set — see DECISIONS.md D12)."""
        assert _compound_gate(corpus, lambda t, l: (t, _swap(l, 0, 1))) is False

    def test_shifted_positions_fail(self, corpus) -> None:
        """Off-by-one in KV-cache indexing. Caught via top1_agreement (the
        token compared at each position is now genuinely a different
        generation step) independently of tmpd."""
        assert _compound_gate(corpus, lambda t, l: (_shift(t), _shift(l))) is False

    def test_corrupted_fraction_fails(self, corpus) -> None:
        """Intermittent kernel fault. Caught via probability_mass_delta: a
        flat per-row shift collapses that row's mass by ~e^-5 (~0.25 worst
        case), far past the 0.01 tolerance."""
        assert _compound_gate(corpus, lambda t, l: (t, _corrupt(l))) is False

    def test_demoted_argmax_fails(self, corpus) -> None:
        """Catastrophic port failure. Caught via probability_mass_delta —
        zeroing the argmax's contribution collapses row mass by ~1.0."""
        assert _compound_gate(corpus, lambda t, l: (t, _demote_argmax(l))) is False


class TestKnownDetectionGap:
    """`scale x1.01` is a disclosed, unresolved gap — not a silent one.

    A 1% temperature error scores ~0.004 on tmpd and ~0.005 on mass, both
    comfortably under their D12 thresholds; top1_agreement never moves for a
    pure scale (rank order is preserved). No metric here catches it, and none
    can: any threshold wide enough to tolerate real cross-hardware noise
    (tmpd ceiling 0.44, set from a measured max of 0.195) is necessarily wide
    enough to also miss a fault that scores 50x smaller than that ceiling.

    This was flagged as "at or past the detection limit" under the ORIGINAL
    D9 calibration too — it is not a regression introduced by D12, just now
    honestly true for cross-hardware comparisons specifically. This test
    exists so the gap stays visible: if it ever starts failing (the gate
    starts catching x1.01), that's an improvement worth noting in
    DECISIONS.md, not a silent behaviour change.
    """

    def test_scale_1_01_is_not_caught(self, corpus) -> None:
        assert _compound_gate(corpus, lambda t, l: (t, _scale(l, 1.01))) is True


# ── The gate must sit in a real gap ─────────────────────────────────


class TestSeparationMargin:
    """The gate is only meaningful if noise and faults are far apart.

    D12 anchors the noise ceiling to a REAL measurement (D11: two NVIDIA GPUs,
    61 prompts, max 0.195), not bf16 rounding — D10 already showed bf16 is a
    95x-too-optimistic proxy for real hardware disagreement and must not be
    used to justify a threshold again. bf16 stays useful as a "must not
    reject" floor (TestEquivalentPortsPass), just not as the noise ceiling.

    The fault side is restricted to the "severe" tier tmpd is now calibrated
    for (swap/shift/corrupt/demote — the value-corruption-at-fixed-position
    class). Scale faults belong to probability_mass_delta's jurisdiction
    (TestProbabilityMassStillGuards) and are deliberately excluded here.
    """

    # D11, corpora/a40_61.json vs rtx6000ada_61.json, worst observed.
    MEASURED_CROSS_HARDWARE_NOISE_MAX = 0.195113840

    def test_threshold_sits_between_real_noise_and_severe_faults(self, corpus) -> None:
        weakest_severe_fault = min(
            _worst(corpus, transform)
            for transform in (
                lambda t, l: (t, _swap(l, 0, 1)),
                lambda t, l: (_shift(t), _shift(l)),
                lambda t, l: (t, _corrupt(l)),
                lambda t, l: (t, _demote_argmax(l)),
            )
        )
        assert (
            self.MEASURED_CROSS_HARDWARE_NOISE_MAX
            < Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX
            < weakest_severe_fault
        ), (
            f"threshold {Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX:.4f} must lie "
            f"strictly between measured real noise "
            f"{self.MEASURED_CROSS_HARDWARE_NOISE_MAX:.4f} and the weakest "
            f"severe fault {weakest_severe_fault:.4f}"
        )

    def test_cache_state_no_longer_separated_by_tmpd_alone(self, corpus) -> None:
        """Honest tradeoff of D12, stated rather than hidden.

        Two CORRECT executions differing only in prefix-cache state measured
        8.65e-02 apart on tmpd (D9) — comfortably UNDER the D12 ceiling
        (0.4402), unlike the old 2.2e-3 threshold this margin was originally
        written against. tmpd alone can no longer distinguish cold-vs-warm
        noise from a genuine severe fault.

        This does NOT mean cache state is safe to leave uncontrolled — the
        runner's `_warm` step is unconditional and structural
        (`test_equivalence.py::TestWarmUpPass` asserts it runs twice per
        prompt per backend, not a numeric margin), so this gap is closed by
        code, not by threshold placement. Whether probability_mass_delta or
        top1_agreement would ALSO catch un-warmed cache noise is not
        measured — D9's cold/warm experiment recorded tmpd and
        topk_max_abs_diff only. Do not assume they would.
        """
        assert MEASURED_COLD_WARM_SPREAD < Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX, (
            "if this ever flips, tmpd alone would separate cache noise again "
            "and the docstring above should be revisited"
        )


class TestProbabilityMassStillGuards:
    """Mass delta catches the scaling class independently of the primary gate."""

    def test_scaling_moves_probability_mass(self, corpus) -> None:
        worst = 0.0
        for entry in corpus:
            logs = entry["top_logprobs"]
            worst = max(
                worst,
                abs(probability_mass(logs) - probability_mass(_scale(logs, 1.05))),
            )
        assert worst > Thresholds.PROB_MASS_TOLERANCE

    def test_correct_port_keeps_mass(self, corpus) -> None:
        worst = 0.0
        for entry in corpus:
            logs = entry["top_logprobs"]
            worst = max(
                worst, abs(probability_mass(logs) - probability_mass(_bf16(logs)))
            )
        assert worst <= Thresholds.PROB_MASS_TOLERANCE
        assert not math.isnan(worst)


class _StubBackend:
    """Minimal backend returning one position with a fixed logprob."""

    def __init__(self, name: str, logprob: float) -> None:
        self.name = name
        self._logprob = logprob

    async def health(self):  # pragma: no cover
        raise NotImplementedError

    async def list_models(self):  # pragma: no cover
        raise NotImplementedError

    async def chat(self, req):
        from ruitong.schemas import (
            ChatResponse, Choice, ChoiceLogprobs, LogprobEntry,
            Message, TopLogprob, Usage,
        )
        return ChatResponse(
            id="x", model="m", backend=self.name,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content="hi"),
                finish_reason="stop",
                logprobs=ChoiceLogprobs(content=[LogprobEntry(
                    token="hi", logprob=self._logprob,
                    top_logprobs=[
                        TopLogprob(token="hi", logprob=self._logprob),
                        TopLogprob(token="yo", logprob=-5.0),
                    ],
                )]),
            )],
        )

    async def stream(self, req):  # pragma: no cover
        raise NotImplementedError
        yield  # type: ignore[unreachable]


class _RowBackend:
    """Backend serving one fixed row of (token, logprob) top-k pairs.

    Unlike `_StubBackend` (a single hardcoded pair), this takes the full row
    so a genuine swap-top-2 fault can be constructed at the wire-format
    level and driven through the REAL `EquivalenceRunner`, not just through
    the metric functions directly.
    """

    def __init__(self, name: str, row: list[tuple[str, float]]) -> None:
        self.name = name
        self._row = row

    async def health(self):  # pragma: no cover
        raise NotImplementedError

    async def list_models(self):  # pragma: no cover
        raise NotImplementedError

    async def chat(self, req):
        from ruitong.schemas import (
            ChatResponse, Choice, ChoiceLogprobs, LogprobEntry,
            Message, TopLogprob, Usage,
        )
        top_token, top_logprob = self._row[0]
        return ChatResponse(
            id="x", model="m", backend=self.name,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=top_token),
                finish_reason="stop",
                logprobs=ChoiceLogprobs(content=[LogprobEntry(
                    token=top_token, logprob=top_logprob,
                    top_logprobs=[TopLogprob(token=t, logprob=lp) for t, lp in self._row],
                )]),
            )],
        )

    async def stream(self, req):  # pragma: no cover
        raise NotImplementedError
        yield  # type: ignore[unreachable]


class TestRunnerGateIntegration:
    """Drives the REAL `EquivalenceRunner`, not a reimplementation of its gate.

    A mutation test on `runner.py` itself (removing the tmpd check from its
    gate composition) proved the other tests in this file do NOT catch that
    class of regression: `_compound_gate` above reimplements the gate formula
    independently, so it stays correct even if the actual runner's wiring
    breaks. This class exists to close that gap — it is the only test in the
    suite that would fail if `runner.py`'s gate logic itself regressed.
    """

    async def test_swap_top2_fails_through_the_real_runner(self) -> None:
        # Same token array on both sides, in the same order — only the
        # LOGPROB values at rank 0/1 are swapped. This is what the fault
        # injectors above model, and what a transposed-operator bug looks
        # like on the wire: the server still lists tokens in its own
        # (correct) order, but attaches the wrong confidence to the wrong one.
        reference_row = [("the", -0.10), ("a", -2.50), ("an", -4.00)]
        swapped_row = [("the", -2.50), ("a", -0.10), ("an", -4.00)]

        runner = EquivalenceRunner(
            _RowBackend("a", reference_row), _RowBackend("b", swapped_row)
        )
        report = await runner.run("m", ["p"])

        assert report.passed is False, (
            "a transposed-operator fault (token order unchanged, confidence "
            "swapped) must not certify as an equivalent port"
        )


class TestNonFiniteLogprobsAreRefused:
    """A server emitting -inf/NaN is a SERVER defect, not a port defect.

    `vllm-ascend` 0.9.1 on Qwen3-8B emits excessive -inf where the same model
    on GPU does not (vllm-ascend#2934); vLLM on ROCm returns -9999 sentinels
    (vllm#19305). Both would otherwise condemn a *correct* port.

    Note the failure mode is the opposite of the one first supposed: -inf is
    NOT silent against the current gate, because exp(-inf) is 0.0 and the
    resulting probability gap is large. The risk is a false FAIL, not a false
    PASS — which is why these are refused rather than graded.
    """

    def test_inf_is_detected(self) -> None:
        from ruitong.equivalence.metrics import count_non_finite

        assert count_non_finite([[-0.1, -3.0]]) == 0
        assert count_non_finite([[-0.1, float("-inf")]]) == 1
        assert count_non_finite([[float("nan"), -3.0]]) == 1

    def test_inf_detection_is_prompt_dependent(self, corpus) -> None:
        """Why refusing beats grading: detection depends on the prompt.

        Corrupting rank 1 scores 5.0e-01 on the worst prompt in the corpus —
        loudly over the 2.2e-03 gate — but only 3.5e-05 on a highly confident
        prompt like "What is the capital of France?", where rank 1 already
        carries almost no probability mass. So whether an upstream -inf bug is
        caught depends on which prompts happen to be in the suite.

        A detector whose sensitivity varies by two orders of magnitude with
        the input is not something to gate on. Refuse instead.
        """
        def corrupt_rank1(entry):
            t, l = entry["top_tokens"], entry["top_logprobs"]
            c = [
                [float("-inf") if i == 1 else v for i, v in enumerate(row)]
                for row in l
            ]
            return token_matched_prob_diff(t, l, t, c, k=10)

        scores = [corrupt_rank1(e) for e in corpus]
        assert max(scores) > Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX
        assert min(scores) < Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX, (
            "if every prompt caught it, grading would be safe and the "
            "refuse-instead guard would be unnecessary"
        )

    async def test_runner_refuses_rather_than_grading(self) -> None:
        """The whole point: mode becomes 'unusable', not a pass/fail verdict."""
        from ruitong.equivalence.runner import EquivalenceRunner

        runner = EquivalenceRunner(
            _StubBackend("a", -0.1), _StubBackend("b", float("-inf"))
        )
        report = await runner.run("m", ["p"])

        assert report.per_prompt_results[0].mode == "unusable"
        assert not report.passed, "must not certify against a faulty server"
        assert any("Non-finite" in w for w in report.warnings)
