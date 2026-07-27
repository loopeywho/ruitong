"""Sensitivity suite — does the gate still detect known-bad ports?

Test count and coverage say nothing about whether a gate *discriminates*.
This suite does: it injects faults that mimic real porting failures and
asserts each one still fails, while a correct BF16 port still passes.

**A fault this suite stops catching is a false PASS the product would ship.**
That is the worst possible outcome for a tool whose entire value is a trust
claim, so these tests are the real regression guard for the product — more so
than any unit test of an individual metric.

Numbers here trace to CALIBRATION.md and DECISIONS.md D7.
"""

from __future__ import annotations

import pytest

from ruitong.equivalence import faults as F
from ruitong.equivalence.metrics import probability_mass, top_k_max_abs_diff
from ruitong.equivalence.runner import Thresholds

# What a server actually returns: the top-k of a large vocabulary, never the
# full distribution. Calibrating on full-vocab data and gating on top-k data
# is how the previous thresholds ended up rejecting every correct port.
TOP_K = 20


def _wire_shape(tensor: list[list[float]]) -> list[list[float]]:
    """Reduce a full-vocab tensor to the top-k an OpenAI server would send."""
    return [sorted(row, reverse=True)[:TOP_K] for row in tensor]


@pytest.fixture
def reference() -> list[list[float]]:
    return _wire_shape(F.synthetic_logprobs(vocab=512, positions=8, seed=7))


def _gate(ref: list[list[float]], cand: list[list[float]]) -> bool:
    """Apply the calibrated gate exactly as the runner does."""
    topk_diff = top_k_max_abs_diff(ref, cand, k=10)
    mass_delta = abs(probability_mass(ref) - probability_mass(cand))
    return (
        topk_diff <= Thresholds.TOPK_MAX_ABS_DIFF_MAX
        and mass_delta <= Thresholds.PROB_MASS_TOLERANCE
    )


class TestEquivalentPortsPass:
    """A correct port must not be rejected — the failure mode that destroys
    trust on first contact."""

    def test_identical_passes(self, reference) -> None:
        assert _gate(reference, reference) is True

    def test_bfloat16_rounding_passes(self, reference) -> None:
        """The closest analogue to two hardware kernels disagreeing.

        Regression guard: the ORIGINAL full-vocab threshold scored this at
        0.4929 against a 0.05 limit, i.e. every correct BF16 port failed.
        """
        assert _gate(reference, F.to_bfloat16(reference)) is True


class TestFaultsAreDetected:
    """Every injected fault must fail the gate. Each mimics a real defect."""

    def test_scaled_logprobs_fail(self, reference) -> None:
        """Softmax/temperature bug. Invisible to cosine similarity, which is
        scale-invariant and scores this a perfect 1.0."""
        assert _gate(reference, F.scale_logprobs(reference, 1.05)) is False

    def test_swapped_top_tokens_fail(self, reference) -> None:
        """Transposed operator output."""
        assert _gate(reference, F.swap_tokens(reference, 0, 1)) is False

    def test_shifted_positions_fail(self, reference) -> None:
        """Off-by-one in KV-cache indexing."""
        assert _gate(reference, F.shift_positions(reference, 1)) is False

    def test_corrupted_fraction_fails(self, reference) -> None:
        """Intermittent kernel fault — the hardest to catch, because most
        positions agree and any metric that averages dilutes the damage."""
        assert _gate(reference, F.corrupt_fraction(reference, 0.125, 5.0)) is False


class TestSeparationMargin:
    """The gate is only meaningful if noise and faults are far apart.

    The original full-vocab metric separated them by 2.2% — no threshold could
    sit in that gap. This asserts the margin survives.
    """

    def test_margin_is_at_least_five_fold(self, reference) -> None:
        noise = top_k_max_abs_diff(reference, F.to_bfloat16(reference), k=10)
        faults = [
            top_k_max_abs_diff(reference, candidate, k=10)
            for candidate in (
                F.scale_logprobs(reference, 1.05),
                F.swap_tokens(reference, 0, 1),
                F.shift_positions(reference, 1),
            )
        ]
        assert noise > 0, "bf16 rounding should produce a non-zero difference"
        assert min(faults) / noise >= 5.0, (
            f"separation collapsed to {min(faults) / noise:.1f}x "
            f"(noise={noise:.4f}, weakest fault={min(faults):.4f}); "
            "the gate can no longer distinguish a correct port from a broken one"
        )

    def test_threshold_sits_between_noise_and_faults(self, reference) -> None:
        """The threshold must be above the noise floor and below every fault,
        or it is guaranteed to produce either false passes or false failures."""
        noise = top_k_max_abs_diff(reference, F.to_bfloat16(reference), k=10)
        weakest_fault = min(
            top_k_max_abs_diff(reference, candidate, k=10)
            for candidate in (
                F.scale_logprobs(reference, 1.05),
                F.swap_tokens(reference, 0, 1),
                F.shift_positions(reference, 1),
            )
        )
        assert noise < Thresholds.TOPK_MAX_ABS_DIFF_MAX < weakest_fault
