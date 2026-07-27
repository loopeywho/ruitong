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

Numbers here trace to CALIBRATION.md and DECISIONS.md D9.
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
)
from ruitong.equivalence.runner import Thresholds

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


def _gate(value: float) -> bool:
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
        assert _gate(_worst(corpus, lambda t, l: (t, _bf16(l)))) is True


# ── Faults must fail ────────────────────────────────────────────────


class TestFaultsAreDetected:
    """Every injected fault must fail the gate. Each mimics a real defect."""

    def test_scale_1_05_fails(self, corpus) -> None:
        """Softmax/temperature bug. Invisible to cosine similarity, which is
        scale-invariant and scores this a perfect 1.0."""
        assert _gate(_worst(corpus, lambda t, l: (t, _scale(l, 1.05)))) is False

    def test_scale_1_01_fails(self, corpus) -> None:
        """The detection limit — a 1% temperature error, the weakest fault the
        gate separates from bf16 noise."""
        assert _gate(_worst(corpus, lambda t, l: (t, _scale(l, 1.01)))) is False

    def test_swapped_top_tokens_fail(self, corpus) -> None:
        """Transposed operator output."""
        assert _gate(_worst(corpus, lambda t, l: (t, _swap(l, 0, 1)))) is False

    def test_shifted_positions_fail(self, corpus) -> None:
        """Off-by-one in KV-cache indexing."""
        assert _gate(_worst(corpus, lambda t, l: (_shift(t), _shift(l)))) is False

    def test_corrupted_fraction_fails(self, corpus) -> None:
        """Intermittent kernel fault — the hardest to catch, because most
        positions agree and any metric that averages dilutes the damage."""
        assert _gate(_worst(corpus, lambda t, l: (t, _corrupt(l)))) is False

    def test_demoted_argmax_fails(self, corpus) -> None:
        """Catastrophic port failure: the most likely token is no longer."""
        assert _gate(_worst(corpus, lambda t, l: (t, _demote_argmax(l)))) is False


# ── The gate must sit in a real gap ─────────────────────────────────


class TestSeparationMargin:
    """The gate is only meaningful if noise and faults are far apart."""

    def test_threshold_sits_between_noise_and_faults(self, corpus) -> None:
        noise = _worst(corpus, lambda t, l: (t, _bf16(l)))
        weakest_fault = min(
            _worst(corpus, transform)
            for transform in (
                lambda t, l: (t, _scale(l, 1.01)),
                lambda t, l: (t, _scale(l, 1.05)),
                lambda t, l: (t, _swap(l, 0, 1)),
                lambda t, l: (_shift(t), _shift(l)),
            )
        )
        assert noise > 0, "bf16 rounding should produce a non-zero difference"
        assert noise < Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX < weakest_fault, (
            f"threshold {Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX:.3e} must lie "
            f"strictly between noise {noise:.3e} and weakest fault "
            f"{weakest_fault:.3e}"
        )

    def test_cache_state_must_be_controlled_not_tolerated(self, corpus) -> None:
        """Justifies the runner's warm-up pass.

        Two CORRECT executions differing only in prefix-cache state measured
        8.65e-02 apart on the A40 — larger than every fault above. A gate wide
        enough to tolerate that would pass a demoted argmax. The only sound
        response is to hold cache state constant, which is what the runner's
        `_warm` does; this test fails if anyone tries to "fix" cold-vs-warm
        noise by widening the threshold instead.
        """
        weakest_fault = _worst(corpus, lambda t, l: (t, _scale(l, 1.01)))
        assert MEASURED_COLD_WARM_SPREAD > weakest_fault
        assert Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX < MEASURED_COLD_WARM_SPREAD


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
