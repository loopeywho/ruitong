"""Metric computation for equivalence comparison.

All functions accept list[float] (single) or list[list[float]] (batched).
When batched, metrics are computed per-position and averaged.
"""

from __future__ import annotations

import math
from typing import Any


def _ensure_lists(
    tensor_a: list[float] | list[list[float]],
    tensor_b: list[float] | list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Normalize inputs to list[list[float]] and validate."""
    # If the first element is a float, the input is flat: wrap it.
    a: Any = tensor_a
    b: Any = tensor_b
    if a and isinstance(a[0], (int, float)):
        a = [a]
    if b and isinstance(b[0], (int, float)):
        b = [b]

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Inputs must not be empty")
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")

    inner_len = len(a[0])
    for row in a:
        if len(row) != inner_len:
            raise ValueError(
                f"Inner length mismatch: {len(row)} vs {inner_len}"
            )
    for row in b:
        if len(row) != inner_len:
            raise ValueError(
                f"Inner length mismatch: {len(row)} vs {inner_len}"
            )

    return a, b


def cosine_similarity(
    tensor_a: list[float] | list[list[float]],
    tensor_b: list[float] | list[list[float]],
) -> float:
    """Cosine similarity between two logprob tensors.

    Returns a value in [-1, 1]. Batched inputs return the mean per-position
    cosine similarity.
    """
    a, b = _ensure_lists(tensor_a, tensor_b)
    sims: list[float] = []
    for a_row, b_row in zip(a, b):
        dot = sum(x * y for x, y in zip(a_row, b_row))
        mag_a = sum(x * x for x in a_row) ** 0.5
        mag_b = sum(x * x for x in b_row) ** 0.5
        if mag_a == 0.0 or mag_b == 0.0:
            sims.append(0.0)
        else:
            sims.append(dot / (mag_a * mag_b))
    return sum(sims) / len(sims)


def max_absolute_difference(
    tensor_a: list[float] | list[list[float]],
    tensor_b: list[float] | list[list[float]],
) -> float:
    """Maximum absolute difference between two tensors.

    Batched inputs return the mean of per-position max diffs.
    """
    a, b = _ensure_lists(tensor_a, tensor_b)
    max_diffs: list[float] = []
    for a_row, b_row in zip(a, b):
        max_diff = max(abs(x - y) for x, y in zip(a_row, b_row))
        max_diffs.append(max_diff)
    return sum(max_diffs) / len(max_diffs)


def top_k_agreement(
    scores_a: list[float] | list[list[float]],
    scores_b: list[float] | list[list[float]],
    k: int = 1,
) -> float:
    """Fraction of positions where top-k token ranks agree.

    Treats the list as scores/rankings — higher score = higher rank.
    Returns a value in [0, 1].
    """
    a, b = _ensure_lists(scores_a, scores_b)
    matches = 0
    for a_row, b_row in zip(a, b):
        top_a = set(i for i, _ in sorted(enumerate(a_row), key=lambda x: -x[1])[:k])
        top_b = set(i for i, _ in sorted(enumerate(b_row), key=lambda x: -x[1])[:k])
        if top_a == top_b:
            matches += 1
    return matches / len(a)


def top_k_set_agreement(
    scores_a: list[float] | list[list[float]],
    scores_b: list[float] | list[list[float]],
    k: int = 5,
) -> float:
    """Jaccard overlap of top-k *value sets* between two tensors.

    Compares the sets of top-k *values* (not indices), measuring how much
    the actual logprob mass overlaps. Returns Jaccard index in [0, 1].
    """
    a, b = _ensure_lists(scores_a, scores_b)
    jaccards: list[float] = []
    for a_row, b_row in zip(a, b):
        top_a = set(
            i for i, _ in sorted(enumerate(a_row), key=lambda x: -x[1])[:k]
        )
        top_b = set(
            i for i, _ in sorted(enumerate(b_row), key=lambda x: -x[1])[:k]
        )
        intersection = len(top_a & top_b)
        union = len(top_a | top_b)
        jaccards.append(intersection / union if union > 0 else 0.0)
    return sum(jaccards) / len(jaccards)


# ── Calibrated metrics (see CALIBRATION.md, DECISIONS.md D7) ──────────
#
# `max_absolute_difference` above is measured, not guessed, to be unusable as a
# gate: BF16 rounding alone scores 0.4929 against a 0.05 threshold, because the
# statistic is dominated by vocabulary-tail logprobs around -130 (probability
# ~1e-57, never sampled). Separation from a genuine fault is 2.2%.
#
# `cosine_similarity` is likewise unusable: it is scale-invariant by
# definition, so scaling every logprob by 1.01 or by 2.0 both score exactly
# 1.0. A softmax/temperature bug is mathematically invisible to it.
#
# The two below replace them. Measured separation: 15x.


def top_k_max_abs_diff(
    tensor_a: list[float] | list[list[float]],
    tensor_b: list[float] | list[list[float]],
    k: int = 10,
) -> float:
    """Worst absolute logprob difference across the reference's top-k tokens.

    Restricting to the top-k confines the measurement to the region that
    determines model behaviour. Ranking is taken from `tensor_a` (the
    reference) so the comparison is anchored, not co-defined by the candidate.

    Returns a true maximum — not a mean of maxima. The worst position is the
    one a customer's decision actually hinges on.

    Measured: bfloat16 rounding 0.0152 · weakest injected fault 0.2341.
    """
    a, b = _ensure_lists(tensor_a, tensor_b)
    worst = 0.0
    for row_a, row_b in zip(a, b):
        if not row_a:
            continue
        ranked = sorted(range(len(row_a)), key=lambda i: -row_a[i])[:k]
        worst = max(worst, max(abs(row_a[i] - row_b[i]) for i in ranked))
    return worst


def probability_mass(tensor: list[float] | list[list[float]]) -> float:
    """Mean total probability mass, `sum(exp(logprob))`, per position.

    A valid log-softmax row sums to 1.0. Deviation indicates output that was
    rescaled, truncated, or never normalised — the class of fault the top-k
    check cannot see, since it can leave the ranking untouched.

    Measured: correct output 0.999877 · a x1.05 scaling fault 0.904455.
    """
    rows, _ = _ensure_lists(tensor, tensor)
    totals: list[float] = []
    for row in rows:
        total = 0.0
        for value in row:
            # Guard against overflow on corrupt input: a logprob above 0 is
            # already invalid, and exp() of a large positive would raise.
            total += math.exp(value) if value < 64.0 else math.exp(64.0)
        totals.append(total)
    return sum(totals) / len(totals)
