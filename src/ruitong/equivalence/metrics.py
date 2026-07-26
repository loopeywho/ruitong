"""Metric computation for equivalence comparison.

All functions accept list[float] (single) or list[list[float]] (batched).
When batched, metrics are computed per-position and averaged.
"""

from __future__ import annotations

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
