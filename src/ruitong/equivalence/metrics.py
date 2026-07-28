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


# ── Token-identity agreement (the metric PLAN.md actually specified) ──
#
# `top_k_agreement` / `top_k_set_agreement` above rank *positions* and compare
# indices. They never see a token. Measured consequences:
#
#   same token every position, confidences reordered  -> reports 0.0 (total
#       disagreement) on output that is in fact identical
#   completely different tokens, same confidences     -> reports 1.0 (perfect
#       agreement) on output that shares nothing
#
# PLAN.md specifies "top-1 agreement" and "top-5 set agreement" in their
# standard meaning: does the argmax *token* match, and do the top-5 *token*
# sets overlap. That is what these two compute. They require token identity,
# which `ChoiceLogprobs.top_k_tokens()` now carries off the wire.


def _reject_flat_token_rows(rows: object, argument: str) -> None:
    """Reject `list[str]` where `list[list[str]]` is required.

    The two shapes are indistinguishable by duck typing — a `str` is itself a
    sequence of `str` — so `rows[0][0]` silently yields a *character* and the
    metric returns a plausible wrong number instead of failing. Caught when a
    self-comparison scored 0.984 rather than 1.0; the caller had passed
    already-extracted rank-0 tokens.
    """
    if isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], str):
        raise TypeError(
            f"{argument} must be list[list[str]] (one row of top-k tokens per "
            f"position), got list[str]. Pass the full top-k rows, not the "
            f"rank-0 tokens — indexing a str yields a character and the "
            f"comparison silently degrades to first-letter matching."
        )


def top1_token_agreement(
    tokens_a: list[list[str]], tokens_b: list[list[str]]
) -> float:
    """Fraction of positions where both backends' most likely TOKEN matches."""
    if not tokens_a or not tokens_b:
        raise ValueError("Inputs must not be empty")
    if len(tokens_a) != len(tokens_b):
        raise ValueError(f"Length mismatch: {len(tokens_a)} vs {len(tokens_b)}")
    _reject_flat_token_rows(tokens_a, "tokens_a")
    _reject_flat_token_rows(tokens_b, "tokens_b")

    matches = 0
    for row_a, row_b in zip(tokens_a, tokens_b):
        if row_a and row_b and row_a[0] == row_b[0]:
            matches += 1
    return matches / len(tokens_a)


def topk_token_set_agreement(
    tokens_a: list[list[str]], tokens_b: list[list[str]], k: int = 5
) -> float:
    """Mean Jaccard overlap of the top-k TOKEN sets, per position.

    Robust to small confidence differences reordering tokens within the top-k —
    which is exactly what differing kernels produce — while still detecting a
    genuinely different candidate set.
    """
    if not tokens_a or not tokens_b:
        raise ValueError("Inputs must not be empty")
    if len(tokens_a) != len(tokens_b):
        raise ValueError(f"Length mismatch: {len(tokens_a)} vs {len(tokens_b)}")
    _reject_flat_token_rows(tokens_a, "tokens_a")
    _reject_flat_token_rows(tokens_b, "tokens_b")

    scores: list[float] = []
    for row_a, row_b in zip(tokens_a, tokens_b):
        set_a, set_b = set(row_a[:k]), set(row_b[:k])
        union = len(set_a | set_b)
        scores.append(len(set_a & set_b) / union if union else 0.0)
    return sum(scores) / len(scores)


def token_matched_prob_diff(
    tokens_a: list[list[str]],
    logprobs_a: list[list[float]],
    tokens_b: list[list[str]],
    logprobs_b: list[list[float]],
    k: int = 10,
) -> float:
    """Worst probability shift for any of the reference's top-k tokens.

    This replaces `top_k_max_abs_diff` as the primary gate. Two defects in
    that metric were exposed by real hardware (Qwen3-8B on an NVIDIA A40,
    2026-07-27) and neither was visible in synthetic calibration:

    1. **It compares by rank, so it compares different tokens.** In the tail
       of the top-k many tokens are near-tied, so a negligible perturbation
       swaps their order. Rank 3 then holds `'高'` on one side and `' like'`
       on the other, and the metric reports the gap between two unrelated
       tokens as disagreement. Matching by token identity removes this
       entirely.

    2. **Log space gives equal weight to irrelevant tokens.** For an 8B model
       the top-10 spans logprob 0 down to about -29. A 0.5 shift at -26 is a
       probability change of 3e-12 — it cannot alter any behaviour — yet it
       scores identically to a 0.5 shift at the argmax, which flips output.
       Comparing probabilities weights each token by how much it can matter.

    Together these produced a same-server noise reading of 0.5 to 28.6
    against a threshold of 0.05: the old gate rejected the reference model
    compared with *itself*.

    A reference token absent from the candidate's top-k is treated as
    probability zero. That is the correct reading — the candidate ranked it
    outside k — and it is self-scaling: a dropped token at p=1e-12 registers
    1e-12, while a dropped token at p=0.3 registers 0.3.

    **Token strings are not unique within a row.** The wire format carries the
    *decoded string*, and many distinct token IDs decode to the same one — in
    the captured A40 corpus a single position held nine separate ids all
    decoding to `""`. Building a plain `{token: logprob}` dict silently keeps
    only the last, so rank 1 gets compared against rank 17 and an identical
    tensor scores 0.67 against itself. Repeated strings are therefore paired
    by order of appearance: the n-th `""` on one side matches the n-th `""`
    on the other.

    Returns a value in [0, 1] directly readable as "no token's probability
    moved by more than this".
    """
    worst = 0.0
    for row_tokens_a, row_logs_a, row_tokens_b, row_logs_b in zip(
        tokens_a, logprobs_a, tokens_b, logprobs_b
    ):
        if not row_tokens_a:
            continue

        # token -> logprobs in the order the candidate listed them
        occurrences_b: dict[str, list[float]] = {}
        for token, logprob in zip(row_tokens_b, row_logs_b):
            occurrences_b.setdefault(token, []).append(logprob)

        seen: dict[str, int] = {}
        limit = min(len(row_tokens_a), len(row_logs_a))
        for index in range(limit):
            token = row_tokens_a[index]
            nth = seen.get(token, 0)
            seen[token] = nth + 1
            if index >= k:
                continue  # still counted above, so pairing stays aligned
            prob_a = math.exp(min(row_logs_a[index], 0.0))
            bucket = occurrences_b.get(token, ())
            prob_b = (
                math.exp(min(bucket[nth], 0.0)) if nth < len(bucket) else 0.0
            )
            worst = max(worst, abs(prob_a - prob_b))
    return worst


def count_non_finite(tensor: list[list[float]]) -> int:
    """Count logprob entries that are not finite (-inf, +inf, NaN).

    A correct log-softmax never emits these. Their presence means the *server*
    is faulty, not that the port is wrong — and the two demand opposite
    responses, which is why the CLI separates exit 1 (gate failed) from exit 2
    (could not run).

    This is not hypothetical. `vllm-ascend` 0.9.1 serving Qwen3-8B emits
    excessive `-inf` logprobs where the same model on GPU does not
    (vllm-ascend issue #2934), and vLLM on ROCm returns `-9999` sentinels
    (vllm-project/vllm#19305).

    Measured against the current gate, `-inf` is *not* silent: `exp(-inf)` is
    0.0, so a corrupted entry shows up as a large probability difference
    (0.0498 for a tail token, 0.951 on the argmax, against a 0.0022
    threshold). The danger is therefore the opposite of the one first
    supposed — not a false PASS that certifies a broken port, but a false
    FAIL that condemns a *correct* port for an upstream sampler defect.
    Counting them lets the runner say "we could not tell" instead.
    """
    return sum(
        1 for row in tensor for value in row if not math.isfinite(value)
    )


def count_degenerate_token_rows(tokens: list[list[str]]) -> int:
    """Count top-k rows whose entries all carry the SAME token identity.

    A correct top-k list names k *distinct* tokens. Duplicate decoded strings
    are normal and common — many token ids decode to the same text, and real
    NVIDIA rows in this repo's corpora reach 18 duplicate slots out of 20
    (two unique strings). But a row with exactly ONE unique value across k>1
    entries is not a ranking; it is a serialisation defect.

    Real and specific: `vllm-ascend` issue #7218 (open, 0.16.0rc2 on 910B3/B4)
    reports every `top_logprobs` entry repeating the selected token's id --
    "top_logprobs的id都和被选择的token一致，看不到其余top token". The logprob
    VALUES in that report are correct and monotonic; only token identity is
    lost.

    Why this needs its own guard: measured against this gate, an Ascend port
    whose logprob values are *bit-identical* to NVIDIA scores 0.925 on
    token_matched_prob_diff and 0.000 on top1_agreement -- a hard FAIL on two
    of three gate metrics -- because every token lookup misses. A numerically
    perfect port would be condemned for an upstream serialisation bug. That is
    the D8 distinction (exit 1 "the port is broken" vs exit 2 "we could not
    tell") and it must resolve to the second.

    Verified not to fire on correct output: 0 of 5,677 rows across all three
    captured NVIDIA corpora.
    """
    return sum(
        1 for row in tokens if len(row) >= 2 and len(set(row)) == 1
    )
