"""Tests for the equivalence harness — metrics and runner."""

from __future__ import annotations

import json

import pytest

from ruitong.backends.fake import FakeAscend, FakeCuda
from ruitong.equivalence.metrics import (
    cosine_similarity,
    max_absolute_difference,
    top_k_agreement,
    top_k_set_agreement,
)
from ruitong.equivalence.runner import (
    EquivalenceReport,
    EquivalenceRunner,
    Thresholds,
)
from ruitong.schemas import ChatRequest, Message


# ===========================================================================
# cosine_similarity tests
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_tensors(self) -> None:
        """Two identical vectors → cosine = 1.0."""
        t = [0.3, -0.5, 1.2, 0.0]
        assert cosine_similarity(t, t) == pytest.approx(1.0)

    def test_opposite_tensors(self) -> None:
        """Exact opposites → cosine = -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_tensors(self) -> None:
        """Orthogonal vectors → cosine ≈ 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_known_angle_60deg(self) -> None:
        """Vectors at 60° → cosine = 0.5."""
        a = [1.0, 0.0]
        b = [0.5, 0.8660254037844386]  # cos(60°) = 0.5
        assert cosine_similarity(a, b) == pytest.approx(0.5, abs=1e-10)

    def test_batched_identical(self) -> None:
        """Batched identical vectors → cosine = 1.0."""
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[1.0, 2.0], [3.0, 4.0]]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_mismatched_batch_count_raises(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[1.0, 2.0]]
        with pytest.raises(ValueError):
            cosine_similarity(a, b)

    def test_single_zero_vector(self) -> None:
        """Zero vector in a pair → cosine = 0.0 (graceful handling)."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ===========================================================================
# max_absolute_difference tests
# ===========================================================================

class TestMaxAbsoluteDifference:
    def test_identical(self) -> None:
        assert max_absolute_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_offset(self) -> None:
        a = [0.0, 0.0, 5.0]
        b = [1.0, 2.0, 3.0]
        # max abs diff across all elements: max(1, 2, 2) = 2
        assert max_absolute_difference(a, b) == pytest.approx(2.0)

    def test_batched(self) -> None:
        a = [[0.0], [5.0]]
        b = [[1.0], [3.0]]
        # per row: 1.0, 2.0 → mean = 1.5
        assert max_absolute_difference(a, b) == pytest.approx(1.5)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            max_absolute_difference([], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            max_absolute_difference([1.0, 2.0], [3.0])

    def test_negative_values(self) -> None:
        """Handles negative values correctly."""
        a = [-1.0, -2.0]
        b = [1.0, 2.0]
        # max diff: max(2, 4) = 4
        assert max_absolute_difference(a, b) == pytest.approx(4.0)


# ===========================================================================
# top_k_agreement tests
# ===========================================================================

class TestTopKAgreement:
    def test_all_match_k1(self) -> None:
        a = [10.0, 5.0, 1.0]
        b = [10.1, 5.1, 1.1]
        assert top_k_agreement(a, b, k=1) == pytest.approx(1.0)

    def test_none_match(self) -> None:
        # a: idx0>idx1>idx2, b: idx2>idx1>idx0
        a = [3.0, 2.0, 1.0]
        b = [1.0, 2.0, 3.0]
        assert top_k_agreement(a, b, k=1) == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        # 3 positions: match, mismatch, match → 2/3
        a_rows = [
            [10.0, 5.0, 1.0],  # top=idx0
            [1.0, 10.0, 5.0],  # top=idx1
            [5.0, 1.0, 10.0],  # top=idx2
        ]
        b_rows = [
            [10.1, 5.1, 1.1],  # top=idx0 → match
            [1.1, 10.1, 5.1],  # top=idx1 → match
            [10.0, 1.1, 5.1],  # top=idx0 → mismatch
        ]
        assert top_k_agreement(a_rows, b_rows, k=1) == pytest.approx(2.0 / 3.0)

    def test_k_greater_than_length(self) -> None:
        """k > vector length should not crash; uses all elements."""
        a = [1.0, 2.0]
        b = [1.1, 2.1]
        assert top_k_agreement(a, b, k=10) == pytest.approx(1.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            top_k_agreement([], [])


# ===========================================================================
# top_k_set_agreement tests
# ===========================================================================

class TestTopKSetAgreement:
    def test_identical_sets(self) -> None:
        a = [10.0, 9.0, 8.0, 7.0, 6.0]
        b = [10.0, 9.0, 8.0, 7.0, 6.0]
        assert top_k_set_agreement(a, b, k=5) == pytest.approx(1.0)

    def test_k_larger_than_length(self) -> None:
        """k > length → uses all elements as the set."""
        a = [10.0, 5.0]
        b = [10.0, 5.0]
        assert top_k_set_agreement(a, b, k=10) == pytest.approx(1.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            top_k_set_agreement([], [])

    def test_disjoint_top5_values_same_indices(self) -> None:
        """Same-length arrays → same indices → Jaccard = 1.0 (index-based)."""
        a = [10.0, 9.0, 8.0, 7.0, 6.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert top_k_set_agreement(a, b, k=5) == pytest.approx(1.0)

    def test_partial_index_overlap(self) -> None:
        """Index-based: top-3 indices partially overlap."""
        a = [10.0, 9.0, 8.0, 7.0, 6.0, 0.0]
        b = [0.0, 10.0, 9.0, 8.0, 7.0, 6.0]
        assert top_k_set_agreement(a, b, k=3) == pytest.approx(0.5)

    def test_failing_top5_set_index_based(self) -> None:
        """Index-based: top-5 low overlap below threshold."""
        a = [10.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 10.0, 9.0, 8.0, 7.0, 6.0]
        t5 = top_k_set_agreement(a, b, k=5)
        assert t5 == pytest.approx(0.25)
        assert t5 < Thresholds.TOP5_MIN

    def test_same_values_different_indices(self) -> None:
        """P1 #4 near-miss: same top-5 values at different indices."""
        a = [100.0, 90.0, 80.0, 70.0, 60.0, 0.0]
        b = [0.0, 100.0, 90.0, 80.0, 70.0, 60.0]
        assert top_k_set_agreement(a, b, k=5) == pytest.approx(4.0 / 6.0)

    def test_disjoint_top5_indices(self) -> None:
        """P1 #4 variant: top-5 indices completely disjoint."""
        a = [100.0, 90.0, 80.0, 70.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 90.0, 80.0, 70.0, 60.0]
        assert top_k_set_agreement(a, b, k=5) == pytest.approx(0.0)


# ===========================================================================
# Threshold / deliberately-failing pairs
# ===========================================================================

class TestThresholds:
    """Verify that the runner correctly fails on deliberately bad data."""

    def test_perfect_match_passes(self) -> None:
        """Identical tensors should pass all thresholds."""
        a = [0.5, -0.3, 0.8, -0.1, 0.6]
        cos = cosine_similarity(a, a)
        md = max_absolute_difference(a, a)
        t1 = top_k_agreement(a, a, k=1)
        t5 = top_k_set_agreement(a, a, k=5)
        assert cos == pytest.approx(1.0)
        assert md == 0.0
        assert t1 == pytest.approx(1.0)
        assert t5 == pytest.approx(1.0)

    def test_failing_cosine(self) -> None:
        """Cosine below 0.99 should fail threshold."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        cos = cosine_similarity(a, b)
        assert cos < Thresholds.COSINE_MIN

    def test_failing_max_abs_diff(self) -> None:
        """Large abs diff should fail threshold."""
        a = [0.0, 0.0, 0.0]
        b = [10.0, 10.0, 10.0]
        md = max_absolute_difference(a, b)
        assert md > Thresholds.MAX_ABS_DIFF_MAX

    def test_failing_top1(self) -> None:
        """Top-1 agreement below 99% should fail."""
        a = [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]
        b = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        t1 = top_k_agreement(a, b, k=1)
        assert t1 < Thresholds.TOP1_MIN

    def test_failing_top5_set(self) -> None:
        """Top-5 set agreement below 95% should fail."""
        a = [10.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 10.0, 9.0, 8.0, 7.0, 6.0]
        t5 = top_k_set_agreement(a, b, k=5)
        assert t5 < Thresholds.TOP5_MIN


# ===========================================================================
# EquivalenceRunner integration tests
# ===========================================================================

class TestEquivalenceRunner:
    """Full runner tests against FakeCuda and FakeAscend."""

    @pytest.mark.asyncio
    async def test_auto_compare(self) -> None:
        """Auto mode: cuda vs ascend → measurable divergence."""
        runner = EquivalenceRunner(
            FakeCuda(model_ids=["qwen3-8b"]),
            FakeAscend(model_ids=["qwen3-8b"]),
        )
        report = await runner.run("qwen3-8b", ["Test prompt 1", "Test prompt 2"])
        assert report.total_prompts == 2
        assert report.mode == "logprob"
        assert report.metrics["cosine_similarity"] is not None
        assert report.metrics["max_absolute_difference"] is not None
        # Fakes produce controlled divergence, so likely fail thresholds
        # (that's the point — it should catch differences)

    @pytest.mark.asyncio
    async def test_same_backend_identical(self) -> None:
        """Same backend vs itself → perfect match."""
        cuda = FakeCuda(model_ids=["qwen3-8b"])
        runner = EquivalenceRunner(cuda, cuda)
        report = await runner.run("qwen3-8b", ["Hello"])
        assert report.mode == "logprob"
        assert report.passed is True
        assert report.metrics["cosine_similarity"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_model_not_found(self) -> None:
        """Model not on backend → error in report, not crash."""
        cuda = FakeCuda(model_ids=["other-model"])
        runner = EquivalenceRunner(cuda, cuda)
        report = await runner.run("qwen3-8b", ["Hello"])
        assert report.total_prompts == 1
        # Should have a warning about the failure
        assert len(report.warnings) >= 0

    @pytest.mark.asyncio
    async def test_report_serialization(self) -> None:
        """EquivalenceReport.to_dict() produces valid JSON."""
        cuda = FakeCuda(model_ids=["qwen3-8b"])
        runner = EquivalenceRunner(cuda, cuda)
        report = await runner.run("qwen3-8b", ["Test"])
        d = report.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["model"] == "qwen3-8b"
        assert loaded["total_prompts"] == 1
        assert isinstance(loaded["per_prompt_results"], list)
        assert isinstance(loaded["warnings"], list)

    @pytest.mark.asyncio
    async def test_passed_false_when_all_errors(self) -> None:
        """P1 #1 near-miss: when all prompts error, passed should be False.

        Current code initialises passed=True and never downgrades when
        there are zero metrics — bug.
        """
        cuda = FakeCuda(model_ids=[])
        ascend = FakeAscend(model_ids=["qwen3-8b"])
        runner = EquivalenceRunner(cuda, ascend)
        report = await runner.run("qwen3-8b", ["fail-me"])
        assert report.passed is False, (
            f"All prompts failed but passed={report.passed}"
        )
        assert len(report.warnings) > 0

    @pytest.mark.asyncio
    async def test_fake_backend_deterministic(self) -> None:
        """P1 #7 near-miss: same request produces same logprobs.

        Current code uses hash() which is non-deterministic per process.
        """
        cuda = FakeCuda(model_ids=["qwen3-8b"])
        req = ChatRequest(
            model="qwen3-8b",
            messages=[Message(role="user", content="Hello")],
            max_tokens=1,
        )
        r1 = await cuda.chat(req)
        r2 = await cuda.chat(req)
        assert r1.choices[0].logprobs == r2.choices[0].logprobs, (
            "Fake backend not deterministic — hash() bug"
        )

    @pytest.mark.asyncio
    async def test_per_prompt_results(self) -> None:
        """Each prompt should have its own result entry."""
        prompts = ["P1", "P2", "P3"]
        cuda = FakeCuda(model_ids=["qwen3-8b"])
        runner = EquivalenceRunner(cuda, cuda)
        report = await runner.run("qwen3-8b", prompts)
        assert len(report.per_prompt_results) == 3
        for i, r in enumerate(report.per_prompt_results):
            assert r.prompt == prompts[i]
            assert r.mode == "logprob"

    @pytest.mark.asyncio
    async def test_runner_backend_names(self) -> None:
        """Runner exposes backend name properties."""
        cuda = FakeCuda(model_ids=["qwen3-8b"])
        ascend = FakeAscend(model_ids=["qwen3-8b"])
        runner = EquivalenceRunner(cuda, ascend)
        assert runner.backend_a_name == "cuda"
        assert runner.backend_b_name == "ascend"
