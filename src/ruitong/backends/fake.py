"""Deterministic fake backends for Phase 2 testing.

Provides FakeCuda and FakeAscend that satisfy the Backend Protocol.
Each can be toggled unhealthy and configured which models it serves.
"""

from __future__ import annotations

import hashlib
from typing import AsyncIterator

from ..errors import ModelNotFound
from ..schemas import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Choice,
    Delta,
    DeltaChoice,
    HealthStatus,
    Message,
    ModelInfo,
    Usage,
)
from .base import Backend


# Deterministic logprob generation helpers
def _fake_logprobs(seed: str, count: int = 3) -> tuple[list[float], list[dict[int, float]]]:
    """Generate deterministic pseudo-logprobs from a seed string.

    Returns (logprobs, top_logprobs) where:
    - logprobs: list of float log-likelihoods for top tokens
    - top_logprobs: list of {token_id: logprob} dicts (5 tokens per position)
    """
    raw = hashlib.sha256(seed.encode()).hexdigest()
    raw_len = len(raw)  # 64
    logprobs: list[float] = []
    top_logprobs: list[dict[int, float]] = []
    # Single linear counter avoids slice-wrap bugs
    idx = 0
    for i in range(count):
        chunk = raw[idx:idx + 12]
        idx += 12
        value = int(chunk, 16) % 10000 / 1000.0 - 5.0  # range ~[-5, 5]
        logprobs.append(round(value, 6))
        top5: dict[int, float] = {}
        for j in range(5):
            pair = raw[idx:idx + 6]
            idx += 6
            if idx >= raw_len:
                idx = (idx) % raw_len
                pair = raw[idx:idx + 6]
                idx += 6
            tok_id = (int(pair, 16) % 50000) + 1000
            lp = round(value - j * 0.3, 6)
            top5[tok_id] = lp
        top_logprobs.append(top5)
    return logprobs, top_logprobs


# Backend-specific multiplier for controlled divergence
_CUDA_OFFSET = 0.001
_ASCEND_OFFSET = 0.012


class FakeCuda(Backend):
    """Deterministic fake CUDA (vLLM) backend."""

    name: str = "cuda"

    def __init__(self, *, model_ids: list[str] | None = None) -> None:
        self._healthy: bool = True
        self._error: Exception | None = None
        id_list = model_ids or ["qwen2.5-7b-instruct", "llama3-8b"]
        self._models: list[ModelInfo] = [
            ModelInfo(id=mid, backend=self.name, max_model_len=4096) for mid in id_list
        ]

    def _set_unhealthy(self, value: bool) -> None:
        """Set *value=True* to make the backend unhealthy."""
        self._healthy = not value

    def _set_error(self, exc: Exception | None) -> None:
        self._error = exc

    async def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._healthy,
            backend="cuda",
            models_served=len(self._models),
            uptime_seconds=3600.0,
        )

    async def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        if self._error is not None:
            raise self._error
        if not any(m.id == req.model for m in self._models):
            raise ModelNotFound(req.model, "cuda")
        logprobs, top_logprobs = _fake_logprobs(
            f"{req.model}-cuda", count=3
        )
        # Apply small deterministic CUDA offset so metrics can measure
        logprobs = [round(lp - _CUDA_OFFSET, 6) for lp in logprobs]
        return ChatResponse(
            id="chatcmpl-cuda-1",
            model=req.model,
            backend="cuda",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="I am CUDA"),
                    finish_reason="stop",
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
            created=1700000000,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        if self._error is not None:
            raise self._error
        if not any(m.id == req.model for m in self._models):
            raise ModelNotFound(req.model, "cuda")
        yield ChatChunk(
            id="chatcmpl-cuda-1",
            model=req.model,
            backend="cuda",
            choices=[
                DeltaChoice(
                    index=0,
                    delta=Delta(role="assistant", content="I am CUDA"),
                    finish_reason="stop",
                )
            ],
        )


class FakeAscend(Backend):
    """Deterministic fake Ascend (CANN) backend."""

    name: str = "ascend"

    def __init__(self, *, model_ids: list[str] | None = None) -> None:
        self._healthy: bool = True
        self._error: Exception | None = None
        id_list = model_ids or ["qwen2.5-7b-instruct", "qwen3-8b"]
        self._models: list[ModelInfo] = [
            ModelInfo(id=mid, backend=self.name, max_model_len=4096) for mid in id_list
        ]

    def _set_unhealthy(self, value: bool) -> None:
        """Set *value=True* to make the backend unhealthy."""
        self._healthy = not value

    def _set_error(self, exc: Exception | None) -> None:
        self._error = exc

    async def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._healthy,
            backend="ascend",
            models_served=len(self._models),
            uptime_seconds=1800.0,
        )

    async def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        if self._error is not None:
            raise self._error
        if not any(m.id == req.model for m in self._models):
            raise ModelNotFound(req.model, "ascend")
        logprobs, top_logprobs = _fake_logprobs(
            f"{req.model}-ascend", count=3
        )
        # Apply larger deterministic Ascend offset so metrics can measure divergence
        logprobs = [round(lp - _ASCEND_OFFSET, 6) for lp in logprobs]
        return ChatResponse(
            id="chatcmpl-ascend-1",
            model=req.model,
            backend="ascend",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="I am Ascend"),
                    finish_reason="stop",
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                )
            ],
            usage=Usage(prompt_tokens=12, completion_tokens=9, total_tokens=21),
            created=1700000001,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        if self._error is not None:
            raise self._error
        if not any(m.id == req.model for m in self._models):
            raise ModelNotFound(req.model, "ascend")
        yield ChatChunk(
            id="chatcmpl-ascend-1",
            model=req.model,
            backend="ascend",
            choices=[
                DeltaChoice(
                    index=0,
                    delta=Delta(role="assistant", content="I am Ascend"),
                    finish_reason="stop",
                )
            ],
        )
