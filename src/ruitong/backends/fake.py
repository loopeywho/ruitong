"""Deterministic fake backends for Phase 2 testing.

Provides FakeCuda and FakeAscend that satisfy the Backend Protocol.
Each can be toggled unhealthy and configured which models it serves.
"""

from __future__ import annotations

import hashlib
from typing import AsyncIterator

from ..errors import ModelNotFound
from ..schemas import (
    ChoiceLogprobs,
    LogprobEntry,
    TopLogprob,
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


def _fake_logprobs(seed: str, count: int = 3, offset: float = 0.0) -> ChoiceLogprobs:
    """Deterministic logprobs in the REAL OpenAI wire shape.

    This must mirror what an OpenAI-compatible server actually returns —
    `{"content": [{token, logprob, top_logprobs: [...]}]}` — not a convenient
    bare list. The previous version returned `list[float]`, which no real
    server emits, and that mismatch hid a validation failure that would have
    made every response from a rented GPU unparseable.

    Values are clamped to <= 0: a logprob is the log of a probability, so a
    positive one is not physically meaningful and would corrupt any
    probability-mass check computed from it.
    """
    raw = hashlib.sha256(seed.encode()).hexdigest()
    raw_len = len(raw)
    entries: list[LogprobEntry] = []
    idx = 0
    for _ in range(count):
        chunk = raw[idx : idx + 12]
        idx = (idx + 12) % raw_len
        value = -(int(chunk, 16) % 10000) / 2000.0 - offset  # ~[-5, 0]
        tops: list[TopLogprob] = []
        for rank in range(5):
            pair = raw[idx : idx + 6]
            idx = (idx + 6) % raw_len
            tok_id = (int(pair, 16) % 50000) + 1000
            tops.append(
                TopLogprob(
                    token=f"tok_{tok_id}",
                    logprob=round(value - rank * 0.3, 6),
                )
            )
        entries.append(
            LogprobEntry(
                token=tops[0].token,
                logprob=tops[0].logprob,
                top_logprobs=tops,
            )
        )
    return ChoiceLogprobs(content=entries)


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
        choice_logprobs = _fake_logprobs(
            f"{req.model}-cuda", count=3, offset=_CUDA_OFFSET
        )
        # Apply small deterministic CUDA offset so metrics can measure
        return ChatResponse(
            id="chatcmpl-cuda-1",
            model=req.model,
            backend="cuda",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="I am CUDA"),
                    finish_reason="stop",
                    logprobs=choice_logprobs,
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
        choice_logprobs = _fake_logprobs(
            f"{req.model}-ascend", count=3, offset=_ASCEND_OFFSET
        )
        # Apply larger deterministic Ascend offset so metrics can measure divergence
        return ChatResponse(
            id="chatcmpl-ascend-1",
            model=req.model,
            backend="ascend",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="I am Ascend"),
                    finish_reason="stop",
                    logprobs=choice_logprobs,
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
