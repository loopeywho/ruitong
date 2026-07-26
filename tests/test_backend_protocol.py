"""Tests that the Backend protocol is structural (duck-typed), not inheritance-based."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from ruitong.backends.base import Backend
from ruitong.schemas import (
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


class TestStructuralTyping:
    """Protocol is satisfied by structural compatibility, not subclassing."""

    def test_no_inheritance_required(self) -> None:
        """A class that does NOT inherit from Backend but has the right shape works."""

        class FakeImpl:
            name: str = "fake"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="fake", models_served=0)

            async def list_models(self) -> list[ModelInfo]:
                return []

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="test",
                    model=req.model,
                    backend="fake",
                    choices=[Choice(index=0, message=Message(role="assistant", content="ok"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                yield ChatChunk(
                    id="test",
                    model=req.model,
                    backend="fake",
                    choices=[],
                )

        # This assignment proves structural typing — FakeImpl has no inheritance
        # from Backend but has all required members.
        backend: Backend = FakeImpl()  # type: ignore[assignment]
        assert backend.name == "fake"

    def test_function_accepts_backend_protocol(self) -> None:
        """A function typed with Backend accepts any structurally compatible class."""

        def use_backend(b: Backend) -> str:
            return b.name

        class Minimal:
            name: str = "minimal"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="m", models_served=0)

            async def list_models(self) -> list[ModelInfo]:
                return []

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="x", model="m", backend="m",
                    choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                return
                yield

        result = use_backend(Minimal())
        assert result == "minimal"


class TestProtocolMethods:
    """Verify that method signatures match what the protocol expects."""

    @pytest.mark.asyncio
    async def test_health_returns_health_status(self) -> None:
        class Quick:
            name: str = "q"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="q", models_served=1)

            async def list_models(self) -> list[ModelInfo]:
                return []

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="x", model="m", backend="q",
                    choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                return
                yield

        b: Backend = Quick()
        status = await b.health()
        assert isinstance(status, HealthStatus)
        assert isinstance(status.healthy, bool)

    def test_name_attribute(self) -> None:
        class NB:
            name: str = "test"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="t", models_served=0)

            async def list_models(self) -> list[ModelInfo]:
                return []

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="x", model="m", backend="t",
                    choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                return
                yield

        b: Backend = NB()
        assert b.name == "test"

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_list(self) -> None:
        class ML:
            name: str = "ml"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="ml", models_served=0)

            async def list_models(self) -> list[ModelInfo]:
                return [ModelInfo(id="qwen", backend="ml", max_model_len=4096)]

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="x", model="m", backend="ml",
                    choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                return
                yield

        b: Backend = ML()
        models = await b.list_models()
        assert len(models) == 1
        assert isinstance(models[0], ModelInfo)
        assert isinstance(models[0].id, str)

    @pytest.mark.asyncio
    async def test_chat_returns_chat_response(self) -> None:
        class CR:
            name: str = "cr"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="cr", models_served=0)

            async def list_models(self) -> list[ModelInfo]:
                return []

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="x", model=req.model, backend="cr",
                    choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                return
                yield

        b: Backend = CR()
        req = ChatRequest(model="test", messages=[Message(role="user", content="hi")])
        resp = await b.chat(req)
        assert isinstance(resp, ChatResponse)
        assert isinstance(resp.id, str)
        assert isinstance(resp.backend, str)
        assert isinstance(resp.choices, list)

    @pytest.mark.asyncio
    async def test_stream_returns_chat_chunk_generator(self) -> None:
        class SG:
            name: str = "sg"

            async def health(self) -> HealthStatus:
                return HealthStatus(healthy=True, backend="sg", models_served=0)

            async def list_models(self) -> list[ModelInfo]:
                return []

            async def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(
                    id="x", model="m", backend="sg",
                    choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
                yield ChatChunk(
                    id="x", model="m", backend="sg",
                    choices=[DeltaChoice(index=0, delta=Delta(role="assistant", content="hi"), finish_reason=None)],
                )

        b: Backend = SG()
        req = ChatRequest(model="test", messages=[Message(role="user", content="hi")])
        chunks = [chunk async for chunk in b.stream(req)]
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatChunk)
        assert chunks[0].backend == "sg"
