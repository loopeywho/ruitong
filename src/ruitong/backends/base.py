"""Ruitong backend interface using structural typing (Protocol)."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from ..schemas import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    HealthStatus,
    ModelInfo,
)


class Backend(Protocol):
    """Protocol for backend implementations.

    This is a structural (duck-typed) protocol — any class with the
    matching method signatures satisfies it, regardless of inheritance.
    No ABC machinery is used.
    """

    name: str  # "cuda" | "ascend"

    async def health(self) -> HealthStatus:
        """Return the backend's health status."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Return the list of models served by this backend."""
        ...

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Process a chat request and return a non-streaming response."""
        ...

    # NOTE: deliberately `def`, not `async def`.
    #
    # An implementation is expected to be an async *generator*:
    #
    #     async def stream(self, req):
    #         yield chunk
    #
    # Calling such a function returns the async iterator directly, so callers
    # write `async for chunk in backend.stream(req)`. Declaring the protocol
    # member as `async def -> AsyncIterator[...]` would instead describe a
    # coroutine that must be awaited before iterating — an incompatible shape
    # that type checkers reject against generator implementations.
    def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield streaming chunks for a chat request."""
        ...
