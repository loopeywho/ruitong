"""Router — resolves requests to backends."""

from __future__ import annotations

from typing import AsyncIterator

from .config import BridgeConfig
from .errors import BackendError, RuitongError
from .registry import BackendRegistry
from .schemas import ChatChunk, ChatRequest, ChatResponse, HealthStatus, ModelInfo


class Router:
    """Routes chat requests to backends using a registry."""

    def __init__(self, registry: BackendRegistry, config: BridgeConfig) -> None:
        self.registry = registry
        self.config = config

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Resolve backend and process a non-streaming chat request."""
        backend = await self.registry.resolve(req.model, req.backend)
        try:
            return await backend.chat(req)
        except RuitongError:
            raise
        except Exception as exc:
            raise BackendError(backend.name, cause=exc) from exc

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Resolve backend and yield streaming chunks."""
        backend = await self.registry.resolve(req.model, req.backend)
        try:
            async for chunk in backend.stream(req):
                yield chunk
        except RuitongError:
            raise
        except Exception as exc:
            raise BackendError(backend.name, cause=exc) from exc

    async def health(self) -> list[HealthStatus]:
        """Return health status for all registered backends."""
        return await self.registry.health()

    async def list_models(self) -> list[ModelInfo]:
        """Return model info from all registered backends."""
        return await self.registry.list_models()
