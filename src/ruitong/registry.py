"""Backend registry — maps name → Backend and resolves routing decisions."""

from __future__ import annotations

import asyncio

from .backends.base import Backend
from .config import BridgeConfig
from .errors import BackendUnavailable, ModelNotFound
from .schemas import HealthStatus, ModelInfo


class BackendRegistry:
    """Holds registered backends and resolves routing."""

    def __init__(self, config: BridgeConfig | None = None) -> None:
        self._backends: dict[str, Backend] = {}
        self.config = config or BridgeConfig()

    def register(self, name: str, backend: Backend) -> None:
        """Register a backend under *name*. Overwrites any existing."""
        self._backends[name] = backend

    def get(self, name: str) -> Backend:
        """Return the backend registered under *name*."""
        if name not in self._backends:
            raise BackendUnavailable(
                name, message=f"No backend registered as '{name}'"
            )
        return self._backends[name]

    def list_backends(self) -> list[str]:
        """Return all registered backend names."""
        return list(self._backends.keys())

    async def health(self) -> list[HealthStatus]:
        """Return health status for every registered backend."""
        tasks = [b.health() for b in self._backends.values()]
        return await asyncio.gather(*tasks)

    async def list_models(self) -> list[ModelInfo]:
        """Return model info from all registered backends."""
        tasks = [b.list_models() for b in self._backends.values()]
        results = await asyncio.gather(*tasks)
        models: list[ModelInfo] = []
        for result in results:
            models.extend(result)
        return models

    async def resolve(self, model: str, preferred_backend: str = "auto") -> Backend:
        """Resolve which backend to use for *model*.

        Policy:
        1. Explicit: use the named backend. Raise BackendUnavailable if not
           registered, ModelNotFound if it doesn't serve the model.
        2. Auto: iterate backends in *auto_backend_priority* order. The first
           healthy backend that serves the model wins. Raise
           BackendUnavailable if none are healthy, ModelNotFound if all
           healthy backends lack the model.
        """
        if preferred_backend == "auto":
            priority = self.config.auto_backend_priority
            candidates = [n for n in priority if n in self._backends]

            if not candidates:
                raise BackendUnavailable(
                    "", message="No backends registered for auto-resolution"
                )

            # Collect health for all candidates
            health_map: dict[str, HealthStatus] = {}
            for name in candidates:
                health_map[name] = await self._backends[name].health()

            # Find first healthy backend that serves the model
            for name in candidates:
                hs = health_map[name]
                if not hs.healthy:
                    continue
                models = await self._backends[name].list_models()
                if any(m.id == model for m in models):
                    return self._backends[name]

            # Healthy backends exist but none serve the model
            healthy_names = [n for n in candidates if health_map[n].healthy]
            if healthy_names:
                raise ModelNotFound(model, healthy_names[0])

            # No healthy backends
            raise BackendUnavailable(
                candidates[0], message="All backends unhealthy"
            )

        # Explicit backend
        b = self.get(preferred_backend)
        models = await b.list_models()
        if not any(m.id == model for m in models):
            raise ModelNotFound(model, preferred_backend)
        return b
