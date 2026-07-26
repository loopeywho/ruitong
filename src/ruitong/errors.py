"""Ruitong domain error types."""

from __future__ import annotations

from typing import Any


class RuitongError(Exception):
    """Base exception for all Ruitong domain errors."""

    pass


class BackendUnavailable(RuitongError):
    """The requested backend is down or unreachable."""

    def __init__(self, backend: str, message: str | None = None) -> None:
        self.backend = backend
        self.message = message or f"Backend '{backend}' is unavailable"
        super().__init__(self.message)


class ModelNotFound(RuitongError):
    """The requested model is not available on the specified backend."""

    def __init__(self, model: str, backend: str) -> None:
        self.model = model
        self.backend = backend
        self.message = f"Model '{model}' not found on backend '{backend}'"
        super().__init__(self.message)


class BackendError(RuitongError):
    """Generic backend failure — wraps a transport exception."""

    def __init__(self, backend: str, cause: Exception | None = None, message: str | None = None) -> None:
        self.backend = backend
        self.cause = cause
        self.message = message or f"Backend '{backend}' error: {cause}"
        super().__init__(self.message)
