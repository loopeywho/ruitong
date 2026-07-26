# Fable Audit: 瑞通 Phase 1 — Contracts

You are **Fable**, the quality assurance auditor. Review Phase 1 of the 瑞通 (Ruitong) CANN/CUDA Bridge API. The builder (Qwen) has completed this phase. Your job: find every bug, design flaw, security issue, or style violation before Phase 2 begins.

## Background

Ruitong is a middleware bridge that routes inference requests to either a CUDA (NVIDIA) or CANN (Huawei Ascend) backend, behind the firewall. Phase 1 establishes the contracts — schemas, Backend protocol, error types — with zero hardware/dependencies.

**Reference model (for test data):** Qwen2.5-7B-Instruct

## Files to Review

### 1. pyproject.toml (`ruitong-bridge/pyproject.toml`)

```toml
[project]
name = "ruitong-bridge"
version = "0.1.0"
description = "瑞通 (Ruitong) — CANN/CUDA Bridge API"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
    "uv>=0.4",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ruitong"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### 2. `src/ruitong/schemas.py`

```python
"""瑞通 (Ruitong) — shared schemas for CANN/CUDA Bridge API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel, frozen=True):
    role: Literal["system", "user", "assistant"] = Field(...)
    content: str = Field(...)


class Usage(BaseModel, frozen=True):
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class ChatRequest(BaseModel, frozen=True):
    model: str = Field(...)
    messages: list[Message] = Field(..., min_length=1)
    backend: Literal["cuda", "ascend", "auto"] = Field(default="auto")
    max_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = Field(default=None)


class Choice(BaseModel, frozen=True):
    index: int = Field(..., ge=0)
    message: Message = Field(...)
    finish_reason: str | None = Field(default=None)


class ChatResponse(BaseModel, frozen=True):
    id: str = Field(...)
    model: str = Field(...)
    backend: str = Field(...)
    choices: list[Choice] = Field(...)
    usage: Usage = Field(...)
    created: int | None = Field(default=None)


class Delta(BaseModel, frozen=True):
    role: str | None = Field(default=None)
    content: str | None = Field(default=None)


class DeltaChoice(BaseModel, frozen=True):
    index: int = Field(..., ge=0)
    delta: Delta = Field(...)
    finish_reason: str | None = Field(default=None)


class ChatChunk(BaseModel, frozen=True):
    id: str = Field(...)
    model: str = Field(...)
    backend: str = Field(...)
    choices: list[DeltaChoice] = Field(...)


class HealthStatus(BaseModel, frozen=True):
    healthy: bool = Field(...)
    backend: str = Field(...)
    models_served: int = Field(..., ge=0)
    uptime_seconds: float | None = Field(default=None)


class ModelInfo(BaseModel, frozen=True):
    id: str = Field(...)
    backend: str = Field(...)
    max_model_len: int | None = Field(default=None)
```

### 3. `src/ruitong/errors.py`

```python
"""瑞通 (Ruitong) — domain-error hierarchy."""

from __future__ import annotations


class RuitongError(Exception):
    """Base for all domain errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class BackendUnavailable(RuitongError):
    """Raised when a backend cannot be reached or is unhealthy."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(f"Backend '{backend_name}' is unavailable")
        self.backend_name = backend_name


class ModelNotFound(RuitongError):
    """Raised when a model is not available on the requested backend."""

    def __init__(self, model: str, backend_name: str) -> None:
        super().__init__(f"Model '{model}' not found on backend '{backend_name}'")
        self.model = model
        self.backend_name = backend_name


class BackendError(RuitongError):
    """Raised when a backend returns an unexpected error."""

    def __init__(self, backend_name: str, cause: Exception) -> None:
        super().__init__(f"Backend '{backend_name}' error: {cause}")
        self.backend_name = backend_name
        self.cause = cause
```

### 4. `src/ruitong/backends/base.py`

```python
"""Backend protocol — the contract every inference backend must satisfy."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from ruitong.schemas import ChatChunk, ChatRequest, ChatResponse, HealthStatus, ModelInfo


class Backend(Protocol):
    """Structural protocol for an inference backend.

    Implementations do NOT inherit from this class — they satisfy it
    structurally by providing all members with compatible signatures.
    """

    name: str

    async def health(self) -> HealthStatus: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def chat(self, req: ChatRequest) -> ChatResponse: ...
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...
```

### 5. `src/ruitong/config.py`

```python
"""Ruitong bridge configuration via environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BridgeConfig:
    """Runtime configuration for the bridge."""

    cuda_base_url: str = os.environ.get("RUITONG_CUDA_BASE_URL", "")
    ascend_base_url: str = os.environ.get("RUITONG_ASCEND_BASE_URL", "")
    auto_backend_priority: list[str] = field(
        default_factory=lambda: ["cuda", "ascend"]
    )
    default_max_tokens: int = 2048
    health_check_timeout: float = 5.0


config = BridgeConfig()
```

### 6. `src/ruitong/__init__.py`

```python
"""瑞通 (Ruitong) — CANN/CUDA Bridge API."""

__version__ = "0.1.0"
```

### 7. Tests

- `tests/test_schemas.py` — 294 lines, covers ChatRequest/Message/ChatResponse/ChatChunk/HealthStatus/ModelInfo/Usage round-trips, validation errors, defaults
- `tests/test_errors.py` — 86 lines, covers hierarchy, raising/catching, meaningful messages
- `tests/test_backend_protocol.py` — 232 lines, covers structural typing (no inheritance required), all 4 methods return correct types
- `tests/conftest.py` — likely empty/event_loop fixture

## What to Review

1. **Design correctness:** Do the schemas match OpenAI-compatible API shape with Ruitong extensions (`backend` field)? Is the Backend protocol complete? (generate, stream, count_tokens vs chat, stream)
2. **pydantic patterns:** Are frozen models correct for immutability? Are `Field` constraints appropriate? Any issues with `ChatChunk` being defined twice?
3. **Error hierarchy:** Does every domain error inherit from RuitongError? Are exception messages actionable?
4. **Config design:** Is dataclass+env-var loading appropriate for this phase? Any threading/singleton concerns?
5. **Test adequacy:** Are edge cases covered? (empty messages, null fields, backend literal validation, large sequences)
6. **Security:** Any info leaks in error messages? Any dangerous defaults?
7. **Structural typing:** Protocol vs ABC — is the duck-typing approach correct for this use case?

## Acceptance Criteria (from PLAN.md)

- ✅ Schemas round-trip through JSON deserialization/serialization
- ✅ Backend is a Protocol (structural typing, not abstract class)
- ✅ Tests cover schema validation (missing required fields, invalid backend literals, empty messages rejected)
- ✅ All tests pass
- ✅ Coverage ≥80% on domain modules

## Instructions

1. Read each file.
2. Record your findings in `QA_FINDINGS.md` at the project root: green-light or red-light each check, plus any issues found.
3. If blocking issues exist, describe exactly what to fix.
4. If green, say "Phase 1 PASS — proceed to Phase 2" and the builder can start `src/ruitong/backends/fake_cuda.py` and `src/ruitong/backends/fake_ascend.py`.

## Deliverable

Reply with your audit results. I will paste them into `QA_FINDINGS.md`.