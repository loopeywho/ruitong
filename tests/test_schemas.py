"""Tests for ruitong.schemas — round-trip, validation, and rejection."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

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


# ---------------------------------------------------------------------------
# ChatRequest — valid round-trip
# ---------------------------------------------------------------------------

class TestChatRequest:
    def test_valid_minimal(self) -> None:
        """Minimal valid request round-trips through JSON."""
        req = ChatRequest(
            model="qwen2.5-7b-instruct",
            messages=[Message(role="user", content="Hello")],
        )
        data = req.model_dump_json()
        loaded = ChatRequest.model_validate_json(data)
        assert loaded.model == "qwen2.5-7b-instruct"
        assert loaded.messages[0].role == "user"
        assert loaded.messages[0].content == "Hello"
        assert loaded.backend == "auto"  # default
        assert loaded.temperature == 0.0

    def test_valid_full(self) -> None:
        """Full request with all fields round-trips."""
        req = ChatRequest(
            model="qwen2.5-7b-instruct",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hi"),
            ],
            backend="ascend",
            max_tokens=512,
            temperature=0.7,
            seed=42,
        )
        data = req.model_dump_json()
        loaded = ChatRequest.model_validate_json(data)
        assert loaded.model == "qwen2.5-7b-instruct"
        assert len(loaded.messages) == 2
        assert loaded.backend == "ascend"
        assert loaded.max_tokens == 512
        assert loaded.temperature == 0.7
        assert loaded.seed == 42

    def test_backend_literal_accepted(self) -> None:
        for val in ("cuda", "ascend", "auto"):
            req = ChatRequest(
                model="m", messages=[Message(role="user", content="x")], backend=val  # type: ignore[arg-type]
            )
            assert req.backend == val

    def test_backend_default_is_auto(self) -> None:
        req = ChatRequest(model="m", messages=[Message(role="user", content="x")])
        assert req.backend == "auto"

    def test_serializes_to_dict(self) -> None:
        req = ChatRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            max_tokens=100,
        )
        d = req.model_dump()
        assert d["model"] == "m"
        assert isinstance(d["messages"], list)
        assert d["backend"] == "auto"
        assert d["max_tokens"] == 100
        assert d["temperature"] == 0.0


# ---------------------------------------------------------------------------
# ChatRequest — rejection of malformed input
# ---------------------------------------------------------------------------

    def test_missing_model_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model"):
            ChatRequest(messages=[Message(role="user", content="hi")])  # type: ignore[call-arg]

    def test_missing_messages_rejected(self) -> None:
        with pytest.raises(ValidationError, match="messages"):
            ChatRequest(model="m")  # type: ignore[call-arg]

    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError, match="messages"):
            ChatRequest(model="m", messages=[])  # type: ignore[arg-type]

    def test_invalid_backend_rejected(self) -> None:
        with pytest.raises(ValidationError, match="backend"):
            ChatRequest(
                model="m",
                messages=[Message(role="user", content="hi")],
                backend="gpu",  # type: ignore[arg-type]
            )

    def test_wrong_message_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(model="m", messages=["hello"])  # type: ignore[list-item]

    def test_temperature_default_zero(self) -> None:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        assert req.temperature == 0.0

    def test_seed_default_none(self) -> None:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        assert req.seed is None


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_valid_message(self) -> None:
        m = Message(role="user", content="Hello world")
        assert m.role == "user"
        assert m.content == "Hello world"

    def test_roundtrip_json(self) -> None:
        m = Message(role="system", content="Be concise")
        data = m.model_dump_json()
        loaded = Message.model_validate_json(data)
        assert loaded.role == "system"
        assert loaded.content == "Be concise"

    def test_wrong_role_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            Message(role=123, content="hi")  # type: ignore[arg-type]

    def test_unknown_role_value_rejected(self) -> None:
        """A well-typed but meaningless role must not reach the backend."""
        with pytest.raises(ValidationError, match="role"):
            Message(role="banana", content="hi")  # type: ignore[arg-type]

    @pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
    def test_known_roles_accepted(self, role: str) -> None:
        assert Message(role=role, content="hi").role == role  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------

class TestChatResponse:
    def test_valid_response_roundtrip(self) -> None:
        resp = ChatResponse(
            id="chatcmpl-123",
            model="qwen2.5-7b-instruct",
            backend="cuda",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        data = resp.model_dump_json()
        loaded = ChatResponse.model_validate_json(data)
        assert loaded.id == "chatcmpl-123"
        assert loaded.backend == "cuda"
        assert len(loaded.choices) == 1
        assert loaded.choices[0].message.content == "Hello!"
        assert loaded.usage.total_tokens == 15

    def test_backend_field_present(self) -> None:
        """Ruitong extension: backend field on response."""
        resp = ChatResponse(
            id="x",
            model="m",
            backend="ascend",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        assert resp.backend == "ascend"
        d = json.loads(resp.model_dump_json())
        assert d["backend"] == "ascend"

    def test_usage_fields(self) -> None:
        resp = ChatResponse(
            id="x",
            model="m",
            backend="cuda",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=42, completion_tokens=7, total_tokens=49),
        )
        assert resp.usage.prompt_tokens == 42
        assert resp.usage.completion_tokens == 7
        assert resp.usage.total_tokens == 49


# ---------------------------------------------------------------------------
# ChatChunk
# ---------------------------------------------------------------------------

class TestChatChunk:
    def test_valid_chunk(self) -> None:
        chunk = ChatChunk(
            id="chatcmpl-123",
            model="qwen2.5-7b-instruct",
            backend="cuda",
            choices=[
                DeltaChoice(
                    index=0,
                    delta=Delta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        data = chunk.model_dump_json()
        loaded = ChatChunk.model_validate_json(data)
        assert loaded.backend == "cuda"
        assert loaded.choices[0].delta.content == "Hello"
        assert loaded.choices[0].finish_reason is None

    def test_chunk_none_fields(self) -> None:
        """Delta fields may be None."""
        chunk = ChatChunk(
            id="x",
            model="m",
            backend="cuda",
            choices=[
                DeltaChoice(index=0, delta=Delta(role=None, content=None), finish_reason=None)
            ],
        )
        assert chunk.choices[0].delta.role is None
        assert chunk.choices[0].delta.content is None


# ---------------------------------------------------------------------------
# HealthStatus
# ---------------------------------------------------------------------------

class TestHealthStatus:
    def test_healthy(self) -> None:
        hs = HealthStatus(healthy=True, backend="cuda", models_served=2, uptime_seconds=3600.0)
        assert hs.healthy is True
        assert hs.uptime_seconds == 3600.0

    def test_unhealthy(self) -> None:
        hs = HealthStatus(healthy=False, backend="ascend", models_served=0)
        assert hs.healthy is False
        assert hs.uptime_seconds is None

    def test_roundtrip(self) -> None:
        hs = HealthStatus(healthy=True, backend="cuda", models_served=1)
        data = json.loads(hs.model_dump_json())
        assert data["healthy"] is True
        assert data["models_served"] == 1


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_valid_model_info(self) -> None:
        mi = ModelInfo(id="qwen2.5-7b-instruct", backend="cuda", max_model_len=4096)
        assert mi.id == "qwen2.5-7b-instruct"
        assert mi.max_model_len == 4096

    def test_roundtrip(self) -> None:
        mi = ModelInfo(id="m", backend="ascend")
        data = json.loads(mi.model_dump_json())
        assert data["id"] == "m"
        assert data["max_model_len"] is None


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

class TestUsage:
    def test_values(self) -> None:
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 150

    def test_roundtrip(self) -> None:
        u = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        data = json.loads(u.model_dump_json())
        assert data["total_tokens"] == 2
