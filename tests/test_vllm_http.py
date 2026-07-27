"""Tests for the vLLM HTTP backend.

Every failure mode is mocked with respx — no live network, fully deterministic.
"""

from __future__ import annotations

import httpx
import json
import pytest
import respx

from ruitong.backends.vllm_http import VllmHttpBackend
from ruitong.errors import BackendError, BackendUnavailable
from ruitong.schemas import ChatRequest, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_req(model: str = "qwen3-8b") -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[Message(role="user", content="hi")],
    )

# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


class TestHealth:
    @respx.mock
    async def test_health_200(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        route = respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json=[{"id": "m"}])
        )
        result = await backend.health()
        assert route.called
        assert result.healthy is True
        assert result.backend == "cuda"
        assert result.models_served == 1

    @respx.mock
    async def test_health_503(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(503)
        )
        result = await backend.health()
        assert result.healthy is False
        assert result.backend == "cuda"
        assert result.models_served == 0

    @respx.mock
    async def test_health_timeout_wont_raise(self) -> None:
        """health() never raises — wraps errors into unhealthy status."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        result = await backend.health()
        assert result.healthy is False

    @respx.mock
    async def test_health_connection_refused(self) -> None:
        """Connection refused → unhealthy, no exception."""
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.get("http://localhost:8001/v1/models").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await backend.health()
        assert result.healthy is False
        assert result.backend == "ascend"

# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


class TestListModels:
    @respx.mock
    async def test_list_models_success(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json=[{"id": "qwen3-8b"}])
        )
        models = await backend.list_models()
        assert len(models) == 1
        assert models[0].id == "qwen3-8b"
        assert models[0].backend == "cuda"

    @respx.mock
    async def test_list_models_multiple(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(
                200, json=[{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
            )
        )
        models = await backend.list_models()
        assert len(models) == 3

    @respx.mock
    async def test_list_models_timeout_raises_unavailable(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        with pytest.raises(BackendUnavailable):
            await backend.list_models()

    @respx.mock
    async def test_list_models_http_500_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(BackendError):
            await backend.list_models()

# ---------------------------------------------------------------------------
# chat() — non-streaming
# ---------------------------------------------------------------------------


class TestChat:
    @respx.mock
    async def test_chat_success(self) -> None:
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        body = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        resp = await backend.chat(_chat_req())
        assert resp.backend == "ascend"
        assert resp.choices[0].message.content == "hello"

    @respx.mock
    async def test_chat_empty_200_body(self) -> None:
        """Empty 200 body → BackendError (known vllm-ascend defect)."""
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="")
        )
        with pytest.raises(BackendError):
            await backend.chat(_chat_req())

    @respx.mock
    async def test_chat_404(self) -> None:
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(BackendError) as exc_info:
            await backend.chat(_chat_req())
        assert "404" in str(exc_info.value)

    @respx.mock
    async def test_chat_timeout_raises_unavailable(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        with pytest.raises(BackendUnavailable):
            await backend.chat(_chat_req())

    @respx.mock
    async def test_chat_connection_refused_raises_unavailable(self) -> None:
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(BackendUnavailable):
            await backend.chat(_chat_req())

    @respx.mock
    async def test_chat_malformed_body_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="not json at all")
        )
        with pytest.raises(BackendError):
            await backend.chat(_chat_req())

    @respx.mock
    async def test_chat_empty_choices_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        body = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        with pytest.raises(BackendError):
            await backend.chat(_chat_req())

    @respx.mock
    async def test_chat_http_500_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(BackendError) as exc_info:
            await backend.chat(_chat_req())
        assert "500" in str(exc_info.value)

    @respx.mock
    async def test_chat_http_400_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(400)
        )
        with pytest.raises(BackendError):
            await backend.chat(_chat_req())

    @respx.mock
    async def test_chat_response_fields_correct(self) -> None:
        """Verify all ChatResponse fields parse correctly."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        body = {
            "id": "cmpl-abc-123",
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test content",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 24,
                "total_tokens": 36,
            },
            "created": 1700000000,
        }
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        resp = await backend.chat(_chat_req())
        assert resp.id == "cmpl-abc-123"
        assert resp.model == "qwen3-8b"
        assert resp.backend == "cuda"
        assert resp.usage.prompt_tokens == 12
        assert resp.usage.completion_tokens == 24
        assert resp.usage.total_tokens == 36
        assert resp.created == 1700000000

# ---------------------------------------------------------------------------
# stream() — SSE parsing
# ---------------------------------------------------------------------------


def _sse_response(json_body: dict) -> str:
    """Build an SSE line for a chunk (no trailing [DONE])."""
    return f"data: {json.dumps(json_body)}\n\n"


def _sse_body(*json_bodies: dict) -> str:
    """Build a complete SSE response from multiple JSON bodies + [DONE]."""
    return "".join(f"data: {json.dumps(b)}\n\n" for b in json_bodies) + "data: [DONE]\n"


def _sse_response_done(json_body: dict) -> str:
    """Build an SSE string including the final [DONE]."""
    return _sse_response(json_body) + "data: [DONE]\n"


class TestStream:
    @respx.mock
    async def test_stream_success(self) -> None:
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        body = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=_sse_body(body))
        )
        chunks = []
        async for chunk in backend.stream(_chat_req()):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].backend == "ascend"
        assert chunks[0].choices[0].delta.content == "hello"

    @respx.mock
    async def test_stream_empty_200_body(self) -> None:
        """Empty 200 body → BackendError (known vllm-ascend defect)."""
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="")
        )
        chunks = []
        with pytest.raises(BackendError):
            async for chunk in backend.stream(_chat_req()):
                chunks.append(chunk)

    @respx.mock
    async def test_stream_empty_choices(self) -> None:
        body = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [],
        }
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=_sse_response_done(body))
        )
        chunks = []
        with pytest.raises(BackendError):
            async for chunk in backend.stream(_chat_req()):
                chunks.append(chunk)

    @respx.mock
    async def test_stream_timeout_raises_unavailable(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        chunks = []
        with pytest.raises(BackendUnavailable):
            async for chunk in backend.stream(_chat_req()):
                chunks.append(chunk)

    @respx.mock
    async def test_stream_connection_refused_raises_unavailable(self) -> None:
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        chunks = []
        with pytest.raises(BackendUnavailable):
            async for chunk in backend.stream(_chat_req()):
                chunks.append(chunk)

    @respx.mock
    async def test_stream_http_500_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )
        chunks = []
        with pytest.raises(BackendError):
            async for chunk in backend.stream(_chat_req()):
                chunks.append(chunk)

    @respx.mock
    async def test_stream_multiple_chunks(self) -> None:
        """Multiple SSE chunks yield multiple ChatChunks."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        chunk1 = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [
                {"index": 0, "delta": {"content": "h"}, "finish_reason": None}
            ],
        }
        chunk2 = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [
                {"index": 0, "delta": {"content": "i"}, "finish_reason": "stop"}
            ],
        }
        sse = _sse_response(chunk1).rstrip() + "\n\ndata: " + json.dumps(chunk2) + "\n\ndata: [DONE]\n"
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse)
        )
        chunks = []
        async for chunk in backend.stream(_chat_req()):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "h"
        assert chunks[1].choices[0].delta.content == "i"
        assert chunks[1].choices[0].finish_reason == "stop"

    @respx.mock
    async def test_stream_skip_invalid_sse_lines(self) -> None:
        """Malformed SSE lines are silently skipped."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        body = {
            "id": "cmpl-1",
            "model": "qwen3-8b",
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
            ],
        }
        # Inject random garbage lines
        sse = (
            "random garbage\n"
            "data: not-json\n"
            f"data: {json.dumps(body)}\n"
            "\n"
            "data: [DONE]\n"
        )
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse)
        )
        chunks = []
        async for chunk in backend.stream(_chat_req()):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "ok"

    @respx.mock
    async def test_stream_404_raises_backend_error(self) -> None:
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            return_value=httpx.Response(404)
        )
        chunks = []
        with pytest.raises(BackendError):
            async for chunk in backend.stream(_chat_req()):
                chunks.append(chunk)

# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    @respx.mock
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """Connection error on first attempt, success on retry."""
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        call_count = 0

        def on_request(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("refused")
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "model": "qwen3-8b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        respx.post("http://localhost:8001/v1/chat/completions").mock(
            side_effect=on_request
        )
        resp = await backend.chat(_chat_req())
        assert call_count == 2
        assert resp.choices[0].message.content == "ok"

    @respx.mock
    async def test_no_retry_on_404(self) -> None:
        """4xx/5xx errors are NOT retried."""
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        call_count = 0

        def on_request(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(404)

        respx.post("http://localhost:8001/v1/chat/completions").mock(
            side_effect=on_request
        )
        with pytest.raises(BackendError):
            await backend.chat(_chat_req())
        assert call_count == 1

    @respx.mock
    async def test_no_retry_on_500(self) -> None:
        """Server errors are NOT retried."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        call_count = 0

        def on_request(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500)

        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=on_request
        )
        with pytest.raises(BackendError):
            await backend.chat(_chat_req())
        assert call_count == 1

    @respx.mock
    async def test_retry_exhausted_both_attempts_fail(self) -> None:
        """Both attempts fail → BackendUnavailable."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        call_count = 0

        def on_request(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("refused")

        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=on_request
        )
        with pytest.raises(BackendUnavailable):
            await backend.chat(_chat_req())
        assert call_count == 2

    @respx.mock
    async def test_retry_timeout_then_timeout(self) -> None:
        """Timeout on both attempts → BackendUnavailable."""
        backend = VllmHttpBackend("ascend", "http://localhost:8001")
        call_count = 0

        def on_request(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("timeout")

        respx.post("http://localhost:8001/v1/chat/completions").mock(
            side_effect=on_request
        )
        with pytest.raises(BackendUnavailable):
            await backend.chat(_chat_req())
        assert call_count == 2

# ---------------------------------------------------------------------------
# Backend protocol conformance
# ---------------------------------------------------------------------------


class TestBackendProtocol:
    def test_has_name_attribute(self) -> None:
        """Every instance has a .name attribute."""
        cuda = VllmHttpBackend("cuda", "http://localhost:8000")
        ascend = VllmHttpBackend("ascend", "http://localhost:8001")
        assert cuda.name == "cuda"
        assert ascend.name == "ascend"

    def test_same_class_both_backends(self) -> None:
        """Both backends are the same class — no branching on name."""
        assert type(VllmHttpBackend("cuda", "http://a")) is VllmHttpBackend
        assert type(VllmHttpBackend("ascend", "http://b")) is VllmHttpBackend

    @respx.mock
    async def test_trailing_slash_stripped(self) -> None:
        """Trailing slashes in base_url are stripped."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000/")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json=[])
        )
        await backend.health()  # should not raise


# ---------------------------------------------------------------------------
# Real wire shapes — pinned to a live server, not to a fixture
# ---------------------------------------------------------------------------


class TestRealModelsEnvelope:
    """`/v1/models` returns an envelope, not a bare list.

    Captured verbatim from vLLM serving Qwen/Qwen3-8B on an NVIDIA A40
    (RunPod pod vjqkdwuzls8hnf, 2026-07-27). Every prior test used a bare
    list, which no OpenAI-compatible server has ever returned — so the whole
    suite agreed with a fixture and missed that model ids resolved to "".
    """

    REAL_BODY = {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3-8B",
                "object": "model",
                "created": 1769450000,
                "owned_by": "vllm",
                "root": "Qwen/Qwen3-8B",
                "parent": None,
                "max_model_len": 8128,
            }
        ],
    }

    @respx.mock
    async def test_list_models_reads_data_envelope(self) -> None:
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json=self.REAL_BODY)
        )
        models = await backend.list_models()
        assert [m.id for m in models] == ["Qwen/Qwen3-8B"]

    @respx.mock
    async def test_health_counts_models_not_envelope_keys(self) -> None:
        """Counting the envelope's keys reported 2 models for any response."""
        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json=self.REAL_BODY)
        )
        status = await backend.health()
        assert status.healthy is True
        assert status.models_served == 1

    @respx.mock
    async def test_api_key_is_sent_as_bearer(self) -> None:
        """A hosted endpoint rejects unauthenticated calls; without this the
        harness can only ever reach an open port."""
        captured: dict[str, str] = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json=self.REAL_BODY)

        backend = VllmHttpBackend(
            "cuda", "http://localhost:8000", api_key="sk-test-123"
        )
        respx.get("http://localhost:8000/v1/models").mock(side_effect=_capture)
        await backend.list_models()
        assert captured["auth"] == "Bearer sk-test-123"

    @respx.mock
    async def test_no_auth_header_when_no_key(self) -> None:
        captured: dict[str, str] = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json=self.REAL_BODY)

        backend = VllmHttpBackend("cuda", "http://localhost:8000")
        respx.get("http://localhost:8000/v1/models").mock(side_effect=_capture)
        await backend.list_models()
        assert captured["auth"] == ""
