"""Acceptance tests for Router, Registry, and FastAPI endpoints."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

import pytest

from ruitong.backends.fake import FakeCuda, FakeAscend
from ruitong.config import BridgeConfig
from ruitong.errors import BackendUnavailable, ModelNotFound, BackendError
from ruitong.registry import BackendRegistry
from ruitong.router import Router
from ruitong.schemas import ChatRequest, Message


# ---------------------------------------------------------------------------
# Helper: build a test FastAPI app with configurable fake backends
# ---------------------------------------------------------------------------


def _make_test_app(
    fake_cuda: FakeCuda | None = None,
    fake_ascend: FakeAscend | None = None,
) -> FastAPI:
    """Build a test app with the given fake backends."""
    config = BridgeConfig()
    registry = BackendRegistry(config)
    cuda = fake_cuda or FakeCuda()
    ascend = fake_ascend or FakeAscend()
    registry.register("cuda", cuda)
    registry.register("ascend", ascend)
    router = Router(registry=registry, config=config)

    test_app = FastAPI()
    test_app.state.router = router

    @test_app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request):
        if req.stream:
            async def event_gen():
                async for chunk in router.stream(req):
                    yield f"data: {chunk.model_dump_json()}\n\n"
            return StreamingResponse(event_gen(), media_type="text/event-stream")
        resp = await router.chat(req)
        return JSONResponse(content=resp.model_dump())

    @test_app.get("/v1/models")
    async def models_list():
        models = await router.list_models()
        return [m.model_dump() for m in models]

    @test_app.get("/v1/health")
    async def health():
        return await router.health()

    @test_app.exception_handler(BackendUnavailable)
    async def unavailable_handler(request: Request, exc: BackendUnavailable):
        return JSONResponse(status_code=503, content={"error": exc.message})

    @test_app.exception_handler(ModelNotFound)
    async def not_found_handler(request: Request, exc: ModelNotFound):
        return JSONResponse(status_code=404, content={"error": exc.message})

    @test_app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError):
        return JSONResponse(status_code=502, content={"error": exc.message})

    return test_app


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        cuda = FakeCuda()
        registry.register("cuda", cuda)
        assert registry.get("cuda") is cuda
        assert registry.list_backends() == ["cuda"]

    def test_register_overwrites(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        cuda1, cuda2 = FakeCuda(), FakeCuda()
        registry.register("cuda", cuda1)
        registry.register("cuda", cuda2)
        assert registry.get("cuda") is cuda2

    def test_get_unknown_raises(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        with pytest.raises(BackendUnavailable):
            registry.get("unknown")

    def test_list_returns_names(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda())
        registry.register("ascend", FakeAscend())
        assert registry.list_backends() == ["cuda", "ascend"]

    @pytest.mark.asyncio
    async def test_resolve_auto_both_healthy(self) -> None:
        config = BridgeConfig(auto_backend_priority=("cuda", "ascend"))
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda())
        registry.register("ascend", FakeAscend())
        backend = await registry.resolve("qwen2.5-7b-instruct")
        assert backend.name == "cuda"

    @pytest.mark.asyncio
    async def test_resolve_auto_one_unhealthy(self) -> None:
        config = BridgeConfig(auto_backend_priority=("cuda", "ascend"))
        registry = BackendRegistry(config)
        cuda = FakeCuda()
        cuda._set_unhealthy(True)
        registry.register("cuda", cuda)
        registry.register("ascend", FakeAscend())
        backend = await registry.resolve("qwen2.5-7b-instruct")
        assert backend.name == "ascend"

    @pytest.mark.asyncio
    async def test_resolve_auto_both_unhealthy(self) -> None:
        config = BridgeConfig(auto_backend_priority=("cuda", "ascend"))
        registry = BackendRegistry(config)
        cuda = FakeCuda()
        cuda._set_unhealthy(True)
        ascend = FakeAscend()
        ascend._set_unhealthy(True)
        registry.register("cuda", cuda)
        registry.register("ascend", ascend)
        with pytest.raises(BackendUnavailable):
            await registry.resolve("qwen2.5-7b-instruct")

    @pytest.mark.asyncio
    async def test_resolve_auto_model_absent(self) -> None:
        config = BridgeConfig(auto_backend_priority=("cuda", "ascend"))
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda(model_ids=["qwen2.5-7b-instruct"]))
        registry.register("ascend", FakeAscend(model_ids=["qwen3-8b"]))
        with pytest.raises(ModelNotFound):
            await registry.resolve("nonexistent-model")

    @pytest.mark.asyncio
    async def test_resolve_auto_no_candidates(self) -> None:
        """When priority references unregistered backends, raise BackendUnavailable."""
        config = BridgeConfig(auto_backend_priority=("gpu", "tpu"))
        registry = BackendRegistry(config)
        with pytest.raises(BackendUnavailable):
            await registry.resolve("m")

    @pytest.mark.asyncio
    async def test_resolve_explicit_healthy(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("ascend", FakeAscend())
        backend = await registry.resolve("qwen3-8b", preferred_backend="ascend")
        assert backend.name == "ascend"

    @pytest.mark.asyncio
    async def test_resolve_explicit_model_not_found(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda())
        with pytest.raises(ModelNotFound):
            await registry.resolve("qwen3-8b", preferred_backend="cuda")

    @pytest.mark.asyncio
    async def test_resolve_explicit_not_registered(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        with pytest.raises(BackendUnavailable):
            await registry.resolve("m", preferred_backend="unknown")


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestRouter:
    @pytest.mark.asyncio
    async def test_chat(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda())
        router = Router(registry=registry, config=config)
        req = ChatRequest(
            model="qwen2.5-7b-instruct",
            messages=[Message(role="user", content="hi")],
        )
        resp = await router.chat(req)
        assert resp.backend == "cuda"
        assert resp.choices[0].message.content == "I am CUDA"

    @pytest.mark.asyncio
    async def test_stream(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("ascend", FakeAscend())
        router = Router(registry=registry, config=config)
        req = ChatRequest(
            model="qwen3-8b",
            messages=[Message(role="user", content="hi")],
        )
        chunks = []
        async for chunk in router.stream(req):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].backend == "ascend"
        assert chunks[0].choices[0].delta.content == "I am Ascend"

    @pytest.mark.asyncio
    async def test_health(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda())
        registry.register("ascend", FakeAscend())
        router = Router(registry=registry, config=config)
        statuses = await router.health()
        assert len(statuses) == 2
        assert all(s.healthy for s in statuses)

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        registry.register("cuda", FakeCuda())
        registry.register("ascend", FakeAscend())
        router = Router(registry=registry, config=config)
        models = await router.list_models()
        assert len(models) == 4  # 2 cuda + 2 ascend

    @pytest.mark.asyncio
    async def test_chat_wraps_non_ruitong_error(self) -> None:
        config = BridgeConfig()
        registry = BackendRegistry(config)
        cuda = FakeCuda()
        cuda._set_error(RuntimeError("gpu memory error"))
        registry.register("cuda", cuda)
        router = Router(registry=registry, config=config)
        req = ChatRequest(
            model="qwen2.5-7b-instruct",
            messages=[Message(role="user", content="hi")],
        )
        with pytest.raises(BackendError) as exc_info:
            await router.chat(req)
        assert "cuda" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, RuntimeError)

    @pytest.mark.asyncio
    async def test_chat_passes_ruitong_error_through(self) -> None:
        """RuitongError from backend.chat propagates without wrapping (line 26)."""
        config = BridgeConfig()
        registry = BackendRegistry(config)
        cuda = FakeCuda()
        cuda._set_error(ModelNotFound("nope", "cuda"))
        registry.register("cuda", cuda)
        router = Router(registry=registry, config=config)
        req = ChatRequest(
            model="nope",
            messages=[Message(role="user", content="hi")],
        )
        with pytest.raises(ModelNotFound):
            await router.chat(req)

    @pytest.mark.asyncio
    async def test_stream_passes_ruitong_error(self) -> None:
        """RuitongError from backend.stream propagates without wrapping (line 37)."""
        config = BridgeConfig()
        registry = BackendRegistry(config)
        ascend = FakeAscend()
        ascend._set_error(ModelNotFound("nope", "ascend"))
        registry.register("ascend", ascend)
        router = Router(registry=registry, config=config)
        req = ChatRequest(
            model="nope",
            messages=[Message(role="user", content="hi")],
        )
        chunks = []
        with pytest.raises(ModelNotFound):
            async for chunk in router.stream(req):
                chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_stream_wraps_non_ruitong_error(self) -> None:
        """Non-RuitongError from backend.stream gets wrapped in BackendError."""
        config = BridgeConfig()
        registry = BackendRegistry(config)
        ascend = FakeAscend()
        ascend._set_error(RuntimeError("oops"))
        registry.register("ascend", ascend)
        router = Router(registry=registry, config=config)
        req = ChatRequest(
            model="qwen2.5-7b-instruct",
            messages=[Message(role="user", content="hi")],
        )
        chunks = []
        with pytest.raises(BackendError) as exc_info:
            async for chunk in router.stream(req):
                chunks.append(chunk)
        assert "ascend" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, RuntimeError)


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestHTTPChatCompletions:
    def test_post_chat_returns_response(self) -> None:
        app = _make_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-7b-instruct",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "cuda"
        assert data["choices"][0]["message"]["content"] == "I am CUDA"

    def test_post_chat_auto_routes_to_ascend(self) -> None:
        app = _make_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "ascend"

    def test_post_chat_explicit_backend(self) -> None:
        app = _make_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "backend": "ascend",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "ascend"

    def test_model_not_found_returns_404(self) -> None:
        app = _make_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404

    def test_backend_unavailable_returns_503(self) -> None:
        cuda = FakeCuda()
        cuda._set_unhealthy(True)
        ascend = FakeAscend()
        ascend._set_unhealthy(True)
        app = _make_test_app(fake_cuda=cuda, fake_ascend=ascend)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-7b-instruct",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 503

    def test_backend_error_returns_502(self) -> None:
        cuda = FakeCuda()
        cuda._set_error(RuntimeError("gpu memory error"))
        app = _make_test_app(fake_cuda=cuda)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-7b-instruct",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 502


class TestHTTPModels:
    def test_get_models(self) -> None:
        app = _make_test_app()
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) == 4


class TestHTTPHealth:
    def test_get_health(self) -> None:
        app = _make_test_app()
        client = TestClient(app)
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        health_data = resp.json()
        assert len(health_data) == 2
        assert all(h["healthy"] for h in health_data)
