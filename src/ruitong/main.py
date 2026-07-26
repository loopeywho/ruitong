"""FastAPI application for Ruitong Bridge."""
from __future__ import annotations

import hmac
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.router import router as port_router
from .backends.fake import FakeAscend, FakeCuda
from .config import BridgeConfig
from .errors import BackendError, BackendUnavailable, ModelNotFound
from .jobs.persistence import JobStore
from .registry import BackendRegistry
from .router import Router
from .schemas import ChatRequest

# ── Exempt paths (no API key required) ───────────────────────────────
AUTH_EXEMPT_PATHS = {"/v1/health", "/v1/models", "/docs", "/openapi.json", "/redoc"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create registry, add fakes, create router, init JobStore."""
    config = BridgeConfig.from_env()
    registry = BackendRegistry(config)
    registry.register("cuda", FakeCuda())
    registry.register("ascend", FakeAscend())
    app.state.router = Router(registry=registry, config=config)
    app.state.config = config
    app.state.job_store = JobStore(db_path=config.job_db_path)
    app.state.rate_limit_buckets = defaultdict(list)
    yield


app = FastAPI(
    title="Ruitong Bridge",
    version="0.1.0",
    lifespan=lifespan,
)

# Phase 5 — Port API
app.include_router(port_router)


# ── Health / discovery endpoints ──────────────────────────────────────


@app.get("/v1/health")
async def health() -> dict[str, str]:
    """Health check — always returns 200 when the app is alive."""
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, list[str]]:
    """List available models (from fake backends in dev)."""
    models_set: set[str] = set()
    for backend_cls in (FakeCuda, FakeAscend):
        backend = backend_cls()
        try:
            for m in await backend.list_models():
                models_set.add(m.id)
        except Exception:
            pass
    return {"models": sorted(models_set)}


# ── Middleware ────────────────────────────────────────────────────────


def _config_for(request: Request) -> BridgeConfig:
    """Return the app's config, falling back to the environment.

    Deliberately not `getattr(state, "config", BridgeConfig.from_env())`:
    Python evaluates a call's arguments eagerly, so that form re-reads
    `os.environ` and rebuilds the config on *every* request even when one is
    already cached — three times per request across the middleware stack.
    """
    config: BridgeConfig | None = getattr(request.app.state, "config", None)
    return config if config is not None else BridgeConfig.from_env()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Check API key on every request unless path is exempt."""
    config = _config_for(request)

    if config.api_key and request.url.path not in AUTH_EXEMPT_PATHS:
        provided = request.headers.get("X-API-Key", "")
        # Constant-time compare: `!=` leaks key length and prefix via response
        # timing, letting an attacker recover the key byte by byte.
        if not hmac.compare_digest(provided, config.api_key):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Missing or invalid X-API-Key header"},
            )

    return await call_next(request)


@app.middleware("http")
async def payload_size_middleware(request: Request, call_next):
    """Reject requests with oversized payloads (Content-Length header)."""
    config = _config_for(request)

    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
                if size > config.max_payload_bytes:
                    max_mb = config.max_payload_bytes // (1024 * 1024)
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload too large",
                            "detail": f"Maximum payload size is {max_mb} MB",
                            "max_bytes": config.max_payload_bytes,
                        },
                    )
            except (ValueError, TypeError):
                pass

    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter per client IP."""
    if request.url.path in AUTH_EXEMPT_PATHS:
        return await call_next(request)

    config = _config_for(request)
    if config.rate_limit_per_minute > 0:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0

        # Lazy-initialize the rate limit buckets
        if not hasattr(request.app.state, "rate_limit_buckets"):
            request.app.state.rate_limit_buckets = defaultdict(list)
        buckets = request.app.state.rate_limit_buckets

        timestamps = buckets[client_ip]
        # Prune old entries
        while timestamps and timestamps[0] < now - window:
            timestamps.pop(0)

        if len(timestamps) >= config.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Max {config.rate_limit_per_minute} requests per minute",
                    "retry_after_seconds": int(timestamps[0] + window - now),
                },
            )

        timestamps.append(now)

    return await call_next(request)


# ── Error handlers ────────────────────────────────────────────────────


@app.exception_handler(BackendUnavailable)
async def unavailable_handler(request: Request, exc: BackendUnavailable):
    return JSONResponse(status_code=503, content={"error": exc.message})


@app.exception_handler(ModelNotFound)
async def not_found_handler(request: Request, exc: ModelNotFound):
    return JSONResponse(status_code=404, content={"error": exc.message})


@app.exception_handler(BackendError)
async def backend_error_handler(request: Request, exc: BackendError):
    return JSONResponse(status_code=502, content={"error": exc.message})


@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception):
    """Catch unexpected errors — ensures every failure names the backend if possible."""
    backend_name = getattr(exc, "backend", None) or "unknown"
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {type(exc).__name__}", "backend": str(backend_name)},
    )