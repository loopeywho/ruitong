"""FastAPI application for Ruitong Bridge."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.router import router as port_router
from .auth.router import router as admin_router
from .auth.keystore import KeyStore
from .backends.fake import FakeAscend, FakeCuda
from .backends.vllm_http import VllmHttpBackend
from .config import BridgeConfig
from .errors import BackendError, BackendUnavailable, ModelNotFound
from .jobs.persistence import JobStore
from .pricing.router import router as pricing_router
from .registry import BackendRegistry
from .router import Router

# ── Exempt paths (no API key required) ───────────────────────────────
AUTH_EXEMPT_PATHS = {"/v1/health", "/v1/models", "/docs", "/openapi.json", "/redoc"}

# ── Bounded rate-limit bucket store parameters ────────────────────────
MAX_TRACKED_PRINCIPALS = 10_000
SWEEP_INTERVAL = 100  # sweep stale entries every N requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create registry, add fakes, create router, init JobStore."""
    config = BridgeConfig.from_env()

    # Warn loudly if auth is off — dev-mode IP-based rate limiting is
    # better than a shared bucket, but it is not a security boundary.
    if not config.api_key and not config.admin_key:
        import logging
        logging.warning(
            "RUITONG_API_KEY and RUITONG_ADMIN_KEY are both unset — "
            "authentication is disabled. Rate limiting keys on client IP. "
            "Do not expose this server to the internet."
        )

    registry = BackendRegistry(config)
    # Register real backends when configured, fakes as fallback (P2.13)
    # accept_any=True: registry-mode fakes accept any model id so the API
    # path behaves identically to the old per-request construction (R7).
    if config.cuda_base_url:
        registry.register("cuda", VllmHttpBackend(name="cuda", base_url=config.cuda_base_url))
    else:
        registry.register("cuda", FakeCuda(accept_any=True))
    if config.ascend_base_url:
        registry.register("ascend", VllmHttpBackend(name="ascend", base_url=config.ascend_base_url))
    else:
        registry.register("ascend", FakeAscend(accept_any=True))
    app.state.router = Router(registry=registry, config=config)
    app.state.config = config
    app.state.job_store = JobStore(db_path=config.job_db_path)
    app.state.key_store = KeyStore(db_path=config.key_db_path)
    app.state.pricing_config = config.pricing_config
    app.state.rate_limit_buckets = {}  # dict[str, list[float]]
    app.state.rate_limit_counter = 0
    # Annotate the local, then assign. `app.state.x: T = ...` is a syntax
    # Python accepts but mypy rejects — annotations are only permitted on
    # names and self attributes, not arbitrary attribute targets.
    background_tasks: set[asyncio.Task] = set()  # track for cleanup
    app.state.background_tasks = background_tasks
    yield
    # Cancel background tasks on shutdown (P2.13 — orphaned tasks)
    for task in app.state.background_tasks:
        task.cancel()
    if app.state.background_tasks:
        await asyncio.wait(app.state.background_tasks, timeout=5.0)


app = FastAPI(
    title="Ruitong Bridge",
    version="0.1.0",
    lifespan=lifespan,
)

# Phase 5 — Port API
app.include_router(port_router)

# Admin API — key lifecycle management
app.include_router(admin_router)

# Pricing API — CNY-native pricing info
app.include_router(pricing_router)


# ── Health / discovery endpoints ──────────────────────────────────────


@app.get("/v1/health")
async def health() -> dict[str, str]:
    """Health check — always returns 200 when the app is alive."""
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, list[str]]:
    """List available models from the registry (R7 — uses registered backends)."""
    router: Router = app.state.router
    models_list = await router.registry.list_models()
    return {"models": sorted({m.id for m in models_list})}


# ── Middleware ────────────────────────────────────────────────────────
# Middleware registration order (last = outermost, runs first):
#   1) payload_size   — reject oversized bodies
#   2) rate_limit     — charge bucket after body check, before handler
#   3) auth           — reject unauthenticated before anything else
#
# This is the inverse of the decorator order below. auth is last so it
# is outermost => runs first. rate_limit is second-outermost, so it only
# sees authenticated requests (fixes H2).
# ──────────────────────────────────────────────────────────────────────


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
async def payload_size_middleware(request: Request, call_next):
    """Reject requests with oversized payloads (Content-Length header).

    Number 3 in the execution order (runs after auth + rate-limit).
    """
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
                return JSONResponse(
                    status_code=400,
                    content={"error": "Bad Request", "detail": "Invalid Content-Length header"},
                )

    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """In-memory rate limiter per authenticated principal (or IP fallback).

    Number 2 in the execution order (runs after auth, before payload check).
    Keys on the authenticated principal set by auth_middleware, so
    unauthenticated traffic never reaches the limiter (fixes H2, S4).

    Bounded dict with periodic sweep prevents unbounded memory growth
    (fixes S3).
    """
    if request.url.path in AUTH_EXEMPT_PATHS:
        return await call_next(request)

    config = _config_for(request)
    if config.rate_limit_per_minute <= 0:
        return await call_next(request)

    now = time.time()
    window = 60.0

    # Lazy-initialize the rate limit state (needed when lifespan doesn't
    # run, e.g. TestClient without asgi_transport).
    if not hasattr(request.app.state, "rate_limit_buckets"):
        request.app.state.rate_limit_buckets = {}
    if not hasattr(request.app.state, "rate_limit_counter"):
        request.app.state.rate_limit_counter = 0

    buckets: dict[str, list[float]] = request.app.state.rate_limit_buckets

    # ── Periodic sweep — every SWEEP_INTERVAL requests ────────────
    # Avoids O(n) scan-per-request that would turn the fix into its own
    # DoS vector.
    request.app.state.rate_limit_counter += 1
    if request.app.state.rate_limit_counter % SWEEP_INTERVAL == 0:
        stale = [k for k, v in buckets.items() if not v or v[-1] < now - window]
        for k in stale:
            del buckets[k]

    # ── Key on the authenticated principal ────────────────────────
    # Set by auth_middleware (which always runs before this middleware).
    # Fall back to IP only when no principal is available (e.g. no auth
    # configured at all — dev mode).
    principal: str | None = getattr(request.state, "api_key_principal", None)
    if principal is None:
        # When no auth is configured, use IP. Behind Cloudflare this is
        # a CF proxy IP — in production with auth configured, this path
        # is never reached for non-exempt endpoints.
        principal = request.client.host if request.client else "unknown"

    # ── Cap total tracked principals ──────────────────────────────
    # Prevents the spray-unique-principals exhaustion vector.
    if principal not in buckets and len(buckets) >= MAX_TRACKED_PRINCIPALS:
        # Evict the stalest bucket (oldest most-recent timestamp)
        stalest_key = min(
            buckets,
            key=lambda k: buckets[k][-1] if buckets[k] else 0,
        )
        del buckets[stalest_key]

    timestamps = buckets.setdefault(principal, [])
    # Prune expired entries for this principal
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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Check API key on every request unless path is exempt.

    Registered LAST (outermost) so it runs FIRST — unauthenticated
    requests are rejected before they reach the rate limiter or
    payload-size check (H2 fix).

    Always sets request.state.api_key_principal so downstream middleware
    can key on the authenticated principal rather than IP (S4 fix).

    Multi-key support:
    - When admin_key is set: authenticate against KeyStore, falling back
      to direct admin_key comparison.
    - When admin_key is empty: fall back to legacy single-key auth
      against config.api_key.
    """
    config = _config_for(request)
    exempt = request.url.path in AUTH_EXEMPT_PATHS

    # Auth is required when api_key OR admin_key is set
    has_auth = bool(config.api_key) or bool(config.admin_key)

    if has_auth and not exempt:
        provided = request.headers.get("X-API-Key", "")

        if config.admin_key:
            # KeyStore mode: try KeyStore first, then admin key directly.
            key_store = getattr(request.app.state, "key_store", None)
            authenticated = False

            if key_store is not None and provided:
                key_id = key_store.authenticate(provided)
                if key_id is not None:
                    request.state.api_key_principal = key_id
                    authenticated = True

            if not authenticated and provided:
                # Fall back to direct admin_key check.
                if hmac.compare_digest(
                    provided.encode("latin-1"),
                    config.admin_key.encode("utf-8"),
                ):
                    request.state.api_key_principal = "admin"
                    authenticated = True

            if not authenticated:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized", "detail": "Missing or invalid X-API-Key header"},
                )
        else:
            # Legacy: single-key auth against config.api_key
            if not provided or not hmac.compare_digest(
                provided.encode("latin-1"),
                config.api_key.encode("utf-8"),
            ):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized", "detail": "Missing or invalid X-API-Key header"},
                )
            # Use a stable hash of the key as principal — never store the raw key
            # in request state or persist it to the database (P2 F4 fix).
            request.state.api_key_principal = hashlib.sha256(
                provided.encode("utf-8")
            ).hexdigest()[:16]

    else:
        # No auth configured (dev mode) — do not set api_key_principal,
        # so rate_limit_middleware keys on IP instead of one shared
        # "anonymous" bucket that a single noisy user exhausts for everyone.
        pass  # principal stays unset → IP-based bucket in rate limiter

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
