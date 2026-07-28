"""Ruitong Bridge — Admin API: API key lifecycle management."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request

from ..config import BridgeConfig
from .keystore import KeyStore

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _check_admin_key(request: Request) -> None:
    """Verify the X-API-Key header matches the configured admin key.

    Fail-closed: if RUITONG_ADMIN_KEY is not set, the admin API is
    disabled (503).  There is no legacy fallback — a data-plane key
    never grants admin access.
    """
    config: BridgeConfig | None = getattr(request.app.state, "config", None)
    if config is None:
        config = BridgeConfig.from_env()

    if not config.admin_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API disabled: RUITONG_ADMIN_KEY is not set",
        )

    provided = request.headers.get("X-API-Key", "")
    if not provided:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if not hmac.compare_digest(
        provided.encode("latin-1"),
        config.admin_key.encode("utf-8"),
    ):
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")


def _get_key_store(request: Request) -> KeyStore:
    """Return the KeyStore instance from app state.

    The store must be wired in ``lifespan`` — there is no fallback
    to a default singleton, because an ephemeral credential store
    is a data-loss risk.
    """
    key_store = getattr(request.app.state, "key_store", None)
    if key_store is None:
        raise HTTPException(
            status_code=503,
            detail="Key store not initialised — server may still be starting",
        )
    return key_store


@router.post("/keys")
async def create_key(request: Request) -> dict:
    """Create a new API key.  Requires the admin key in X-API-Key header."""
    _check_admin_key(request)
    key_store = _get_key_store(request)

    body = await request.json()
    name: str = body.get("name", "")
    if not name:
        raise HTTPException(status_code=400, detail="Body must include 'name'")

    key_id, plaintext = key_store.create_key(name)
    return {
        "key_id": key_id,
        "plaintext_key": plaintext,
        "name": name,
    }


@router.get("/keys")
async def list_keys(request: Request) -> list[dict]:
    """List all API keys (metadata only — hashes are never exposed)."""
    _check_admin_key(request)
    key_store = _get_key_store(request)
    return key_store.list_keys()


@router.delete("/keys/{key_id}")
async def revoke_key(request: Request, key_id: str) -> dict:
    """Revoke (deactivate) an API key."""
    _check_admin_key(request)
    key_store = _get_key_store(request)

    revoked = key_store.revoke_key(key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
    return {"revoked": True}