"""Tests for KeyStore and Admin API."""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from ruitong.auth.keystore import KeyStore


# ── KeyStore unit tests ───────────────────────────────────────────────


class TestKeyStore:
    """KeyStore CRUD and auth tests — always use :memory: for isolation."""

    def test_create_key_returns_rt_prefixed_plaintext(self) -> None:
        """create_key returns (key_id, plaintext) where plaintext starts with rt_."""
        ks = KeyStore(":memory:")
        try:
            key_id, plaintext = ks.create_key("test-key")
            assert key_id is not None
            assert plaintext.startswith("rt_")
            assert len(plaintext) > len("rt_")
        finally:
            ks.close()

    def test_authenticate_accepts_plaintext(self) -> None:
        """authenticate returns key_id when given the correct plaintext key."""
        ks = KeyStore(":memory:")
        try:
            key_id, plaintext = ks.create_key("auth-test")
            result = ks.authenticate(plaintext)
            assert result == key_id
        finally:
            ks.close()

    def test_authenticate_rejects_wrong_key(self) -> None:
        """authenticate returns None for an incorrect key."""
        ks = KeyStore(":memory:")
        try:
            result = ks.authenticate(
                "rt_000000000000000000000000000000000000000000000000"
            )
            assert result is None
        finally:
            ks.close()

    def test_revoke_key_makes_key_invalid(self) -> None:
        """After revoke, authenticate returns None for the same key."""
        ks = KeyStore(":memory:")
        try:
            key_id, plaintext = ks.create_key("revoke-test")
            assert ks.authenticate(plaintext) == key_id

            revoked = ks.revoke_key(key_id)
            assert revoked is True

            assert ks.authenticate(plaintext) is None
        finally:
            ks.close()

    def test_revoke_key_nonexistent(self) -> None:
        """revoking an unknown key returns False."""
        ks = KeyStore(":memory:")
        try:
            revoked = ks.revoke_key("nonexistent-id")
            assert revoked is False
        finally:
            ks.close()

    def test_list_keys_returns_metadata_only(self) -> None:
        """list_keys returns dicts with metadata, no hashes."""
        ks = KeyStore(":memory:")
        try:
            ks.create_key("first")
            ks.create_key("second")
            keys = ks.list_keys()
            assert len(keys) == 2
            for k in keys:
                assert "key_id" in k
                assert "name" in k
                assert "prefix" in k
                assert "created_at" in k
                assert "is_active" in k
                # key_hash should NOT be present
                assert "key_hash" not in k
        finally:
            ks.close()

    def test_list_keys_after_revoke(self) -> None:
        """Revoked keys still appear in list_keys with is_active=0."""
        ks = KeyStore(":memory:")
        try:
            key_id, _ = ks.create_key("to-revoke")
            ks.revoke_key(key_id)
            keys = ks.list_keys()
            revoked = [k for k in keys if k["key_id"] == key_id]
            assert len(revoked) == 1
            assert revoked[0]["is_active"] == 0
        finally:
            ks.close()

    def test_update_last_used(self) -> None:
        """update_last_used sets the timestamp."""
        ks = KeyStore(":memory:")
        try:
            key_id, plaintext = ks.create_key("last-used")
            # Initially None
            keys = ks.list_keys()
            first = [k for k in keys if k["key_id"] == key_id][0]
            assert first["last_used_at"] is None

            ks.update_last_used(key_id)

            keys = ks.list_keys()
            updated = [k for k in keys if k["key_id"] == key_id][0]
            assert updated["last_used_at"] is not None
        finally:
            ks.close()


# ── Admin API endpoint tests ──────────────────────────────────────────


class TestAdminAPI:
    """Admin API endpoint tests — requires admin_key = "admin-secret"."""

    ENV_VARS = {"RUITONG_ADMIN_KEY": "admin-secret", "RUITONG_API_KEY": "legacy-key"}

    def _setup_client(self, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        """Set up env vars, reload app, wire an in-memory KeyStore into app state."""
        for k, v in self.ENV_VARS.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("RUITONG_KEY_DB_PATH", raising=False)

        import ruitong.main

        importlib.reload(ruitong.main)

        # Wire an in-memory KeyStore into app state (lifespan doesn't run
        # under TestClient, so we do it manually).
        ruitong.main.app.state.key_store = KeyStore(":memory:")
        ruitong.main.app.state.config = ruitong.main.BridgeConfig.from_env()

        return TestClient(ruitong.main.app)

    def test_create_key_returns_plaintext(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create key returns key_id, plaintext_key, and name."""
        client = self._setup_client(monkeypatch)

        resp = client.post(
            "/v1/admin/keys",
            json={"name": "my-key"},
            headers={"X-API-Key": "admin-secret"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "key_id" in data
        assert "plaintext_key" in data
        assert data["plaintext_key"].startswith("rt_")
        assert data["name"] == "my-key"

    def test_list_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """List keys returns all keys with metadata."""
        client = self._setup_client(monkeypatch)

        # Create a key first
        client.post(
            "/v1/admin/keys",
            json={"name": "listed-key"},
            headers={"X-API-Key": "admin-secret"},
        )

        resp = client.get("/v1/admin/keys", headers={"X-API-Key": "admin-secret"})
        assert resp.status_code == 200, resp.text
        keys = resp.json()
        assert len(keys) >= 1
        assert any(k["name"] == "listed-key" for k in keys)

    def test_admin_rejects_non_admin_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-admin (regular) key gets 403 on admin endpoints."""
        client = self._setup_client(monkeypatch)

        # Create a regular KeyStore key using admin
        create_resp = client.post(
            "/v1/admin/keys",
            json={"name": "regular-user"},
            headers={"X-API-Key": "admin-secret"},
        )
        assert create_resp.status_code == 200
        regular_key = create_resp.json()["plaintext_key"]

        # Use the regular key on admin endpoint → should get 403
        resp = client.post(
            "/v1/admin/keys",
            json={"name": "should-fail"},
            headers={"X-API-Key": regular_key},
        )
        assert resp.status_code == 403, resp.text

    def test_admin_rejects_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Admin endpoints reject requests without key (401)."""
        client = self._setup_client(monkeypatch)

        resp = client.get("/v1/admin/keys")
        # No key header → auth middleware rejects with 401
        assert resp.status_code == 401

    def test_revoke_key_via_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete /v1/admin/keys/{key_id} revokes a key."""
        client = self._setup_client(monkeypatch)

        # Create a key
        create_resp = client.post(
            "/v1/admin/keys",
            json={"name": "to-revoke"},
            headers={"X-API-Key": "admin-secret"},
        )
        key_id = create_resp.json()["key_id"]

        # Revoke it
        resp = client.delete(
            f"/v1/admin/keys/{key_id}",
            headers={"X-API-Key": "admin-secret"},
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True

    def test_revoke_nonexistent_key_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Revoking a non-existent key returns 404."""
        client = self._setup_client(monkeypatch)

        resp = client.delete(
            "/v1/admin/keys/nonexistent-id",
            headers={"X-API-Key": "admin-secret"},
        )
        assert resp.status_code == 404

    def test_admin_disabled_when_no_admin_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Admin endpoint returns 503 when RUITONG_ADMIN_KEY is not set."""
        monkeypatch.delenv("RUITONG_ADMIN_KEY", raising=False)
        monkeypatch.delenv("RUITONG_API_KEY", raising=False)
        import ruitong.main

        importlib.reload(ruitong.main)
        ruitong.main.app.state.key_store = KeyStore(":memory:")
        ruitong.main.app.state.config = ruitong.main.BridgeConfig.from_env()

        client = TestClient(ruitong.main.app)

        resp = client.post(
            "/v1/admin/keys",
            json={"name": "any-key"},
            headers={"X-API-Key": "some-key"},
        )
        # Admin API is disabled — 503
        assert resp.status_code == 503, resp.text
        assert "not set" in resp.text


# ── Auth middleware end-to-end tests ──────────────────────────────────


class TestAuthMiddleware:
    """End-to-end auth middleware tests with KeyStore."""

    def test_without_key_returns_401_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When legacy api_key is set, missing key returns 401."""
        monkeypatch.setenv("RUITONG_API_KEY", "legacy-123")
        monkeypatch.delenv("RUITONG_ADMIN_KEY", raising=False)
        import ruitong.main

        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)

        resp = client.get("/v1/health")
        # Health is exempt — should 200
        assert resp.status_code == 200

        resp = client.post("/v1/port", json={"model": "Qwen3-8B"})
        assert resp.status_code == 401

    def test_with_valid_legacy_key_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid legacy key is accepted."""
        monkeypatch.setenv("RUITONG_API_KEY", "legacy-123")
        monkeypatch.delenv("RUITONG_ADMIN_KEY", raising=False)
        import ruitong.main

        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)

        resp = client.post(
            "/v1/port",
            json={"model": "Qwen3-8B"},
            headers={"X-API-Key": "legacy-123"},
        )
        assert resp.status_code == 200, resp.text

    def test_admin_mode_with_key_store_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KeyStore key authenticates regular endpoints."""
        monkeypatch.delenv("RUITONG_API_KEY", raising=False)
        monkeypatch.delenv("RUITONG_ADMIN_KEY", raising=False)
        import ruitong.main

        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)

        # Without admin_key, we fall back to api_key which is empty — anonymous
        resp = client.post("/v1/port", json={"model": "Qwen3-8B"})
        assert resp.status_code == 200, resp.text

    def test_with_valid_key_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid X-API-Key header is accepted."""
        monkeypatch.setenv("RUITONG_API_KEY", "test-key-123")
        monkeypatch.delenv("RUITONG_ADMIN_KEY", raising=False)
        import ruitong.main

        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)

        resp = client.post(
            "/v1/port",
            json={"model": "Qwen3-8B"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["validation_level"] == "simulated"