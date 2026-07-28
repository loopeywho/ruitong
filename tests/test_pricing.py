"""Tests for the CNY-native pricing module."""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient


class TestPricingList:
    """GET /v1/pricing — list all pricing."""

    def test_returns_empty_list_when_no_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without RUITONG_PRICING set, returns empty list."""
        monkeypatch.delenv("RUITONG_PRICING", raising=False)
        import ruitong.main
        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)
        resp = client.get("/v1/pricing")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_configured_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With RUITONG_PRICING set, returns configured pricing list."""
        pricing = json.dumps({
            "Qwen3-8B": {
                "name": "standard",
                "price_per_input_token_cny": 0.0007,
                "price_per_output_token_cny": 0.0028,
            }
        })
        monkeypatch.setenv("RUITONG_PRICING", pricing)
        import ruitong.main
        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)
        resp = client.get("/v1/pricing")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model"] == "Qwen3-8B"
        assert data[0]["currency"] == "CNY"
        assert data[0]["tier"] == "standard"


class TestPricingModel:
    """GET /v1/pricing/{model} — get one model's pricing."""

    def test_returns_model_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns pricing for the requested model."""
        pricing = json.dumps({
            "Qwen3-8B": {
                "name": "standard",
                "price_per_input_token_cny": 0.0007,
                "price_per_output_token_cny": 0.0028,
            },
            "Llama3-70B": {
                "name": "premium",
                "price_per_input_token_cny": 0.0040,
                "price_per_output_token_cny": 0.0160,
            }
        })
        monkeypatch.setenv("RUITONG_PRICING", pricing)
        import ruitong.main
        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)

        resp = client.get("/v1/pricing/Qwen3-8B")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Qwen3-8B"
        assert data["currency"] == "CNY"
        assert data["tier"] == "standard"

    def test_unknown_model_returns_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown model returns 404."""
        pricing = json.dumps({
            "Qwen3-8B": {
                "name": "standard",
                "price_per_input_token_cny": 0.0007,
                "price_per_output_token_cny": 0.0028,
            }
        })
        monkeypatch.setenv("RUITONG_PRICING", pricing)
        import ruitong.main
        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)

        resp = client.get("/v1/pricing/unknown-model")
        assert resp.status_code == 404


class TestPricingEmpty:
    """GET /v1/pricing/{model} when no pricing is configured."""

    def test_unknown_model_404_no_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without pricing configured, any model returns 404."""
        monkeypatch.delenv("RUITONG_PRICING", raising=False)
        import ruitong.main
        importlib.reload(ruitong.main)
        from fastapi.testclient import TestClient

        client = TestClient(ruitong.main.app)
        resp = client.get("/v1/pricing/any-model")
        assert resp.status_code == 404