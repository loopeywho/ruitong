from __future__ import annotations

import pytest

from ruitong.config import (
    DEFAULT_BACKEND_PRIORITY,
    DEFAULT_HEALTH_CHECK_TIMEOUT_S,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_S,
    BridgeConfig,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every RUITONG_* var so a test sees documented defaults.

    Without this, a developer's shell export silently changes test outcomes.
    """
    import os

    for key in list(os.environ):
        if key.startswith("RUITONG_"):
            monkeypatch.delenv(key, raising=False)


class TestDefaults:
    def test_defaults_when_env_is_empty(self, clean_env: None) -> None:
        config = BridgeConfig.from_env()

        assert config.cuda_base_url == ""
        assert config.ascend_base_url == ""
        assert config.auto_backend_priority == DEFAULT_BACKEND_PRIORITY
        assert config.default_max_tokens == DEFAULT_MAX_TOKENS
        assert config.health_check_timeout == DEFAULT_HEALTH_CHECK_TIMEOUT_S
        assert config.request_timeout == DEFAULT_REQUEST_TIMEOUT_S

    def test_cuda_is_tried_before_ascend(self) -> None:
        assert DEFAULT_BACKEND_PRIORITY == ("cuda", "ascend")


class TestEnvReadAtCallTime:
    """Regression guard: config must not snapshot env at import time."""

    def test_reads_backend_urls_from_env(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUITONG_CUDA_BASE_URL", "http://cuda.local:8000")
        monkeypatch.setenv("RUITONG_ASCEND_BASE_URL", "http://ascend.local:8000")

        config = BridgeConfig.from_env()

        assert config.cuda_base_url == "http://cuda.local:8000"
        assert config.ascend_base_url == "http://ascend.local:8000"

    def test_env_set_after_import_is_still_seen(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bug this replaces: dataclass defaults froze env at import."""
        before = BridgeConfig.from_env()
        assert before.cuda_base_url == ""

        monkeypatch.setenv("RUITONG_CUDA_BASE_URL", "http://late.local:8000")

        assert BridgeConfig.from_env().cuda_base_url == "http://late.local:8000"


class TestNumericOverrides:
    def test_timeouts_are_overridable(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUITONG_REQUEST_TIMEOUT_S", "12.5")
        monkeypatch.setenv("RUITONG_HEALTH_TIMEOUT_S", "2")

        config = BridgeConfig.from_env()

        assert config.request_timeout == 12.5
        assert config.health_check_timeout == 2.0

    def test_max_tokens_is_overridable(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUITONG_DEFAULT_MAX_TOKENS", "512")

        assert BridgeConfig.from_env().default_max_tokens == 512

    def test_empty_string_falls_back_to_default(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUITONG_REQUEST_TIMEOUT_S", "")

        assert BridgeConfig.from_env().request_timeout == DEFAULT_REQUEST_TIMEOUT_S

    def test_non_numeric_timeout_is_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUITONG_REQUEST_TIMEOUT_S", "soon")

        with pytest.raises(ValueError, match="RUITONG_REQUEST_TIMEOUT_S"):
            BridgeConfig.from_env()

    def test_non_integer_max_tokens_is_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUITONG_DEFAULT_MAX_TOKENS", "lots")

        with pytest.raises(ValueError, match="RUITONG_DEFAULT_MAX_TOKENS"):
            BridgeConfig.from_env()


class TestImmutability:
    def test_config_is_frozen(self, clean_env: None) -> None:
        config = BridgeConfig.from_env()

        with pytest.raises(AttributeError):
            config.cuda_base_url = "http://elsewhere"  # type: ignore[misc]

    def test_priority_is_immutable(self, clean_env: None) -> None:
        """A tuple, not a list — a shared mutable default is a footgun."""
        config = BridgeConfig.from_env()

        assert isinstance(config.auto_backend_priority, tuple)
