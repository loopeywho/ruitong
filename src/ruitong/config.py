"""Ruitong bridge configuration via environment variables.

Configuration is read when `from_env()` is called, never at import time.
Construct one instance at application startup and pass it down explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_MAX_TOKENS = 2048
DEFAULT_HEALTH_CHECK_TIMEOUT_S = 5.0
DEFAULT_REQUEST_TIMEOUT_S = 60.0
DEFAULT_BACKEND_PRIORITY = ("cuda", "ascend")


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class BridgeConfig:
    """Runtime configuration for the bridge.

    An empty base URL means "backend not configured" — the registry treats it
    as absent rather than failing startup. A dev machine legitimately has
    neither backend available.
    """

    # Base URL for the CUDA (vLLM) backend. Empty = not configured.
    cuda_base_url: str = ""

    # Base URL for the Ascend (vllm-ascend) backend. Empty = not configured.
    ascend_base_url: str = ""

    # Backend resolution order when backend="auto".
    # First healthy backend that serves the requested model wins.
    auto_backend_priority: tuple[str, ...] = DEFAULT_BACKEND_PRIORITY

    # Default maximum response tokens when a request omits max_tokens.
    default_max_tokens: int = DEFAULT_MAX_TOKENS

    # Timeout in seconds for backend health checks.
    health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT_S

    # Timeout in seconds for inference requests.
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_S

    # Legacy API key for authenticating requests. Empty = auth disabled.
    api_key: str = ""

    # Admin key for API key management endpoints. Empty = admin API disabled.
    admin_key: str = ""

    # Maximum payload size in bytes. Default 10 MB.
    max_payload_bytes: int = 10 * 1024 * 1024

    # Rate limit: max requests per IP per minute. 0 = unlimited.
    rate_limit_per_minute: int = 30

    # Path to the job persistence SQLite database. Empty = in-memory.
    job_db_path: str = ""

    # Path to the key store SQLite database. Defaults to ./ruitong-keys.db.
    key_db_path: str = ""

    # Pricing configuration (raw dicts keyed by model name).
    pricing_config: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Build a config from the current environment.

        Read at call time so tests and runtime overrides both work.
        """
        raw_pricing = os.environ.get("RUITONG_PRICING", "")
        pricing: dict[str, dict] = {}
        if raw_pricing:
            try:
                pricing = json.loads(raw_pricing)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"RUITONG_PRICING must be valid JSON, got {raw_pricing!r}"
                ) from exc

        return cls(
            cuda_base_url=os.environ.get("RUITONG_CUDA_BASE_URL", ""),
            ascend_base_url=os.environ.get("RUITONG_ASCEND_BASE_URL", ""),
            auto_backend_priority=DEFAULT_BACKEND_PRIORITY,
            default_max_tokens=_int_from_env(
                "RUITONG_DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS
            ),
            health_check_timeout=_float_from_env(
                "RUITONG_HEALTH_TIMEOUT_S", DEFAULT_HEALTH_CHECK_TIMEOUT_S
            ),
            request_timeout=_float_from_env(
                "RUITONG_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S
            ),
            api_key=os.environ.get("RUITONG_API_KEY", ""),
            admin_key=os.environ.get("RUITONG_ADMIN_KEY", ""),
            max_payload_bytes=_int_from_env(
                "RUITONG_MAX_PAYLOAD_BYTES", 10 * 1024 * 1024
            ),
            rate_limit_per_minute=_int_from_env(
                "RUITONG_RATE_LIMIT_PER_MINUTE", 30
            ),
            job_db_path=os.environ.get("RUITONG_JOB_DB_PATH", ""),
            key_db_path=os.environ.get("RUITONG_KEY_DB_PATH", ""),
            pricing_config=pricing,
        )