"""Tests for ruitong.errors — hierarchy, raising, catching."""

from __future__ import annotations

import pytest

from ruitong.errors import (
    BackendError,
    BackendUnavailable,
    ModelNotFound,
    RuitongError,
)


class TestErrorHierarchy:
    def test_all_inherit_from_ruitong_error(self) -> None:
        """Every domain error derives from RuitongError."""
        assert issubclass(BackendUnavailable, RuitongError)
        assert issubclass(ModelNotFound, RuitongError)
        assert issubclass(BackendError, RuitongError)

    def test_ruitong_error_is_base_exception(self) -> None:
        """RuitongError is a distinct exception, not just BaseException."""
        with pytest.raises(RuitongError):
            raise RuitongError("general")

    def test_backend_unavailable_is_ruitong_error(self) -> None:
        with pytest.raises(RuitongError):
            raise BackendUnavailable("cuda")

    def test_model_not_found_is_ruitong_error(self) -> None:
        with pytest.raises(RuitongError):
            raise ModelNotFound("m", "cuda")

    def test_backend_error_is_ruitong_error(self) -> None:
        with pytest.raises(RuitongError):
            raise BackendError("generic", Exception("oh no"))


class TestRaisingAndCatching:
    def test_backend_unavailable_raises_and_catches(self) -> None:
        with pytest.raises(BackendUnavailable) as exc_info:
            raise BackendUnavailable("cuda")
        assert "cuda" in str(exc_info.value)

    def test_model_not_found_raises_and_catches(self) -> None:
        with pytest.raises(ModelNotFound) as exc_info:
            raise ModelNotFound("qwen-7b", "ascend")
        assert "qwen-7b" in str(exc_info.value)
        assert "ascend" in str(exc_info.value)

    def test_backend_error_raises_and_catches(self) -> None:
        original = ConnectionError("connection refused")
        with pytest.raises(BackendError) as exc_info:
            raise BackendError("cuda", original)
        assert "cuda" in str(exc_info.value)
        assert exc_info.value.cause is original

    def test_catch_ruitong_error_catches_subclasses(self) -> None:
        """A bare RuitongError handler catches all domain errors."""
        with pytest.raises(RuitongError):
            raise ModelNotFound("x", "cuda")

        with pytest.raises(RuitongError):
            raise BackendUnavailable("ascend")

        with pytest.raises(RuitongError):
            raise BackendError("cuda", RuntimeError("boom"))


class TestMeaningfulMessages:
    def test_backend_unavailable_message(self) -> None:
        err = BackendUnavailable("cuda")
        msg = str(err)
        assert "cuda" in msg.lower() or "cuda" in msg

    def test_model_not_found_message(self) -> None:
        err = ModelNotFound("qwen2.5-7b-instruct", "ascend")
        msg = str(err)
        assert "qwen2.5-7b-instruct" in msg
        assert "ascend" in msg

    def test_backend_error_message(self) -> None:
        err = BackendError("cuda", RuntimeError("timeout"))
        msg = str(err)
        assert "cuda" in msg
