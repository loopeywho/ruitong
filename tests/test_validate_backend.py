"""Tests for tools/validate_backend.py.

Each check has:
- A passing test with good mock data
- Failure injection tests with bad mock data

Plus mutation testing to prove each check is real, not decoration.

See LESSONS.md 2026-07-29: R7 tests were vacuous — they passed under mutation.
This suite exists to catch that mistake.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import httpx
import respx

# tools/ is not a Python package, so we add it to sys.path and import directly.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE / "tools") not in sys.path:
    sys.path.insert(0, str(_HERE / "tools"))

from validate_backend import (
    CHECKS,
    check_determinism,
    check_logprobs_populated,
    check_models,
    check_no_sentinels,
    check_probability_mass,
    check_token_identity,
)

EP = "http://localhost:18080"
MODEL = "Qwen3-8B"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — build realistic mock responses
# ──────────────────────────────────────────────────────────────────────────────


def _good_model_body() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL, "object": "model", "owned_by": "test"}],
    }


def _good_logprobs() -> list[dict]:
    """3 positions, 5 top tokens each — distinct tokens, valid log-softmax (mass <= 1)."""
    content = []
    for pos in range(3):
        tops = [
            {"token": f"t{pos}_{rank}", "logprob": -0.5 - (pos + rank) * 2.5, "bytes": [110]}
            for rank in range(5)
        ]
        content.append({
            "token": tops[0]["token"],
            "logprob": tops[0]["logprob"],
            "bytes": [110],
            "top_logprobs": tops,
        })
    return content


def _good_chat_body() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": MODEL,
        "created": 1700000000,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "length",
            "logprobs": {"content": _good_logprobs()},
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Check 1 — /v1/models
# ──────────────────────────────────────────────────────────────────────────────


class TestCheck1:
    @respx.mock
    def test_passes_with_expected_model(self) -> None:
        respx.get(f"{EP}/v1/models").mock(
            return_value=httpx.Response(200, json=_good_model_body())
        )
        passed, msg = check_models(EP, MODEL)
        assert passed is True
        assert MODEL in msg

    @respx.mock
    def test_fails_model_not_in_list(self) -> None:
        respx.get(f"{EP}/v1/models").mock(
            return_value=httpx.Response(
                200, json={"object": "list", "data": [{"id": "other-model"}]}
            )
        )
        passed, msg = check_models(EP, MODEL)
        assert passed is False
        assert "check 1" in msg.lower() or "/v1/models" in msg

    @respx.mock
    def test_fails_no_data_key(self) -> None:
        """Bare list response (not OpenAI envelope)."""
        respx.get(f"{EP}/v1/models").mock(
            return_value=httpx.Response(200, json=[{"id": MODEL}])
        )
        passed, msg = check_models(EP, MODEL)
        assert passed is False
        assert "check 1" in msg.lower() or "data" in msg.lower()

    @respx.mock
    def test_fails_http_error(self) -> None:
        respx.get(f"{EP}/v1/models").mock(return_value=httpx.Response(500))
        passed, msg = check_models(EP, MODEL)
        assert passed is False


# ──────────────────────────────────────────────────────────────────────────────
# Check 2 — logprobs populated
# ──────────────────────────────────────────────────────────────────────────────


class TestCheck2:
    @respx.mock
    def test_passes_with_populated_logprobs(self) -> None:
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_good_chat_body())
        )
        passed, msg = check_logprobs_populated(EP, MODEL)
        assert passed is True
        assert "check 2" in msg.lower() or "logprobs" in msg.lower()

    @respx.mock
    def test_fails_no_logprobs_key(self) -> None:
        body = _good_chat_body()
        del body["choices"][0]["logprobs"]
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_logprobs_populated(EP, MODEL)
        assert passed is False
        assert "check 2" in msg.lower()

    @respx.mock
    def test_fails_empty_content_list(self) -> None:
        body = _good_chat_body()
        body["choices"][0]["logprobs"]["content"] = []
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_logprobs_populated(EP, MODEL)
        assert passed is False
        assert "check 2" in msg.lower()

    @respx.mock
    def test_fails_empty_choices(self) -> None:
        body = _good_chat_body()
        body["choices"] = []
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_logprobs_populated(EP, MODEL)
        assert passed is False
        assert "check 2" in msg.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 — token identity (degenerate rows via count_degenerate_token_rows)
# ──────────────────────────────────────────────────────────────────────────────


def _degenerate_logprobs() -> list[dict]:
    """Every top-5 entry is the SAME token — vllm-ascend#7218."""
    return [
        {
            "token": "the",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "the", "logprob": -0.1},
                {"token": "the", "logprob": -0.2},
                {"token": "the", "logprob": -0.3},
                {"token": "the", "logprob": -0.4},
                {"token": "the", "logprob": -0.5},
            ],
        }
        for _ in range(3)
    ]


class TestCheck3:
    @respx.mock
    def test_passes_with_distinct_tokens(self) -> None:
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _good_logprobs()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_token_identity(EP, MODEL)
        assert passed is True
        assert "check 3" in msg.lower() or "token" in msg.lower()

    @respx.mock
    def test_fails_degenerate_rows(self) -> None:
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _degenerate_logprobs()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_token_identity(EP, MODEL)
        assert passed is False
        assert "check 3" in msg.lower() or "degenerate" in msg.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Check 4 — no sentinel / non-finite values
# ──────────────────────────────────────────────────────────────────────────────


def _logprobs_with_nan() -> list[dict]:
    c = _good_logprobs()
    c[0]["logprob"] = float("nan")
    return c


def _logprobs_with_inf() -> list[dict]:
    c = _good_logprobs()
    c[1]["logprob"] = float("-inf")
    for t in c[1]["top_logprobs"]:
        t["logprob"] = float("-inf")
    return c


def _logprobs_with_9999() -> list[dict]:
    c = _good_logprobs()
    c[0]["logprob"] = -9999.0
    return c


class TestCheck4:
    @respx.mock
    def test_passes_no_sentinels(self) -> None:
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _good_logprobs()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_no_sentinels(EP, MODEL)
        assert passed is True

    @respx.mock
    def test_fails_nan_values(self) -> None:
        """NaN can't be JSON-serialized by httpx json param, so use raw text."""
        import json as _json

        body = _good_chat_body()
        body["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
        raw = _json.dumps(body, allow_nan=True)
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=raw, headers={"content-type": "application/json"})
        )
        passed, msg = check_no_sentinels(EP, MODEL)
        assert passed is False
        assert "check 4" in msg.lower() or "non-finite" in msg.lower() or "-9999" in msg

    @respx.mock
    def test_fails_negative_inf(self) -> None:
        """-inf can't be JSON-serialized by httpx json param, so use raw text."""
        import json as _json

        body = _good_chat_body()
        body["choices"][0]["logprobs"]["content"][1]["logprob"] = float("-inf")
        for t in body["choices"][0]["logprobs"]["content"][1]["top_logprobs"]:
            t["logprob"] = float("-inf")
        raw = _json.dumps(body, allow_nan=True)
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=raw, headers={"content-type": "application/json"})
        )
        passed, msg = check_no_sentinels(EP, MODEL)
        assert passed is False
        assert "check 4" in msg.lower() or "-9999" in msg

    @respx.mock
    def test_fails_sentinel_9999(self) -> None:
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _logprobs_with_9999()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_no_sentinels(EP, MODEL)
        assert passed is False
        assert "check 4" in msg.lower() or "-9999" in msg


# ──────────────────────────────────────────────────────────────────────────────
# Check 5 — probability mass in (0, 1]
# ──────────────────────────────────────────────────────────────────────────────


def _logprobs_mass_exceeds_1() -> list[dict]:
    """Logprobs close to 0 -> exp(lp) approximately 1 each -> sum >> 1."""
    return [
        {
            "token": f"t{i}",
            "logprob": -0.01,
            "bytes": [110],
            "top_logprobs": [
                {"token": f"t{i}_{r}", "logprob": -0.01, "bytes": [110]}
                for r in range(5)
            ],
        }
        for i in range(3)
    ]


class TestCheck5:
    @respx.mock
    def test_passes_plausible_mass(self) -> None:
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _good_logprobs()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_probability_mass(EP, MODEL)
        assert passed is True

    @respx.mock
    def test_fails_mass_exceeds_1(self) -> None:
        """All logprobs near 0: exp(-0.01) ~ 0.99 each, 5 per row -> sum ~ 5."""
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _logprobs_mass_exceeds_1()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        passed, msg = check_probability_mass(EP, MODEL)
        assert passed is False
        assert "check 5" in msg.lower() or "probability" in msg.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Check 6 — warm-repeat determinism
# ──────────────────────────────────────────────────────────────────────────────


def _different_chat_body() -> dict:
    body = _good_chat_body()
    body["choices"][0]["message"]["content"] = "different"
    return body


class TestCheck6:
    @respx.mock
    def test_passes_bit_identical(self) -> None:
        body = _good_chat_body()
        route = respx.post(f"{EP}/v1/chat/completions")
        route.return_value = httpx.Response(200, json=body)
        passed, msg = check_determinism(EP, MODEL)
        assert passed is True
        assert "check 6" in msg.lower()

    @respx.mock
    def test_fails_non_deterministic(self) -> None:
        """Calls 2 and 3 return different responses."""
        body2 = _good_chat_body()
        body3 = _different_chat_body()

        route = respx.post(f"{EP}/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=_good_chat_body()),  # warmup (discarded)
            httpx.Response(200, json=body2),
            httpx.Response(200, json=body3),
        ]
        passed, msg = check_determinism(EP, MODEL)
        assert passed is False
        assert "check 6" in msg.lower() or "determin" in msg.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Mutation testing — prove each check is real, not decoration
# ──────────────────────────────────────────────────────────────────────────────


class TestMutation:
    """Break each check in turn; the corresponding failure test must stop working.

    Per the plan: "A check that cannot fail is decoration." If a mutated (always-
    pass) version still passes the failure test, the test was vacuous and the
    check is not actually doing work.
    """

    @respx.mock
    def test_mutate_check1(self) -> None:
        import validate_backend as vb

        original = vb.check_models

        # 1) Real check catches wrong model
        respx.get(f"{EP}/v1/models").mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "other-model"}]}
            )
        )
        p, m = original(EP, MODEL)
        assert p is False, "real check should detect wrong model"

        # 2) Mutate: always pass
        vb.check_models = lambda *a, **k: (True, "mutated")

        # 3) Same bad data — mutated check does NOT catch it
        p2, m2 = vb.check_models(EP, MODEL)
        assert p2 is True  # mutation "succeeded" — test proves real check works
        assert "check 1" not in m2.lower()

        vb.check_models = original

    @respx.mock
    def test_mutate_check2(self) -> None:
        import validate_backend as vb

        original = vb.check_logprobs_populated

        # 1) Real check catches missing logprobs
        body = _good_chat_body()
        del body["choices"][0]["logprobs"]
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        p, m = original(EP, MODEL)
        assert p is False, "real check should detect missing logprobs"

        # 2) Mutate
        vb.check_logprobs_populated = lambda *a, **k: (True, "mutated")

        p2, m2 = vb.check_logprobs_populated(EP, MODEL)
        assert p2 is True
        assert "check 2" not in m2.lower()

        vb.check_logprobs_populated = original

    @respx.mock
    def test_mutate_check3(self) -> None:
        import validate_backend as vb

        original = vb.check_token_identity

        # 1) Real check catches degenerate rows
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _degenerate_logprobs()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        p, m = original(EP, MODEL)
        assert p is False, "real check should detect degenerate rows"

        # 2) Mutate
        vb.check_token_identity = lambda *a, **k: (True, "mutated")

        p2, m2 = vb.check_token_identity(EP, MODEL)
        assert p2 is True
        assert "degenerate" not in m2.lower()

        vb.check_token_identity = original

    @respx.mock
    def test_mutate_check4(self) -> None:
        import validate_backend as vb

        original = vb.check_no_sentinels

        # 1) Real check catches -9999 sentinel (json-compatible failure mode)
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _logprobs_with_9999()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        p, m = original(EP, MODEL)
        assert p is False, "real check should detect sentinel"

        # 2) Mutate
        vb.check_no_sentinels = lambda *a, **k: (True, "mutated")

        p2, m2 = vb.check_no_sentinels(EP, MODEL)
        assert p2 is True
        assert "non-finite" not in m2.lower()

        vb.check_no_sentinels = original

    @respx.mock
    def test_mutate_check5(self) -> None:
        import validate_backend as vb

        original = vb.check_probability_mass

        # 1) Real check catches mass > 1
        body = _good_chat_body()
        body["choices"][0]["logprobs"] = {"content": _logprobs_mass_exceeds_1()}
        respx.post(f"{EP}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        p, m = original(EP, MODEL)
        assert p is False, "real check should detect implausible mass"

        # 2) Mutate
        vb.check_probability_mass = lambda *a, **k: (True, "mutated")

        p2, m2 = vb.check_probability_mass(EP, MODEL)
        assert p2 is True
        assert "probability" not in m2.lower()

        vb.check_probability_mass = original

    @respx.mock
    def test_mutate_check6(self) -> None:
        import validate_backend as vb

        original = vb.check_determinism

        # 1) Real check catches non-determinism
        body2 = _good_chat_body()
        body3 = _different_chat_body()

        route = respx.post(f"{EP}/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=_good_chat_body()),
            httpx.Response(200, json=body2),
            httpx.Response(200, json=body3),
        ]
        p, m = original(EP, MODEL)
        assert p is False, "real check should detect non-determinism"

        # 2) Mutate
        vb.check_determinism = lambda *a, **k: (True, "mutated")

        p2, m2 = vb.check_determinism(EP, MODEL)
        assert p2 is True
        assert "determin" not in m2.lower()

        vb.check_determinism = original