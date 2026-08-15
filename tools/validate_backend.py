#!/usr/bin/env python3
"""Validate a backend's OpenAI-compatible API for pre-flight checks.

Runs 6 checks — cheapest and most decisive first — then exits:
  0 = usable
  1 = unusable (the port is broken)
  2 = inconclusive (could not tell — e.g. vllm-ascend#7218 bug masks token identity)

Checks:
  1. /v1/models responds and lists the expected model.
  2. Single request with logprobs=true, top_logprobs=5 returns populated logprobs.
  3. Token identity usable — top_logprobs rows not all degenerate (imported from
     equivalence/metrics.count_degenerate_token_rows).
  4. No sentinel/non-finite values (-9999, -inf, NaN).  Count reported (imported
     from equivalence/metrics.count_non_finite).
  5. Probability mass plausible — sum(exp(logprob)) over top-k in (0, 1].
  6. Warm-repeat determinism — 3 identical prompts, discard first, calls 2 and 3
     are bit-identical.

Usage:
    python tools/validate_backend.py --endpoint http://localhost:8000 --model Qwen3-8B
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import httpx

# ── Import calibrated metrics from the equivalence package ────────────────────

# The metrics module lives in src/ruitong/equivalence/metrics.py.
# We compute the absolute path from this script's location.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent  # ruitong-bridge/
_SRC = _PROJECT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ruitong.equivalence.metrics import (
    count_degenerate_token_rows,
    count_non_finite,
)

# ── Result code constants ─────────────────────────────────────────────────────
PASS = 0
FAIL = 1
INCONCLUSIVE = 2


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chat_endpoint(endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/v1/chat/completions"


def _extract_top_logprob_tokens(content: list[dict]) -> list[list[str]]:
    """Extract top-k token strings from logprobs content."""
    rows: list[list[str]] = []
    for entry in content:
        tops = entry.get("top_logprobs", [])
        rows.append([t["token"] for t in tops])
    return rows


# ── Check 1: /v1/models responds ──────────────────────────────────────────────


def check_models(endpoint: str, model: str) -> tuple[int, str]:
    url = f"{endpoint.rstrip('/')}/v1/models"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (FAIL, f"Check 1 (/v1/models) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (FAIL, f"Check 1 (/v1/models) failed: {exc}")

    body = resp.json()
    models = [m["id"] for m in body.get("data", [])]
    if model not in models:
        return (FAIL, f"Check 1 (/v1/models) failed: model '{model}' not found in {models}")
    return (PASS, f"Check 1 passed: model '{model}' found among {len(models)} models")


# ── Check 2: logprobs populated ───────────────────────────────────────────────


def check_logprobs_populated(endpoint: str, model: str) -> tuple[int, str]:
    url = _chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say something interesting."}],
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (FAIL, f"Check 2 (logprobs) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (FAIL, f"Check 2 (logprobs) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (FAIL, "Check 2 (logprobs) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (FAIL, "Check 2 (logprobs) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])
    if not content:
        return (FAIL, "Check 2 (logprobs) failed: empty content")

    # At least one position should have top_logprobs
    has_top = any(entry.get("top_logprobs") for entry in content)
    if not has_top:
        return (FAIL, "Check 2 (logprobs) failed: no top_logprobs in any position")

    return (PASS, f"Check 2 passed: {len(content)} positions, top_logprobs present")


# ── Check 3: Token identity usable ────────────────────────────────────────────
# Uses count_degenerate_token_rows from equivalence/metrics.py.


def check_token_identity(endpoint: str, model: str) -> tuple[int, str]:
    url = _chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say something interesting."}],
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (FAIL, f"Check 3 (token identity) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (FAIL, f"Check 3 (token identity) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (FAIL, "Check 3 (token identity) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (FAIL, "Check 3 (token identity) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])
    if not content:
        return (FAIL, "Check 3 (token identity) failed: empty content")

    token_rows = _extract_top_logprob_tokens(content)

    if not token_rows:
        return (FAIL, "Check 3 (token identity) failed: no token data")

    degenerate = count_degenerate_token_rows(token_rows)
    total_rows = len(token_rows)

    # Check for synthetic token_id:N references (vllm-ascend#7218)
    synthetic_refs = sum(
        1 for row in token_rows for t in row if t.startswith("token_id:")
    )

    if degenerate > 0:
        # vllm-ascend#7218: top_logprobs token IDs are degenerate (all show the
        # sampled token's ID) but the logprob VALUES are correct. When synthetic
        # token_id:N refs are found, the backend is probably usable — we just
        # can't tell from token identity alone.
        if synthetic_refs > 0:
            return (
                INCONCLUSIVE,
                f"Check 3 inconclusive: {degenerate}/{total_rows} rows degenerate "
                f"({synthetic_refs} token_id:N refs — known vllm-ascend#7218 bug, "
                f"logprob values likely correct)",
            )
        return (
            FAIL,
            f"Check 3 (token identity) failed: {degenerate}/{total_rows} "
            f"rows are degenerate (all identical tokens), "
            f"{synthetic_refs} synthetic token_id:N refs found",
        )
    return (
        PASS,
        f"Check 3 passed: {total_rows} rows, {degenerate} degenerate, "
        f"{synthetic_refs} synthetic refs",
    )


# ── Check 4: no sentinel/non-finite values ────────────────────────────────────
# Uses count_non_finite from equivalence/metrics.py.


def check_no_sentinels(endpoint: str, model: str) -> tuple[int, str]:
    url = _chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say something interesting."}],
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (FAIL, f"Check 4 (sentinel values) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (FAIL, f"Check 4 (sentinel values) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (FAIL, "Check 4 (sentinel values) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (FAIL, "Check 4 (sentinel values) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])

    # Flatten all logprob values (top-k per position)
    all_logs: list[float] = []
    for entry in content:
        for t in entry.get("top_logprobs", []):
            all_logs.append(t["logprob"])
        # Also the chosen token
        all_logs.append(entry["logprob"])

    if not all_logs:
        return (PASS, "Check 4 passed: no logprob values to check")

    non_finite = count_non_finite([all_logs])

    # Also flag the specific -9999 sentinel (vllm#19305)
    sentinel_9999 = sum(1 for v in all_logs if v == -9999)

    if non_finite > 0 or sentinel_9999 > 0:
        return (
            FAIL,
            f"Check 4 (sentinel values) failed: {non_finite} non-finite, "
            f"{sentinel_9999} sentinels (-9999)",
        )
    return (PASS, f"Check 4 passed: {len(all_logs)} values, all finite")


# ── Check 5: probability mass plausible ───────────────────────────────────────


def check_probability_mass(endpoint: str, model: str) -> tuple[int, str]:
    url = _chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say something interesting."}],
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (FAIL, f"Check 5 (probability mass) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (FAIL, f"Check 5 (probability mass) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (FAIL, "Check 5 (probability mass) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (FAIL, "Check 5 (probability mass) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])
    if not content:
        return (FAIL, "Check 5 (probability mass) failed: empty content")

    # For each position, sum exp(logprob) over top-k entries
    bad_positions: list[int] = []
    for pos, entry in enumerate(content):
        tops = entry.get("top_logprobs", [])
        mass = 0.0
        for t in tops:
            lp = t["logprob"]
            if lp < 64.0:  # guard against overflow
                mass += math.exp(lp)
            else:
                mass += math.exp(64.0)
        if mass <= 0.0 or mass > 1.001:  # tiny tolerance above 1
            bad_positions.append(pos)

    if bad_positions:
        return (
            FAIL,
            f"Check 5 (probability mass) failed: {len(bad_positions)}/{len(content)} "
            f"positions have implausible mass (not in (0,1]): "
            f"indices {bad_positions[:5]}{'' if len(bad_positions) <= 5 else '...'}",
        )
    return (
        PASS,
        f"Check 5 passed: {len(content)} positions, all mass in (0,1]",
    )


# ── Check 6: warm-repeat determinism ──────────────────────────────────────────


def check_determinism(endpoint: str, model: str) -> tuple[int, str]:
    url = _chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say something deterministic."}],
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            # Call 1: warm-up (discarded)
            resp1 = client.post(url, json=payload)
            resp1.raise_for_status()

            # Call 2
            resp2 = client.post(url, json=payload)
            resp2.raise_for_status()

            # Call 3
            resp3 = client.post(url, json=payload)
            resp3.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (FAIL, f"Check 6 (determinism) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (FAIL, f"Check 6 (determinism) failed: {exc}")

    body2 = resp2.json()
    body3 = resp3.json()

    # Serialize to JSON for bit-comparison (stable ordering matters)
    import json as _json

    str2 = _json.dumps(body2, sort_keys=True, separators=(",", ":"))
    str3 = _json.dumps(body3, sort_keys=True, separators=(",", ":"))

    if str2 != str3:
        # Find first difference for diagnostics
        min_len = min(len(str2), len(str3))
        diff_pos = None
        for i in range(min_len):
            if str2[i] != str3[i]:
                diff_pos = i
                break
        return (
            FAIL,
            f"Check 6 (determinism) failed: calls 2 and 3 differ at byte {diff_pos} "
            f"(len2={len(str2)}, len3={len(str3)})",
        )
    return (PASS, "Check 6 passed: calls 2 and 3 are bit-identical")


# ── Main ──────────────────────────────────────────────────────────────────────

CHECKS = [
    ("1. /v1/models lists expected model", check_models),
    ("2. logprobs populated", check_logprobs_populated),
    ("3. Token identity usable", check_token_identity),
    ("4. No sentinel/non-finite values", check_no_sentinels),
    ("5. Probability mass plausible", check_probability_mass),
    ("6. Warm-repeat determinism", check_determinism),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a backend's OpenAI-compatible API for pre-flight checks."
    )
    parser.add_argument("--endpoint", required=True, help="Base URL (e.g. http://localhost:8000)")
    parser.add_argument("--model", required=True, help="Expected model ID")
    args = parser.parse_args()

    print(f"Validating backend at {args.endpoint} (model: {args.model})")
    print("-" * 60)

    has_fail = False
    has_inconclusive = False
    for label, check_fn in CHECKS:
        code, msg = check_fn(args.endpoint, args.model)
        if code == PASS:
            status = "PASS"
        elif code == INCONCLUSIVE:
            status = "INCONCLUSIVE"
            has_inconclusive = True
        else:
            status = "FAIL"
            has_fail = True
        print(f"[{status}] {label}")
        print(f"       {msg}")

    print("-" * 60)
    if not has_fail and not has_inconclusive:
        print("All checks passed — backend is usable.")
        return 0
    elif has_fail:
        print("One or more checks FAILED — backend is NOT usable.")
        return 1
    else:
        print("All checks passed or inconclusive — backend is usable with caveats.")
        return 2


if __name__ == "__main__":
    sys.exit(main())