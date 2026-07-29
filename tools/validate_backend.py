#!/usr/bin/env python3
"""Validate a backend's OpenAI-compatible API for pre-flight checks.

Runs 6 checks — cheapest and most decisive first — then exits 0 usable / 1 unusable,
printing *why* a check failed.

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

TODO: R8 — refine check 3 so it distinguishes "all identical tokens" from "all
identical *decoded strings but distinct token IDs" (vllm-ascend#7218).
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

# ── Helpers ───────────────────────────────────────────────────────────────────

Result = tuple[bool, str]  # (passed, message)


def _models_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/v1/models"


def _chat_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/v1/chat/completions"


# ── Check 1: /v1/models responds and lists expected model ─────────────────────


def check_models(endpoint: str, model: str) -> Result:
    url = _models_endpoint(endpoint)
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (False, f"Check 1 (/v1/models) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (False, f"Check 1 (/v1/models) failed: {exc}")

    body = resp.json()
    # OpenAI-compatible envelope: {"object": "list", "data": [...]}
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return (False, "Check 1 (/v1/models) failed: response has no 'data' list")

    ids = [entry.get("id", "") if isinstance(entry, dict) else str(entry) for entry in data]
    if model not in ids:
        return (
            False,
            f"Check 1 (/v1/models) failed: expected model '{model}' not found "
            f"in {ids}",
        )
    return (True, f"Check 1 passed: model '{model}' listed")


# ── Check 2: logprobs populated ───────────────────────────────────────────────


def check_logprobs_populated(endpoint: str, model: str) -> Result:
    url = _chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 4,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return (False, f"Check 2 (logprobs populated) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (False, f"Check 2 (logprobs populated) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (False, "Check 2 (logprobs populated) failed: empty choices")

    logprobs = choices[0].get("logprobs")
    if logprobs is None:
        return (False, "Check 2 (logprobs populated) failed: no logprobs in response")

    content = logprobs.get("content", [])
    if not content:
        return (False, "Check 2 (logprobs populated) failed: content list is empty")

    return (True, f"Check 2 passed: {len(content)} logprob entries returned")


# ── Check 3: token identity usable ────────────────────────────────────────────
# Uses count_degenerate_token_rows from equivalence/metrics.py.


def check_token_identity(endpoint: str, model: str) -> Result:
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
        return (
            False,
            f"Check 3 (token identity) failed: HTTP {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        return (False, f"Check 3 (token identity) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (False, "Check 3 (token identity) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (False, "Check 3 (token identity) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])
    # Extract top-k token lists per position
    token_rows: list[list[str]] = []
    for entry in content:
        tops = entry.get("top_logprobs", [])
        row_tokens = [t.get("token", "") for t in tops]
        token_rows.append(row_tokens)

    if not token_rows:
        return (False, "Check 3 (token identity) failed: no token data")

    degenerate = count_degenerate_token_rows(token_rows)
    total_rows = len(token_rows)

    # Check for synthetic token_id:N references (R8 future refinement)
    synthetic_refs = sum(
        1 for row in token_rows for t in row if t.startswith("token_id:")
    )

    if degenerate > 0:
        return (
            False,
            f"Check 3 (token identity) failed: {degenerate}/{total_rows} "
            f"rows are degenerate (all identical tokens), "
            f"{synthetic_refs} synthetic token_id:N refs found",
        )
    return (
        True,
        f"Check 3 passed: {total_rows} rows, {degenerate} degenerate, "
        f"{synthetic_refs} synthetic refs",
    )


# ── Check 4: no sentinel/non-finite values ────────────────────────────────────
# Uses count_non_finite from equivalence/metrics.py.


def check_no_sentinels(endpoint: str, model: str) -> Result:
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
        return (
            False,
            f"Check 4 (sentinel values) failed: HTTP {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        return (False, f"Check 4 (sentinel values) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (False, "Check 4 (sentinel values) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (False, "Check 4 (sentinel values) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])

    # Flatten all logprob values (top-k per position)
    all_logs: list[float] = []
    for entry in content:
        for t in entry.get("top_logprobs", []):
            all_logs.append(t["logprob"])
        # Also the chosen token
        all_logs.append(entry["logprob"])

    if not all_logs:
        return (True, "Check 4 passed: no logprob values to check")

    non_finite = count_non_finite([all_logs])

    # Also flag the specific -9999 sentinel (vllm#19305)
    sentinel_9999 = sum(1 for v in all_logs if v == -9999)

    if non_finite > 0 or sentinel_9999 > 0:
        return (
            False,
            f"Check 4 (sentinel values) failed: {non_finite} non-finite values "
            f"(-inf/NaN) + {sentinel_9999} sentinel -9999 values "
            f"among {len(all_logs)} entries",
        )
    return (True, f"Check 4 passed: {len(all_logs)} values, 0 non-finite")


# ── Check 5: probability mass plausible ───────────────────────────────────────


def check_probability_mass(endpoint: str, model: str) -> Result:
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
        return (
            False,
            f"Check 5 (probability mass) failed: HTTP {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        return (False, f"Check 5 (probability mass) failed: {exc}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        return (False, "Check 5 (probability mass) failed: empty choices")

    logprobs_obj = choices[0].get("logprobs")
    if logprobs_obj is None:
        return (False, "Check 5 (probability mass) failed: no logprobs in response")

    content = logprobs_obj.get("content", [])
    if not content:
        return (False, "Check 5 (probability mass) failed: empty content")

    # For each position, sum exp(logprob) over top-k entries
    bad_positions: list[int] = []
    sums: list[float] = []
    for pos, entry in enumerate(content):
        tops = entry.get("top_logprobs", [])
        mass = 0.0
        for t in tops:
            lp = t["logprob"]
            if lp < 64.0:  # guard against overflow
                mass += math.exp(lp)
            else:
                mass += math.exp(64.0)
        sums.append(mass)
        if mass <= 0.0 or mass > 1.001:  # tiny tolerance above 1
            bad_positions.append(pos)

    if bad_positions:
        return (
            False,
            f"Check 5 (probability mass) failed: {len(bad_positions)}/{len(content)} "
            f"positions have implausible mass (not in (0,1]): "
            f"indices {bad_positions[:5]}{'' if len(bad_positions) <= 5 else '...'}, "
            f"example sums {sums[bad_positions[0]]:.6f}",
        )
    return (
        True,
        f"Check 5 passed: {len(content)} positions, all mass in (0,1]",
    )


# ── Check 6: warm-repeat determinism ──────────────────────────────────────────


def check_determinism(endpoint: str, model: str) -> Result:
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
        return (False, f"Check 6 (determinism) failed: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return (False, f"Check 6 (determinism) failed: {exc}")

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
            False,
            f"Check 6 (determinism) failed: calls 2 and 3 differ at byte {diff_pos} "
            f"(len2={len(str2)}, len3={len(str3)})",
        )
    return (True, "Check 6 passed: calls 2 and 3 are bit-identical")


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

    all_pass = True
    for label, check_fn in CHECKS:
        passed, msg = check_fn(args.endpoint, args.model)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}")
        print(f"       {msg}")
        if not passed:
            all_pass = False

    print("-" * 60)
    if all_pass:
        print("All checks passed — backend is usable.")
        return 0
    else:
        print("One or more checks FAILED — backend is NOT usable.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
