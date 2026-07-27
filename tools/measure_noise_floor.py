"""Measure the numerical noise floor of a *correct* execution.

Why this exists
---------------
Every threshold in `Thresholds` must sit above the noise floor and below the
weakest real fault. Until now the noise floor was estimated from synthetic
bf16 rounding, because no GPU was available (CALIBRATION.md, and the standing
caveat in PHASE_4_5_PROPOSAL.md 4.5a). A threshold calibrated on synthetic
data is a guess, and a guess is exactly what this product cannot ship.

This measures it against a real server. It compares executions that are all
*correct* — same weights, same hardware, same server — and differ only in
execution path. Any spread they show is noise, not a defect. A port must be
allowed to differ by at least this much.

Three conditions, in increasing distance from the reference:

  repeat     same request, sent again, sequentially.
             Isolates pure run-to-run nondeterminism.

  batched    the same request, but issued concurrently with filler traffic so
             vLLM schedules it in a larger batch. Reduction order over the
             batch dimension changes, so sums are reassociated and results
             move. This is the closest single-vendor analogue to a hardware
             port: identical maths, different order of operations.

  reordered  the same batch, submitted in a different order.

Usage:
    RUITONG_API_KEY=... python tools/measure_noise_floor.py \
        --endpoint https://host/v1 --model Qwen/Qwen3-8B --out noise.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from typing import Any

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ruitong.equivalence.metrics import (  # noqa: E402
    probability_mass,
    top_k_max_abs_diff,
    top1_token_agreement,
    topk_token_set_agreement,
)

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain what a transformer model is, briefly.",
    "Write one line of Python that reverses a string.",
    "Name three primary colours.",
    "What is 17 multiplied by 23?",
    "Summarise the water cycle in two sentences.",
    "Translate 'good morning' into Mandarin Chinese.",
    "What does the acronym GPU stand for?",
]

FILLER = [
    "Describe the taste of salt.",
    "How many continents are there?",
    "What is the boiling point of water at sea level?",
    "List two renewable energy sources.",
    "Who wrote Hamlet?",
    "What colour is the sky on a clear day?",
    "Define the word 'concise'.",
    "What is the largest ocean?",
]


def _payload(model: str, prompt: str, max_tokens: int, top_k: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
        "logprobs": True,
        "top_logprobs": top_k,
        # Qwen3 emits <think> blocks by default, which burn the token budget on
        # reasoning we are not measuring. Disabling keeps every comparison on
        # the answer itself.
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def _one(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> dict[str, Any] | None:
    """Return {tokens, matrix} or None when the request failed."""
    try:
        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 — a failed probe is data, not a crash
        print(f"  ! request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    logprobs = data["choices"][0].get("logprobs")
    if not logprobs or not logprobs.get("content"):
        print("  ! no logprobs in response", file=sys.stderr)
        return None

    content = logprobs["content"]
    return {
        "tokens": [entry["token"] for entry in content],
        # top_logprobs, ranked as the server returned them
        "matrix": [
            [t["logprob"] for t in entry.get("top_logprobs", [])] for entry in content
        ],
        "top_tokens": [
            [t["token"] for t in entry.get("top_logprobs", [])] for entry in content
        ],
        "text": data["choices"][0]["message"].get("content", ""),
    }


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two runs using the product's own gate metrics."""
    # Token sequences can diverge outright. Everything past the first
    # divergence is a different continuation, so comparing it measures nothing;
    # truncate and report the divergence point separately.
    limit = min(len(a["tokens"]), len(b["tokens"]))
    divergence: int | None = None
    for i in range(limit):
        if a["tokens"][i] != b["tokens"][i]:
            divergence = i
            break
    usable = limit if divergence is None else divergence

    result: dict[str, Any] = {
        "positions_compared": usable,
        "token_sequence_identical": divergence is None
        and len(a["tokens"]) == len(b["tokens"]),
        "first_divergent_position": divergence,
        "text_identical": a["text"] == b["text"],
    }
    if usable == 0:
        return result

    ma, mb = a["matrix"][:usable], b["matrix"][:usable]
    ta, tb = a["top_tokens"][:usable], b["top_tokens"][:usable]
    result["topk_max_abs_diff"] = top_k_max_abs_diff(ma, mb, k=10)
    result["probability_mass_delta"] = abs(probability_mass(ma) - probability_mass(mb))
    result["top1_agreement"] = top1_token_agreement(
        [row[0] for row in ta], [row[0] for row in tb]
    )
    result["top5_set_agreement"] = topk_token_set_agreement(ta, tb, k=5)
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("RUITONG_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = args.endpoint.rstrip("/") + "/chat/completions"

    conditions: dict[str, list[dict[str, Any] | None]] = {}

    async with httpx.AsyncClient(timeout=300.0, headers=headers) as client:
        # ── reference: strictly one request in flight at a time ──────
        print("condition: reference (sequential, batch size 1)")
        reference = []
        for prompt in PROMPTS:
            reference.append(
                await _one(client, url, _payload(args.model, prompt, args.max_tokens, args.top_k))
            )
        conditions["reference"] = reference

        # ── repeat: identical protocol, run again ────────────────────
        print("condition: repeat (sequential, batch size 1, second pass)")
        repeat = []
        for prompt in PROMPTS:
            repeat.append(
                await _one(client, url, _payload(args.model, prompt, args.max_tokens, args.top_k))
            )
        conditions["repeat"] = repeat

        # ── batched: same prompts, scheduled alongside filler ────────
        print(f"condition: batched (concurrent with {len(FILLER)} filler requests)")
        tasks = [
            _one(client, url, _payload(args.model, p, args.max_tokens, args.top_k))
            for p in PROMPTS + FILLER
        ]
        batched_all = await asyncio.gather(*tasks)
        conditions["batched"] = list(batched_all[: len(PROMPTS)])

        # ── reordered: same concurrent batch, submitted back-to-front ─
        print("condition: reordered (same batch, reversed submission order)")
        pairs = list(enumerate(PROMPTS + FILLER))[::-1]
        tasks_r = [
            _one(client, url, _payload(args.model, p, args.max_tokens, args.top_k))
            for _, p in pairs
        ]
        gathered = await asyncio.gather(*tasks_r)
        restored: list[dict[str, Any] | None] = [None] * len(pairs)
        for (original_index, _), value in zip(pairs, gathered):
            restored[original_index] = value
        conditions["reordered"] = restored[: len(PROMPTS)]

    # ── compare every condition against the reference ────────────────
    report: dict[str, Any] = {
        "model": args.model,
        "endpoint_host": args.endpoint.split("/")[2] if "//" in args.endpoint else "",
        "max_tokens": args.max_tokens,
        "top_k_requested": args.top_k,
        "prompts": len(PROMPTS),
        "conditions": {},
    }

    for name in ("repeat", "batched", "reordered"):
        rows = []
        for index, prompt in enumerate(PROMPTS):
            a, b = conditions["reference"][index], conditions[name][index]
            if a is None or b is None:
                continue
            row = _compare(a, b)
            row["prompt"] = prompt
            rows.append(row)

        def _worst(key: str, mode: str) -> float | None:
            values = [r[key] for r in rows if r.get(key) is not None]
            if not values:
                return None
            return max(values) if mode == "max" else min(values)

        diffs = [r["topk_max_abs_diff"] for r in rows if r.get("topk_max_abs_diff") is not None]
        report["conditions"][name] = {
            "prompts_compared": len(rows),
            "token_sequences_identical": sum(
                1 for r in rows if r["token_sequence_identical"]
            ),
            "texts_identical": sum(1 for r in rows if r["text_identical"]),
            "worst_topk_max_abs_diff": _worst("topk_max_abs_diff", "max"),
            "median_topk_max_abs_diff": statistics.median(diffs) if diffs else None,
            "worst_probability_mass_delta": _worst("probability_mass_delta", "max"),
            "worst_top1_agreement": _worst("top1_agreement", "min"),
            "worst_top5_set_agreement": _worst("top5_set_agreement", "min"),
            "per_prompt": rows,
        }

    print()
    print("=== Measured noise floor (all conditions are CORRECT executions) ===")
    header = f"{'condition':<12}{'worst topk diff':>18}{'worst mass δ':>15}{'worst top-1':>13}{'identical text':>16}"
    print(header)
    print("-" * len(header))
    for name, summary in report["conditions"].items():
        def _fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.6f}"

        print(
            f"{name:<12}{_fmt(summary['worst_topk_max_abs_diff']):>18}"
            f"{_fmt(summary['worst_probability_mass_delta']):>15}"
            f"{_fmt(summary['worst_top1_agreement']):>13}"
            f"{str(summary['texts_identical']) + '/' + str(summary['prompts_compared']):>16}"
        )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
