"""Calibrate the equivalence gate against a REAL captured corpus.

Supersedes the synthetic calibration in CALIBRATION.md. Everything here runs
offline against `corpora/*.json` — real Qwen3-8B logprobs pulled from an
NVIDIA A40 — so thresholds are derived from model output rather than from a
fixture that happened to have a convenient distribution.

The question a threshold has to answer is always the same: is there a gap
between how far two CORRECT executions drift and how far the WEAKEST real
fault moves? If yes, a threshold can live in the gap. If no, the metric is
unusable no matter what number is chosen.

Usage:
    python tools/calibrate_from_corpus.py corpora/cuda_a40_qwen3_8b.json
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ruitong.equivalence import faults as F  # noqa: E402
from ruitong.equivalence.metrics import (  # noqa: E402
    probability_mass,
    token_matched_prob_diff,
    top_k_max_abs_diff,
    top1_token_agreement,
    topk_token_set_agreement,
)


def _shift(rows: list[list[float]], by: int = 1) -> list[list[float]]:
    return rows[by:] + rows[:by] if len(rows) > by else rows


def _swap(rows: list[list[float]], i: int = 0, j: int = 1) -> list[list[float]]:
    out = []
    for row in rows:
        new = list(row)
        if len(new) > max(i, j):
            new[i], new[j] = new[j], new[i]
        out.append(new)
    return out


def _scale(rows: list[list[float]], factor: float) -> list[list[float]]:
    return [[v * factor for v in row] for row in rows]


def _corrupt(rows: list[list[float]], every: int = 8, delta: float = 5.0) -> list[list[float]]:
    return [
        [v - delta for v in row] if index % every == 0 else list(row)
        for index, row in enumerate(rows)
    ]


def _drop_top(rows: list[list[float]], floor: float = -30.0) -> list[list[float]]:
    """Argmax demoted below the tail — the single most severe port failure."""
    out = []
    for row in rows:
        new = list(row)
        if new:
            new[0] = floor
        out.append(new)
    return out


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "corpora/cuda_a40_qwen3_8b.json"
    corpus = json.load(open(path))
    entries = corpus["entries"]

    print(f"corpus:  {path}")
    print(f"model:   {corpus['model']}  ({corpus['label']})")
    print(f"prompts: {len(entries)}   bit-exact warm: {corpus['reproducible_when_warm']}/{corpus['total']}")
    print()

    # Conditions. Each returns (tokens, logprobs) for the candidate side.
    def identical(t, l):
        return t, l

    def bf16(t, l):
        return t, [[F.bfloat16_round(v) for v in row] for row in l]

    conditions = {
        "identical (perfect port)": identical,
        "bf16 rounding (correct port, other precision)": bf16,
        "-- faults below this line --": None,
        "scale x1.05 (softmax/temperature bug)": lambda t, l: (t, _scale(l, 1.05)),
        "scale x1.01 (subtle temperature bug)": lambda t, l: (t, _scale(l, 1.01)),
        "swap top-2 (transposed operator output)": lambda t, l: (t, _swap(l, 0, 1)),
        "shift positions by 1 (KV-cache off-by-one)": lambda t, l: (_shift(t), _shift(l)),
        "corrupt 1 position in 8 (intermittent kernel)": lambda t, l: (t, _corrupt(l)),
        "demote argmax (catastrophic port failure)": lambda t, l: (t, _drop_top(l)),
    }

    header = (
        f"{'condition':<48}{'tok-matched Δp':>16}{'old topk Δlogp':>16}"
        f"{'mass δ':>12}{'top-1':>8}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, float]] = {}
    for name, transform in conditions.items():
        if transform is None:
            print("-" * len(header))
            continue

        worst_new = 0.0
        worst_old = 0.0
        worst_mass = 0.0
        worst_top1 = 1.0

        for entry in entries:
            tokens = entry["top_tokens"]
            logs = entry["top_logprobs"]
            cand_tokens, cand_logs = transform(tokens, logs)
            n = min(len(tokens), len(cand_tokens))
            if n == 0:
                continue
            ta, la = tokens[:n], logs[:n]
            tb, lb = cand_tokens[:n], cand_logs[:n]

            worst_new = max(worst_new, token_matched_prob_diff(ta, la, tb, lb, k=10))
            worst_old = max(worst_old, top_k_max_abs_diff(la, lb, k=10))
            worst_mass = max(worst_mass, abs(probability_mass(la) - probability_mass(lb)))
            worst_top1 = min(worst_top1, top1_token_agreement(ta, tb))

        results[name] = {
            "token_matched_prob_diff": worst_new,
            "top_k_max_abs_diff": worst_old,
            "probability_mass_delta": worst_mass,
            "top1_agreement": worst_top1,
        }
        print(
            f"{name:<48}{worst_new:>16.3e}{worst_old:>16.5f}"
            f"{worst_mass:>12.5f}{worst_top1:>8.3f}"
        )

    # ── Can a threshold live in the gap? ─────────────────────────────
    correct = ["identical (perfect port)", "bf16 rounding (correct port, other precision)"]
    faulty = [k for k in results if k not in correct]

    print()
    print("=== Separation analysis ===")
    for metric, direction in (
        ("token_matched_prob_diff", "max"),
        ("top_k_max_abs_diff", "max"),
    ):
        noise = max(results[c][metric] for c in correct)
        weakest = min(results[f][metric] for f in faulty)
        ratio = (weakest / noise) if noise > 0 else math.inf
        verdict = "USABLE" if weakest > noise else "UNUSABLE — faults hide inside the noise"
        print(f"\n{metric}")
        print(f"  noise ceiling (worst correct port): {noise:.6e}")
        print(f"  weakest fault:                      {weakest:.6e}")
        print(f"  separation:                         {ratio:.1f}x   -> {verdict}")
        if weakest > noise:
            suggested = math.sqrt(noise * weakest) if noise > 0 else weakest / 10
            print(f"  geometric-mean threshold:           {suggested:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
