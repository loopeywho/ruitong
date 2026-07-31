"""Compare two captured corpora with the product's own gate.

This is `ruitong port` run offline. Both corpora were captured warm, with
identical prompts, sampling parameters and seed, so the only remaining
variable is the backend that produced them.

Capturing once and comparing offline is deliberate: GPU time is the scarce
resource, and a saved corpus can be re-analysed indefinitely as the metrics
change — which they have, three times.

Usage:
    python tools/compare_corpora.py \
        corpora/cuda_a40_qwen3_8b.json corpora/cuda_rtx6000ada_qwen3_8b.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ruitong.equivalence.metrics import (  # noqa: E402
    probability_mass,
    token_matched_prob_diff,
    top1_token_agreement,
    top_k_max_abs_diff,
    topk_token_set_agreement,
)
from ruitong.equivalence.runner import Thresholds  # noqa: E402


def _load(path: str) -> dict:
    corpus = json.load(open(path))
    if corpus.get("reproducible_when_warm") != corpus.get("total"):
        print(
            f"WARNING: {path} reports "
            f"{corpus.get('reproducible_when_warm')}/{corpus.get('total')} "
            f"bit-exact on the warm repeat. A corpus captured cold measures the "
            f"prefix cache, not the hardware.",
            file=sys.stderr,
        )
    return corpus


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    ref_path, cand_path = sys.argv[1], sys.argv[2]
    ref, cand = _load(ref_path), _load(cand_path)

    # --subset-of FILE restricts the comparison to prompts also present in an
    # earlier corpus. Needed for an honest like-for-like: the corpus was
    # widened from 16 to 61 prompts (R5) with the additions deliberately
    # biased toward near-ties, since that is where cross-silicon divergence
    # was observed. A raw rate from the wider set is therefore NOT comparable
    # to the 19% published from the original 16 — quoting it as an increase
    # would be measuring our own prompt selection.
    subset_label = ""
    if "--subset-of" in sys.argv:
        base_path = sys.argv[sys.argv.index("--subset-of") + 1]
        allowed = {e["prompt"] for e in _load(base_path)["entries"]}
        before = len(ref["entries"])
        ref["entries"] = [e for e in ref["entries"] if e["prompt"] in allowed]
        subset_label = (
            f"  [subset: {len(ref['entries'])} of {before} prompts, "
            f"those also in {os.path.basename(base_path)}]"
        )

    if ref["model"] != cand["model"]:
        print(
            f"REFUSED: different models ({ref['model']} vs {cand['model']}). "
            f"An equivalence claim across different weights is meaningless.",
            file=sys.stderr,
        )
        return 2
    if ref["label"] == cand["label"]:
        print(
            f"REFUSED: both corpora are labelled {ref['label']!r}. A backend "
            f"compared with itself always agrees and certifies nothing.",
            file=sys.stderr,
        )
        return 2

    by_prompt = {e["prompt"]: e for e in cand["entries"]}
    rows = []
    for entry in ref["entries"]:
        other = by_prompt.get(entry["prompt"])
        if other is None:
            continue
        n = min(len(entry["top_tokens"]), len(other["top_tokens"]))
        if n == 0:
            continue

        # Truncate at the first divergent SAMPLED token. Past that point the
        # two backends are continuing different sentences, so position i on one
        # side and position i on the other describe unrelated contexts and any
        # distance between them measures the divergence, not the hardware.
        # Comparing them anyway pins every diverged prompt at 1.0 and drags the
        # aggregate to a number that says nothing.
        sa, sb = entry["sampled_tokens"], other["sampled_tokens"]
        divergence: int | None = None
        for i in range(min(n, len(sa), len(sb))):
            if sa[i] != sb[i]:
                divergence = i
                break
        usable = n if divergence is None else divergence
        if usable == 0:
            # Diverged on the very first token — nothing is comparable, but the
            # divergence itself is the finding and must not be dropped silently.
            rows.append(
                {
                    "prompt": entry["prompt"],
                    "positions": 0,
                    "text_identical": False,
                    "tokens_identical": False,
                    "diverged_at": 0,
                    "token_matched_prob_diff": None,
                    "topk_max_abs_diff": None,
                    "probability_mass_delta": None,
                    "top1_agreement": None,
                    "top5_set_agreement": None,
                }
            )
            continue

        n = usable
        ta, la = entry["top_tokens"][:n], entry["top_logprobs"][:n]
        tb, lb = other["top_tokens"][:n], other["top_logprobs"][:n]
        rows.append(
            {
                "prompt": entry["prompt"],
                "positions": n,
                "text_identical": entry["text"] == other["text"],
                "tokens_identical": sa == sb,
                "diverged_at": divergence,
                "token_matched_prob_diff": token_matched_prob_diff(ta, la, tb, lb, k=10),
                "topk_max_abs_diff": top_k_max_abs_diff(la, lb, k=10),
                "probability_mass_delta": abs(probability_mass(la) - probability_mass(lb)),
                "top1_agreement": top1_token_agreement(ta, tb),
                "top5_set_agreement": topk_token_set_agreement(ta, tb, k=5),
            }
        )

    if not rows:
        print("REFUSED: no prompts in common. Nothing was compared.", file=sys.stderr)
        return 2

    # Worst case, never mean: a mean lets one catastrophic prompt average into
    # silence and makes the gate weaker the more prompts you add.
    scored = [r for r in rows if r["token_matched_prob_diff"] is not None]
    diverged = [r for r in rows if r["diverged_at"] is not None]
    if not scored:
        print("REFUSED: every prompt diverged on its first token.", file=sys.stderr)
        return 2
    worst_prob = max(r["token_matched_prob_diff"] for r in scored)
    worst_mass = max(r["probability_mass_delta"] for r in scored)
    worst_top1 = min(r["top1_agreement"] for r in scored)
    worst_top5 = min(r["top5_set_agreement"] for r in scored)
    worst_topk = max(r["topk_max_abs_diff"] for r in scored)

    print("=== Ruitong Equivalence Report (corpus comparison) ===")
    print(f"Model:      {ref['model']}")
    print(f"Reference:  {ref['label']}")
    print(f"Candidate:  {cand['label']}")
    print(f"Prompts:    {len(rows)} compared{subset_label}")
    print(
        f"Identical output text: "
        f"{sum(1 for r in rows if r['text_identical'])}/{len(rows)}"
    )
    print(f"Diverged token stream: {len(diverged)}/{len(rows)}")
    print(
        "Metrics below are computed only over positions BEFORE any divergence "
        "— past that point the two backends are continuing different sentences."
    )
    print()

    # top5_set_agreement is intentionally NOT a gate (runner.py Thresholds,
    # D12): every real cross-hardware measurement so far lands at ~0.9167,
    # just under 0.95, for reasons unrelated to correctness, and it catches
    # nothing top1_agreement doesn't already catch. Reported only.
    gate = [
        ("Token-matched prob diff [GATE]", worst_prob, "<=", Thresholds.TOKEN_MATCHED_PROB_DIFF_MAX),
        ("Prob-mass delta         [GATE]", worst_mass, "<=", Thresholds.PROB_MASS_TOLERANCE),
        ("Top-1 agreement         [GATE]", worst_top1, ">=", Thresholds.TOP1_MIN),
    ]
    passed = True
    for label, value, op, limit in gate:
        ok = value <= limit if op == "<=" else value >= limit
        passed = passed and ok
        print(f"{label}: {value:>12.9f}  {op} {limit:<10} {'PASS' if ok else 'FAIL'}")
    top5_ok = worst_top5 >= Thresholds.TOP5_MIN
    print(f"{'Top-5 set agreement     (reported)':<32}: {worst_top5:>12.9f}  >= {Thresholds.TOP5_MIN:<10} {'PASS' if top5_ok else 'below (expected, not gated)'}")
    print(f"{'Top-k max abs diff      (reported)':<32}: {worst_topk:>12.6f}")
    print()
    print(f"VERDICT: {'PASS — equivalent within calibrated tolerance' if passed else 'FAIL — not equivalent'}")
    print()

    print("Per prompt (worst first):")
    for r in sorted(rows, key=lambda r: -(r["token_matched_prob_diff"] or 0)):
        flag = "same-text" if r["text_identical"] else f"DIVERGED@{r['diverged_at']}"
        value = (
            "   n/a     " if r["token_matched_prob_diff"] is None
            else f"{r['token_matched_prob_diff']:.9f}"
        )
        print(
            f"  Δp={value}  cmp={r['positions']:>3}  {flag:<14} | {r['prompt'][:42]}"
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
