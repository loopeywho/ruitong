"""Capture a real logprob corpus from a live server, once, to disk.

GPU time is the scarce resource; analysis is free. This pulls a warm,
reproducible corpus so every subsequent calibration experiment — fault
injection, threshold search, metric comparison — runs offline against real
model output instead of synthetic fixtures.

Warm matters. A prompt's first execution differs from later ones (prefix
cache); once warm, this server is bit-exact reproducible. Each prompt is
therefore sent twice and only the second response is kept, with the pair's
agreement recorded so the corpus carries proof it was captured warm.

Usage:
    RUITONG_API_KEY=... python tools/capture_corpus.py \
        --endpoint https://host/v1 --model Qwen/Qwen3-8B --out corpus.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys

import httpx

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain what a transformer model is, briefly.",
    "Write one line of Python that reverses a string.",
    "Name three primary colours.",
    "What is 17 multiplied by 23?",
    "Summarise the water cycle in two sentences.",
    "Translate 'good morning' into Mandarin Chinese.",
    "What does the acronym GPU stand for?",
    "List the first five prime numbers.",
    "In one sentence, what is photosynthesis?",
    "Give the chemical symbol for gold.",
    "Explain recursion to a beginner in two sentences.",
    "用一句话解释什么是机器学习。",
    "What year did the Apollo 11 mission land on the Moon?",
    "Write a haiku about rain.",
    "What is the difference between a list and a tuple in Python?",
    # ── Extension to ~60 (R5, 2026-07-28). The 16 prompts above are frozen —
    # past corpora were captured with exactly that list, and changing any of
    # them silently breaks comparability. Additions are biased toward genuine
    # near-ties (open-ended generation, arbitrary-order lists, subjective
    # description), because D10 showed that is where cross-silicon divergence
    # actually happens: 3/16 diverged, all three on open-ended prompts.
    # ── open-ended generation / creative (highest divergence pressure) ──
    "Write a two-line poem about the sea.",
    "Describe the smell of fresh bread in one sentence.",
    "Invent a name for a coffee shop and explain it in one sentence.",
    "Write the opening sentence of a detective novel.",
    "Give a metaphor for time, in one sentence.",
    "Describe autumn to someone who has never seen it, in two sentences.",
    "Write a one-sentence toast for a wedding.",
    "Suggest a title for a documentary about ants.",
    # ── arbitrary-order lists (near-ties by construction) ──
    "Name four fruits.",
    "List three European capital cities.",
    "Name two famous physicists.",
    "List three uses for a paperclip.",
    "Name three programming languages.",
    "List four animals commonly kept as pets.",
    # ── short factual (control group — should rarely diverge) ──
    "What is the boiling point of water in Celsius at sea level?",
    "How many days are there in a leap year?",
    "What planet is known as the Red Planet?",
    "What is the chemical formula for water?",
    "Who wrote Romeo and Juliet?",
    "What is the square root of 144?",
    # ── code generation ──
    "Write a Python function that returns the factorial of n.",
    "Write one line of Python that sums a list of numbers.",
    "Show a SQL query that counts rows in a table called orders.",
    "Write a Python list comprehension that squares the numbers 1 to 10.",
    "What does the following do: `x[::-1]` in Python? One sentence.",
    # ── multi-step reasoning ──
    "A train leaves at 09:10 and arrives at 11:45. How long is the journey?",
    "If a shirt costs $20 after a 20% discount, what was the original price?",
    "I have 3 boxes with 12 eggs each and use 7 eggs. How many remain?",
    "Which is larger: 2 to the power of 10, or 10 to the power of 3?",
    "If today is Wednesday, what day is it in 100 days?",
    # ── 中文 ──
    "用两句话解释什么是区块链。",
    "写一首关于秋天的五言绝句。",
    "列举三个中国的省份。",
    "用一句话描述早茶文化。",
    "把「熟能生巧」翻译成英文并解释。",
    "简述粤语和普通话的一个主要区别。",
    # ── translation (both directions) ──
    "Translate 'The weather is beautiful today' into French.",
    "Translate '我今天很忙' into English.",
    "Translate 'thank you very much' into Japanese.",
    "Translate 'Der Hund schläft unter dem Tisch' into English.",
    # ── longer generation (more positions per prompt) ──
    "Explain in three sentences why the sky is blue.",
    "Summarise the plot of Cinderella in three sentences.",
    "Describe how to make a cup of tea, step by step, in four steps.",
    "Explain the difference between weather and climate in two sentences.",
    "Give three tips for learning a new language, one sentence each.",
]


async def _fetch(
    client: httpx.AsyncClient, url: str, model: str, prompt: str,
    max_tokens: int, top_k: int,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
        "logprobs": True,
        "top_logprobs": top_k,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # RunPod's proxy intermittently drops a route and returns 404 while the
    # pod itself is healthy (observed 2026-07-27). A capture is minutes of GPU
    # time, so a single blip must not discard it.
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = await client.post(url, json=body)
            response.raise_for_status()
            break
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last = exc
            print(f"    retry {attempt + 1}/5 after {type(exc).__name__}", file=sys.stderr)
            await asyncio.sleep(3.0 * (attempt + 1))
    else:
        raise RuntimeError(f"endpoint unreachable after 5 attempts: {last}")

    data = response.json()
    choice = data["choices"][0]
    content = choice["logprobs"]["content"]
    return {
        "sampled_tokens": [e["token"] for e in content],
        "top_tokens": [[t["token"] for t in e["top_logprobs"]] for e in content],
        "top_logprobs": [[t["logprob"] for t in e["top_logprobs"]] for e in content],
        "text": choice["message"].get("content", ""),
        "finish_reason": choice.get("finish_reason"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--label", default="cuda-a40")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    key = os.environ.get("RUITONG_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = args.endpoint.rstrip("/") + "/chat/completions"

    entries = []
    async with httpx.AsyncClient(timeout=300.0, headers=headers) as client:
        for index, prompt in enumerate(PROMPTS, 1):
            # Three fetches, not two. The first execution of a novel prompt
            # differs from every later one (prefix cache), so comparing #1 to
            # #2 measures the cache, not reproducibility. Discard #1, then
            # check #2 against #3 — both warm.
            await _fetch(client, url, args.model, prompt, args.max_tokens, args.top_k)
            warm_a = await _fetch(
                client, url, args.model, prompt, args.max_tokens, args.top_k
            )
            final = await _fetch(
                client, url, args.model, prompt, args.max_tokens, args.top_k
            )
            reproducible = warm_a["top_logprobs"] == final["top_logprobs"]
            entries.append({"prompt": prompt, "reproducible_when_warm": reproducible, **final})
            flag = "ok " if reproducible else "VARIED"
            print(f"[{index:>2}/{len(PROMPTS)}] {flag} {len(final['sampled_tokens']):>3} tok  {prompt[:48]}")

    corpus = {
        "label": args.label,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "captured_by": platform.node(),
        "reproducible_when_warm": sum(1 for e in entries if e["reproducible_when_warm"]),
        "total": len(entries),
        "entries": entries,
    }
    with open(args.out, "w") as handle:
        json.dump(corpus, handle)
    size_kb = os.path.getsize(args.out) // 1024
    print(
        f"\n{corpus['reproducible_when_warm']}/{corpus['total']} bit-exact on the warm repeat"
    )
    print(f"written to {args.out} ({size_kb} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
