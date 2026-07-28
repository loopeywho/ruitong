#!/usr/bin/env python3
"""Phase N audit — calls Claude Opus 5 via OpenRouter API, writes QA_FINDINGS.md.

Usage: uv run python scripts/phase_audit.py [--phase N]

Persists in the repo so the Finn loop is self-contained. Reads source files,
builds an audit prompt, calls Opus 5, and writes findings to QA_FINDINGS.md.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL = "anthropic/claude-opus-5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 32000
FINDINGS_FILE = REPO_ROOT / "QA_FINDINGS.md"

# Tokens to reserve for the response
RESPONSE_BUDGET = 20000
MODEL_MAX = 200000  # Opus 5 context window
ENCODING = "o200k_base"  # claude models use o200k


def get_api_key() -> str:
    # Try env first, then common key files
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    # Check ~/.hermes/.env
    hermes_env = Path.home() / ".hermes" / ".env"
    if hermes_env.exists():
        for line in hermes_env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Set it in env or ~/.hermes/.env"
    )


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding(ENCODING)
    return len(enc.encode(text))


def build_prompt() -> str:
    sections = []
    for ext in (".py", ".toml", ".md"):
        for f in sorted(REPO_ROOT.rglob(f"*{ext}")):
            rel = f.relative_to(REPO_ROOT)
            if any(
                part.startswith(".") or part == "__pycache__" or part == "node_modules"
                for part in rel.parts
            ):
                continue
            if rel.name in ("QA_FINDINGS.md", "QA_PHASE4.md", "requirements.txt"):
                continue
            sections.append(f"## File: {rel}")
            sections.append(f.read_text())

    # Get real test output
    test_output = "(pytest and mypy output not available)"
    try:
        import subprocess
        result = subprocess.run(
            ["uv", "run", "--extra", "dev", "pytest", "-q", "--tb=short", "-Wignore"],
            capture_output=True, text=True, timeout=60, cwd=REPO_ROOT,
        )
        test_output = result.stdout + result.stderr
    except Exception as e:
        test_output = f"(error running tests: {e})"

    prompt = (
        "You are auditing Ruitong Bridge, a CANN/CUDA bridge middleware for the Chinese AI market. "
        "The codebase below is Phase 5 (async REST API: key management + CNY pricing + auth middleware). "
        "Audit for: correctness, security, edge cases, test adequacy, and code quality. "
        "For each finding, include: severity (P1/P2/P3), the exact file path, "
        "the function name, what's wrong, and how to fix it. "
        "Be precise about line numbers by reading the actual files. "
        "If you cannot determine the exact line number, say 'line ~N' rather than fabricating. "
        "Verdict: PASS, CONDITIONAL PASS, or FAIL. "
        "Key areas: multi-key auth with HMAC-SHA256 key hashing, SQLite-backed KeyStore, "
        "admin API (CRUD keys), auth middleware, and CNY pricing module. "
        "Check that security-sensitive operations (admin key verification, key hashing, "
        "rate limiting) are robust and that pricing endpoint has proper auth.\n\n"
        "Codebase:\n\n"
        + "\n".join(sections)
        + "\n\n## Test & lint output\n```\n"
        + test_output
        + "\n```"
    )
    return prompt


def trim_to_budget(prompt: str, budget: int) -> str:
    """Trim the prompt to leave room for the response."""
    tok = count_tokens(prompt)
    if tok <= budget:
        return prompt
    # Trim by dropping source files from the end (keep the instruction)
    enc = tiktoken.get_encoding(ENCODING)
    tokens = enc.encode(prompt)
    # Keep the first 500 tokens (instruction), then trim the rest
    head = tokens[:500]
    tail = tokens[500:budget]
    return enc.decode(head + tail)


def call_opus5(prompt: str) -> dict[str, Any]:
    api_key = get_api_key()
    trimmed = trim_to_budget(prompt, MODEL_MAX - RESPONSE_BUDGET)
    print(f"Prompt tokens: {count_tokens(trimmed)} / budget {MODEL_MAX - RESPONSE_BUDGET}")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": trimmed}],
        "max_tokens": MAX_TOKENS,
    }

    with httpx.Client(timeout=300) as client:
        resp = client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def extract_content(data: dict[str, Any]) -> str:
    """Extract text content from Opus 5 response, handling thinking mode."""
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content")

    if content is None:
        # Opus 5 thinking mode — content is null, reasoning is in a separate field
        reasoning = message.get("reasoning", "")
        # Sometimes the actual text is embedded in the reasoning field
        if reasoning:
            return reasoning
        raise ValueError("Opus 5 returned null content with no reasoning field")

    if isinstance(content, str):
        return content

    # Content is a list of content blocks (text + thinking)
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                parts.append(f"[thinking: {block.get('thinking', '')}]")
        elif isinstance(block, str):
            parts.append(block)
    if parts:
        return "\n".join(parts)

    raise ValueError(f"Unexpected content format: {type(content)}: {content}")


def main() -> int:
    print(f"Building audit prompt...")
    prompt = build_prompt()
    print(f"Calling {MODEL}...")
    t0 = time.time()
    result = call_opus5(prompt)
    elapsed = time.time() - t0
    print(f"Response in {elapsed:.1f}s")

    content = extract_content(result)
    usage = result.get("usage", {})
    inp = usage.get("prompt_tokens", "?")
    out = usage.get("completion_tokens", "?")

    header = f"""# Ruitong Bridge — Phase 4 Audit (Claude Opus 5)

*Audited: {time.strftime('%Y-%m-%d %H:%M UTC')} | Model: {MODEL}*
*Input tokens: {inp} | Output tokens: {out} | Duration: {elapsed:.0f}s*
"""

    full = header + "\n" + content
    FINDINGS_FILE.write_text(full)
    print(f"Written to {FINDINGS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())