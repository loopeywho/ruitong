#!/usr/bin/env python3
"""Focused Opus 5 audit on R3 — single-target self-comparison false-pass fix."""
import json, os, sys, time, subprocess
from pathlib import Path

REPO = Path.home() / "Projects/ruitong-bridge"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    env_file = Path.home() / ".hermes" / ".env"
    for line in env_file.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip().strip("\"'")
            break
if not API_KEY:
    print("No OPENROUTER_API_KEY found", file=sys.stderr)
    sys.exit(1)

# Get the R3 commit and current HEAD
R3_SHA = "e516a80"
CUR_SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()

# Read R3 diff (the commit that introduced this fix)
diff = subprocess.run(
    ["git", "diff", f"{R3_SHA}~1..{R3_SHA}", "--",
     "src/ruitong/api/__init__.py", "src/ruitong/api/router.py", "tests/test_api.py"],
    capture_output=True, text=True, cwd=REPO
).stdout

# Read current files
init_py = (REPO / "src/ruitong/api/__init__.py").read_text()
router = (REPO / "src/ruitong/api/router.py").read_text()
tests = (REPO / "tests/test_api.py").read_text()

# Run tests
test_result = subprocess.run(
    ["uv", "run", "--extra", "dev", "pytest", "-q", "--tb=short", "-Wignore", "-k", "port"],
    capture_output=True, text=True, timeout=120, cwd=REPO,
)

prompt = f"""You are auditing Ruitong Bridge R3 — removal of single-target self-comparison false pass.

R3 was committed at {R3_SHA}. Current HEAD is {CUR_SHA} (includes R2 fix on top).

## R3 diff
```diff
{diff[:5000]}
```

## Current PortRequest schema (__init__.py)
```python
{init_py[:3000]}
```

## Current router.py (_make_runner)
```python
{router[:3000]}
```

## Current test_api.py (port section)
```python
{tests[:5000]}
```

## Test output (port tests only)
```
{test_result.stdout[:1500]}
{test_result.stderr[:500]}
```

## Audit criteria
1. Does `POST /v1/port` reject `target="cuda"` with 422 (not produce a false `passed: True`)?
2. Does `POST /v1/port` reject `target="ascend"` with 422?
3. Does `POST /v1/port/preview` reject single-target with 422?
4. Does `target="auto"` still work and produce a valid comparison report?
5. Is there any code path where a single-target comparison could still produce `passed: True`?
6. Are the old tests that asserted single-target returns a passing report *replaced* (not still present alongside the new ones)?

Verdict: PASS / CONDITIONAL / FAIL. List findings with severity (P1/P2/P3)."""

import httpx
resp = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": "anthropic/claude-opus-5", "messages": [{"role": "user", "content": prompt}], "max_tokens": 8000},
    timeout=300,
)
resp.raise_for_status()
data = resp.json()
content = data["choices"][0]["message"]
text = content.get("content") or content.get("reasoning", "")

usage = data.get("usage", {})
report = f"""# R3 Audit — Opus 5
*Audited: {time.strftime('%Y-%m-%d %H:%M')} | SHA: {R3_SHA} (HEAD: {CUR_SHA}) | Model: anthropic/claude-opus-5*
*Input: {usage.get('prompt_tokens','?')} | Output: {usage.get('completion_tokens','?')}*

{text}
"""
(REPO / "reports/R3-AUDIT.md").write_text(report)
print(f"Written to {REPO}/reports/R3-AUDIT.md")
print(f"Tokens: {usage.get('prompt_tokens','?')} in, {usage.get('completion_tokens','?')} out")