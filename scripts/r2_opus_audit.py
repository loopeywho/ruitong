#!/usr/bin/env python3
"""Focused Opus 5 audit on R2 cross-tenant isolation changes."""
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

# Read R2 diff
diff = subprocess.run(
    ["git", "diff", "3a16cb6..0592fc5", "--", "src/ruitong/api/router.py", "tests/test_api.py"],
    capture_output=True, text=True, cwd=REPO
).stdout

# Read the current router and test files
router = (REPO / "src/ruitong/api/router.py").read_text()
tests = (REPO / "tests/test_api.py").read_text()

# Run tests
test_result = subprocess.run(
    ["uv", "run", "--extra", "dev", "pytest", "-q", "--tb=short", "-Wignore"],
    capture_output=True, text=True, timeout=120, cwd=REPO,
)

prompt = f"""You are auditing Ruitong Bridge R2 — cross-tenant security isolation.

## Diff (R2 changes in router.py and test_api.py)
```diff
{diff[:8000]}
```

## Current router.py
```python
{router[:6000]}
```

## Current test_api.py (TestCrossTenantIsolation section)
```python
{tests[:6000]}
```

## Test output
```
{test_result.stdout[:2000]}
{test_result.stderr[:2000]}
```

## Audit criteria
1. Does LIST scope by owner? (fail-closed — missing owner => empty list, not all jobs)
2. Does DELETE scope by owner? (fail-closed — missing owner => 404, not 204)
3. Does GET return 404 for cross-tenant access (not 403 — must not confirm existence)?
4. Are edge cases covered? (empty list, delete non-existent job that you own, fresh key sees nothing)
5. Is there any code path where a missing/empty owner allows access?

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
report = f"""# R2 Audit — Opus 5
*Audited: {time.strftime('%Y-%m-%d %H:%M')} | SHA: 0592fc5 | Model: anthropic/claude-opus-5*
*Input: {usage.get('prompt_tokens','?')} | Output: {usage.get('completion_tokens','?')}*

{text}
"""
(REPO / "reports/R2-AUDIT.md").write_text(report)
print(f"Written to {REPO}/reports/R2-AUDIT.md")
print(f"Tokens: {usage.get('prompt_tokens','?')} in, {usage.get('completion_tokens','?')} out")