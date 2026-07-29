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

# Read R2 fix diff (ce1a10b vs 0592fc5 — Kimi's fix round)
PREV_SHA = "0592fc5"
CUR_SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
diff = subprocess.run(
    ["git", "diff", f"{PREV_SHA}..{CUR_SHA}", "--",
     "src/ruitong/api/router.py", "src/ruitong/jobs/persistence.py",
     "src/ruitong/main.py", "tests/test_api.py", "tests/test_persistence.py"],
    capture_output=True, text=True, cwd=REPO
).stdout

# Read the current files
router = (REPO / "src/ruitong/api/router.py").read_text()
persistence = (REPO / "src/ruitong/jobs/persistence.py").read_text()
main_py = (REPO / "src/ruitong/main.py").read_text()
tests = (REPO / "tests/test_api.py").read_text()
persistence_tests = (REPO / "tests/test_persistence.py").read_text()

# Run tests
test_result = subprocess.run(
    ["uv", "run", "--extra", "dev", "pytest", "-q", "--tb=short", "-Wignore"],
    capture_output=True, text=True, timeout=120, cwd=REPO,
)

prompt = f"""You are auditing Ruitong Bridge R2 — cross-tenant security isolation.

This audit covers the FIX ROUND (commit {CUR_SHA}) that was applied after the initial R2 audit found 2× P1.

## Fix diff (0592fc5..{CUR_SHA})
```diff
{diff[:6000]}
```

## Current router.py
```python
{router[:4000]}
```

## Current persistence.py
```python
{persistence[:4000]}
```

## Current main.py (principal derivation)
```python
{main_py[:2000]}
```

## Current test_api.py
```python
{tests[:4000]}
```

## Current test_persistence.py (new — add-layer tests)
```python
{persistence_tests[:3000]}
```

## Test output
```
{test_result.stdout[:1500]}
{test_result.stderr[:1500]}
```

## Audit criteria
1. Does LIST scope by owner? (fail-closed — missing owner => empty list, not all jobs)
2. Does DELETE scope by owner? (fail-closed — missing owner => 404, not 204)
3. Does GET return 404 for cross-tenant access (not 403 — must not confirm existence)?
4. Are edge cases covered? (empty list, delete non-existent job that you own, fresh key sees nothing)
5. Is there any code path where a missing/empty owner allows access?

Previous audit findings (check if each is resolved):
- P1-1: LIST/DELETE use _require_owner not _principal
- P1-2: Store-level fail-closed on empty owner
- P1-3: Dev mode auth-off collapse
- P2-1: JobStore.default() fallback
- P2-2: Auth-required tests for new routes
- P2-3: Delete/worker race
- P3-1: Edge-case tests

Verdict: RESOLVED / CONDITIONAL / FAIL. For each previous finding, state RESOLVED or NOT RESOLVED with evidence."""

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
report = f"""# R2 Audit — Opus 5 (Fix Round)
*Audited: {time.strftime('%Y-%m-%d %H:%M')} | R1 SHA: {PREV_SHA} → R2 SHA: {CUR_SHA} | Model: anthropic/claude-opus-5*
*Input: {usage.get('prompt_tokens','?')} | Output: {usage.get('completion_tokens','?')}*

{text}
"""
(REPO / "reports/R2-AUDIT.md").write_text(report)
print(f"Written to {REPO}/reports/R2-AUDIT.md")
print(f"Tokens: {usage.get('prompt_tokens','?')} in, {usage.get('completion_tokens','?')} out")