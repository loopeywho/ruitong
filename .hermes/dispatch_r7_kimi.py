"""Dispatch KIMI K3 on R7 (P2.13 — wire /v1/port to the registry)."""
import subprocess
import sys
import os

PROMPT = """I am KIMI K3 on the Redstar profile. My task is R7 from REDSTAR_PLAN.md.

## Context

Read these files first:
1. ~/Projects/ruitong-bridge/REDSTAR_PLAN.md — section R7 (lines 280-332)
2. ~/Projects/ruitong-bridge/src/ruitong/api/router.py — the full file
3. ~/Projects/ruitong-bridge/src/ruitong/main.py — the full file
4. ~/Projects/ruitong-bridge/src/ruitong/registry.py — the BackendRegistry class
5. ~/Projects/ruitong-bridge/src/ruitong/backends/fake.py — FakeCuda/FakeAscend classes
6. ~/Projects/ruitong-bridge/tests/test_api.py — existing port tests

## Do NOT redo — already done in 667a808

- Job concurrency cap is wired (router.py:160)
- Orphaned asyncio tasks are tracked and drained on shutdown
- main.py lifespan already registers VllmHttpBackend when config.cuda_base_url is set
- Leave all of that alone

## The actual gap

_make_runner (api/router.py:27) hardcodes FakeCuda()/FakeAscend() on every call
and completely ignores the registry. The config plumbing exists but is unused.

## Scope (3 items)

### 1. _make_runner pulls from the registry
Change its signature to accept a BackendRegistry (or get it from request.app.state).
When target=="auto", get both backends from the registry. When a backend is not
registered, raise HTTPException(422) — not a silent fallback to fakes.

### 2. GET /v1/models uses the registry
main.py:107 currently instantiates FakeCuda/FakeAscend per request unconditionally.
Use app.state.router.registry.list_models() instead.

### 3. validation_level must tell the truth
Currently hardcoded "simulated" at both call sites in router.py (lines 142 and 198).
Derive it from what actually ran: "simulated" when either side is a fake, "live" (or
similar) when both are real backends. This is the most important item — a report
that says "simulated" while using real hardware is merely unhelpful, but one that
omits the label while using fakes is a false trust claim (D5).

## Acceptance

- With no endpoints configured: POST /v1/port still works, still returns
  validation_level="simulated", and TestPortEndpoint passes unchanged.
- With both endpoints configured (point at mock_vllm.py — no spend):
  validation_level is NOT "simulated".
- A test asserting _make_runner returns registry-held instances, not fresh fakes.
- mypy clean, pytest green.

## Explicitly OUT of scope

- C1 blocking-loop remainder (performance only, not correctness)
- Any GPU spend (use tools/mock_vllm.py stub server)

## Rules

1. Read the files above before writing code.
2. Run pytest and mypy after every change. Quote real output.
3. Log any mistakes to LESSONS.md in the existing format.
4. Append results to QA_FINDINGS.md (never overwrite).
5. Never edit corpora/* or reports/*.
6. Start now. Read the plan files first, then implement.
"""


def main() -> int:
    # Use absolute paths to avoid shell interpretation
    cwd = os.path.expanduser("~/Projects/ruitong-bridge")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [
            "hermes",
            "--profile", "redstar",
            "-t", "file,web,terminal",
            "chat",
            "-q", PROMPT,
        ],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output in real time
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    proc.wait()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())