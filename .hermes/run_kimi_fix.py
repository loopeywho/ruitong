#!/usr/bin/env python3
"""Run KIMI K3 with a large prompt, bypassing shell quoting issues."""
import subprocess, sys, os

prompt = r"""R2 fix round. Opus 5 audit found 2x P1 findings on the cross-tenant isolation at commit 0592fc5. Fix them now.

## Context
Current code in repository at /Users/loopey/Projects/ruitong-bridge. The R2 changes added LIST and DELETE endpoints with owner scoping in src/ruitong/api/router.py. The JobStore lives in src/ruitong/persistence/ -- read it before coding.

## P1 -- F1: Empty principal bypasses all scoping
_principal() returns '' when auth is disabled or middleware didn't run. The new endpoints pass this straight to the store. If JobStore uses "if owner:" filtering, owner='' means NO filter -- full tenant data dump.

Fix -- belt AND braces:
1. Add _require_owner() in router.py that raises 403 if owner is empty
2. Add "if not owner: raise ValueError('owner is required')" at the top of JobStore.get(), JobStore.delete(), JobStore.list_by_owner() in persistence
3. Replace _principal(request) calls in the new LIST/DELETE/GET endpoints with _require_owner(request)
4. The existing GET get_port_job endpoint also needs _require_owner -- it already scopes but doesn't guard empty owner

## P1 -- F2: Store-level fail-closed tests don't exist
Add direct JobStore tests in tests/test_persistence.py (create if missing):
- test_list_by_owner_rejects_empty
- test_delete_rejects_empty
- test_get_rejects_empty
- test_delete_rejects_none
- test_owner_matched_exactly -> tenantB should not see tenantA jobs

## Also fix P2 -- F4: Plaintext key as owner
_principal returns the raw API key. Keys in owner column = query log exposure + rotation orphans every job.
- Use a stable non-secret key_id (the row-id / hash) as owner instead
- Confirm JobInfo response model has NO owner field exposed
- Add test: assert 'owner' not in response body

## Steps
1. Read the persistence layer first: find the JobStore class
2. Understand what _principal returns and where it's called
3. Implement fixes
4. Run tests: 'uv run pytest -q --tb=short -Wignore'
5. Commit with message: 'fix(security): R2 -- require_owner guard + store-level fail-closed + key_id owner'
6. Report back the SHA and test count
"""

os.chdir("/Users/loopey/Projects/ruitong-bridge")
proc = subprocess.run(
    ["hermes", "--profile", "redstar", "-t", "file,web,terminal", "chat", "-q", prompt],
    capture_output=True, text=True, timeout=600
)
sys.stdout.write(proc.stdout)
sys.stderr.write(proc.stderr)
sys.exit(proc.returncode)