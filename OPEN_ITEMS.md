# Open Items

This is the short operational tracker. Detailed requirements and acceptance
criteria remain in `REDSTAR_PLAN.md` under “R8 / R9”.

## R8 — Ascend `token_id:NNNNN` research

- **Status:** Queued — blocking product decision
- **Hardware/spend:** None
- **Deliverable:** Append evidence-backed findings to
  `RESEARCH_ASCEND_LOGPROBS.md`.
- **Done when:** The evidence determines whether CUDA↔Ascend token identity is
  usable directly, usable with a version/configuration constraint, or
  structurally incompatible with the current gate.

## R9 — Ascend validation script

- **Status:** Waiting on R8; implementation and NVIDIA/mock debugging require no
  new rental.
- **Hardware/spend:** Existing NVIDIA endpoint or `tools/mock_vllm.py`; no
  Ascend spend.
- **Deliverable:** `tools/validate_backend.py` with specific diagnostics for
  model identity, populated logprobs, usable tokens, finite values, plausible
  probability mass, and warm-repeat determinism.
- **Done when:** The happy path exits 0, each injected failure exits 1 and names
  its failed check, and mutation testing demonstrates that every check is
  exercised.

## Update convention

For each item, record the date, evidence or commit link, remaining blocker, and
next action when its status changes. GPU instances belong in the separate
`ruitong-gpu-comparison/QUEUE.md` instance log.
