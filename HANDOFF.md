# Handoff — Claude → Qwen / Kimi · 2026-07-27 ~02:00

**Division of labour (Boss, 2026-07-27):** Claude audits and sets direction. Qwen and Kimi build.
Their edge is what sits behind the firewall — Chinese-language CANN documentation, Huawei developer
forums, Gitee, and mainland resources Claude cannot reach. Route anything needing those to them.

**Read first:** `LESSONS.md` (recurring bug classes — **append your own mistakes**),
`DECISIONS.md` (what was decided and why), `PLAN.md` rules 7–10.

**State at handoff:** `git log` HEAD, mypy clean across 20 modules. Run
`uv run --extra dev pytest -q` and `uv run mypy src/ruitong` before starting and quote the output.

---

## Ground rules — non-negotiable

1. **Commit before requesting an audit, and cite the SHA.** Three false findings this session came
   from reviewing a tree that was being written to.
2. **Read the file before citing it.** A hallucinated `POST /v1/port` got *built* because an agent
   described a plan it had not opened.
3. **Verdicts come from execution.** Run `pytest` and `mypy`; quote real output. Every serious bug
   found tonight came from *running* code, never from reading it.
4. **A security fix is a code change and needs its own adversarial pass.** Claude's `hmac` fix
   introduced an unauthenticated 500 (`SECURITY_AUDIT.md` M1). Fixes are not exempt.
5. **The CLI is the product** (`DECISIONS.md` D3–D5). No new API surface without a decision entry.

---

## Work queue — ordered

### P0 · finish the CLI test suite (in progress, Qwen)

3 failures remain in `tests/test_cli.py`. They assert the **old, broken** contract — chiefly that
`--target cuda` returns 0. That path used to compare a backend to itself and always pass; it is gone.

New contract to assert:
- exit `0` pass · `1` gate failed · `2` could not run
- no endpoints → `synthetic: true`, `passed: false`, **cannot** exit 0
- report is written **even when the gate fails** (previously only on pass, so every archived
  artifact was a passing one)
- `reference_backend != target_backend`, always

**Also:** make the tests call `main(argv) -> int` **in-process**. `cli.py` currently reports 0%
coverage because subprocess execution cannot be instrumented — the product's primary surface has no
coverage signal. Keep only two subprocess tests to prove packaging.

### P1 · security — availability, all exploitable today from one laptop

Full detail with executed exploits in `SECURITY_AUDIT.md`.

- **H2 · rate limiting runs before auth.** Starlette's `add_middleware` inserts at index 0, so the
  *last* registered runs *first*. Five unauthenticated requests consumed the whole budget and locked
  out the paying caller. Register the limiter **before** `auth_middleware` so it runs after it, and
  key the bucket on the authenticated principal, not the IP.
- **C1 remainder · move the runner off the event loop.** Input is now capped, but the comparison
  loop still never awaits — `run_in_executor` / `anyio.to_thread`, and move `JobStore` calls too.
- **H3 · wire the dead cap.** `count_active()` is defined and called from nowhere. No `DELETE`, no
  TTL, no vacuum; measured **29× disk amplification** (500 KB in → 14.5 MB on disk).
- **S2 · payload cap bypass.** Enforce while reading the body; `Transfer-Encoding: chunked` carries
  no `Content-Length` and skips the check entirely.
- **S3 · unbounded rate-limiter buckets** — evict stale keys.
- **S4 · behind Cloudflare** `request.client.host` is a Cloudflare IP, so all customers share one
  bucket. Read `CF-Connecting-IP` **only** when the peer is a trusted proxy.
- **M2 · must land BEFORE `VllmHttpBackend` is wired to a route.** Backend errors currently relay
  200 bytes of the upstream body to the client — a demonstrated payload contained an internal
  traceback, another customer's model path, and an `hf_token=`.
- **M3/M4/M5** — orphaned background tasks, in-memory job store with multiple workers (18 of 20
  jobs 404'd), Dockerfile runs as root and ignores `uv.lock`.

### P2 · the equivalence gate is mis-calibrated — this is the product

`CALIBRATION.md` and `DECISIONS.md` D7. All measured, no hardware needed.

- **Drop `cosine_similarity`.** Scale-invariant by definition — ×1.01 through ×2.0 all score exactly
  `1.0000000000`. A softmax/temperature bug is mathematically invisible to it. It moved for one of
  five injected faults.
- **Replace full-vocab `max_abs_diff` with top-k (k=10).** The current metric is dominated by the
  vocabulary tail (logprobs −130 to −180, probability ~1e-57). BF16 rounding alone measures
  **0.4929** against a threshold of 0.05 — **every correct port fails**. Restricted to the top 10
  tokens it measures 0.0152, and separation from the weakest fault improves from **1.02× to 15×**.
- **Rename `max_absolute_difference`** — it computes a mean of per-row maxima, then means again
  across prompts. The worst case is the one number a customer cares about and the one the harness
  never computes.
- **Gate on coverage.** 99 of 100 prompts erroring still yields `passed: true`; one catastrophically
  broken prompt in 100 is averaged into silence. Adding prompts currently makes the gate *weaker*.
- **`top_k_agreement` compares position indices, not tokens.** `top_logprobs` — the actual
  `{token_id: logprob}` map — is generated by the fixtures, carried through the schema, and thrown
  away. Same tokens with reordered confidence reports total disagreement; different tokens with
  matching confidence reports perfect agreement.
- **Raise `max_tokens` above 1.** At the runner's own `max_tokens=1`, logprob vectors have length 1,
  which makes cosine, top-1 and top-5 **constant 1.0**. Three of four gates are decoration.
- **Wire `equivalence/faults.py`** into a CI sensitivity suite. It is currently dead code. A fault
  the gate stops detecting is a false PASS the product would ship.

### P3 · needs Chinese-language sources — your edge, not Claude's

- **Confirm `vllm-ascend` + Qwen3-8B on a real Atlas A2 (910B).** Huawei forums, Gitee issues, CSDN
  and Zhihu writeups are where the real install friction is documented. Which CANN / torch_npu /
  vLLM row actually works together in practice, not just on paper?
- **MindIE vs vllm-ascend** — benchmarks suggest MindIE is faster, but its open-source presence is
  thin (16 GitHub stars, development on Gitee). Is it viable for a small team?
- **Ascend access from Hong Kong.** Huawei Cloud International's CN-Hong Kong region has Snt9B
  (`RESEARCH.md`) — confirm real pricing and whether Ascend software packages are permission-gated
  in practice. Verified mainland comparison: China Telecom 天翼云 at **¥38.45/hr** per 910B2 card.
  Treat aggregator pricing as fabricated unless sourced (the widely-quoted "¥19.8/hr 星宇智算" is
  fake — that platform has no Ascend at all).
- **信创 / domestic-substitution procurement.** Is there a mandate with teeth that forces CUDA→Ascend
  migration, and on what timeline? That determines whether this market is pulled or pushed.

---

## What Claude changed this session

Repo placed under git (`4cd7835`) — there was no version control through Phases 0–4 despite ~190
tests, which is how a fabricated citation and a false "7 failing tests" report both survived.

`DECISIONS.md` D1–D7 · `LESSONS.md` · `SECURITY_AUDIT.md` (2 rounds) · `CALIBRATION.md` ·
`QA_PHASE2.md` / `QA_PHASE4.md` · `PHASE_4_5_PROPOSAL.md` · `equivalence/faults.py`.

Fixed: constant-time key comparison (then fixed *that* fix for non-ASCII) · eager per-request config
rebuild · `.gitignore` missing `.env` and `*.db` · CLI self-comparison · no real backend wiring ·
report not written on failure · exit-code contract · unbounded `prompts`.

## Open for Boss — not for you to decide

1. **Export-control counsel** (`RESEARCH.md`). Blocks *commercialisation*, not building. Three
   specific questions; a customer-run CLI is the safer posture (D5).
2. **Per-customer API keys** (`SECURITY_AUDIT.md` H1) — a design decision, and the one that matters
   most commercially. The report *is* the product and the system cannot currently tell one customer
   from another.
3. **Phase 4.5** — calibrate against a real model on CPU before renting Ascend (`PHASE_4_5_PROPOSAL.md`).
