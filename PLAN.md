# 瑞通 (Ruitong) — CANN/CUDA Bridge · Implementation Plan

> **Implementer:** Qwen. Work task-by-task, in order. Do not skip ahead.
> **QA:** Claude reviews each completed phase. Update `CLAUDE.md` when a phase is done — that is the handoff signal. Findings land in `QA_FINDINGS.md`.
> **Reference:** `RESEARCH.md` (read once, not every round).

**Product:** Automated model porting between NVIDIA CUDA and Huawei Ascend (CANN), with proof of numerical equivalence. The unified inference API is the delivery surface, not the product.

**Stack:** Python 3.11, uv, FastAPI, pytest, httpx, Docker.

**Serving architecture (S1 resolved 2026-07-26):** **Both backends are vLLM speaking OpenAI-compatible HTTP.** CUDA uses vLLM; Ascend uses `vllm-ascend`. There is no ONNX conversion pipeline — that path was investigated and rejected (see RESEARCH.md). This makes the Ascend backend a second instance of the CUDA backend, not a research project.

**Reference model: `Qwen3-8B`.** Not Qwen2.5-7B — it was deprecated and stripped from vllm-ascend CI in Apr 2026 and carries open crash and prefix-cache-correctness bugs. Qwen3-8B is Core-tier with live CI. BF16 at TP=1 on one 32 GB Atlas A2 (910B) card.

---

## Rules for the implementer

1. **No hardware is available.** No GPU, no NPU. Every phase must build and pass tests on CPU. Hardware-dependent code sits behind an interface and is exercised by fakes.
2. **Never call a real backend in tests.** Offline, deterministic, no network.
3. **Do not deploy anything.** No Traefik, no VPS, no public exposure. Deployment is gated on a separate security review (Phase 6, not approved).
4. **Ask before adding a dependency** beyond the stack above.
5. **Minimum code that satisfies the acceptance criteria.** No speculative abstraction.
6. A phase is done when its acceptance criteria pass, not when the code looks finished.
7. **The CLI is the product; the API is not.** See `DECISIONS.md` D3–D5. `ruitong port` must run
   standalone — no HTTP server, no job store, no auth required. If it ever depends on the API being
   up, that is a regression. **Do not add new API surface** without a decision recorded in
   `DECISIONS.md`.
8. **Read the file before you cite it.** Every line number, symbol, or file claim in an audit or
   status note must be quoted or grepped at the time of writing. Three fabricated citations have
   already survived review this way — one of them (`POST /v1/port`) was hallucinated into a plan
   description and then actually built.
9. **Audit a pinned revision.** The repo is under git as of `4cd7835`. Commit before requesting an
   audit, and cite the SHA in the audit. Never audit a tree that is being written to.
10. **Verdicts come from execution, not reading.** An audit must run `uv run --extra dev pytest -q`
    and `uv run mypy src/ruitong` and quote the real output.

---

## Phase 0 — Scaffolding

**Files:** `pyproject.toml`, `src/ruitong/__init__.py`, `src/ruitong/config.py`, `tests/conftest.py`, `.gitignore`, `Dockerfile`

Config via env vars with defaults; no secrets in the repo.

**Acceptance:** `uv sync` succeeds · `uv run pytest` runs · `docker build .` succeeds.

---

## Phase 1 — Contracts

Define the data model and backend interface before any backend exists. Load-bearing — get it QA'd before Phase 2.

**Files:** `src/ruitong/schemas.py`, `src/ruitong/backends/base.py`, `src/ruitong/errors.py`

```python
# schemas.py — OpenAI-compatible, plus a backend selector.
# ChatRequest:  model, messages, backend: Literal["cuda","ascend","auto"] = "auto",
#               max_tokens, temperature = 0.0, seed, stream: bool = False
# ChatResponse: id, model, backend, choices, usage
#   -> `backend` is a Ruitong extension. Every other field strictly OpenAI-shaped.

class Backend(Protocol):
    name: str                                    # "cuda" | "ascend"
    async def health(self) -> HealthStatus: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def chat(self, req: ChatRequest) -> ChatResponse: ...
    def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...
```

**Streaming is a Phase 1 contract decision, not a later feature.** Real OpenAI clients send `stream: true` and expect SSE. Retrofitting reshapes this protocol and everything above it. Define `stream()` and `ChatChunk` now; fakes implement them. A real backend may return `501` for `stream: true` in the MVP — but the *shape* must exist from the start.

Errors: `BackendUnavailable`, `ModelNotFound`, `BackendError`. Backends raise these; never leak transport exceptions upward.

**Acceptance:** schemas round-trip through JSON with full validation · `Backend` is a `Protocol` (structural) · no backend implementations yet · tests cover malformed-request rejection.

---

## Phase 2 — Router + fake backends

**Files:** `src/ruitong/router.py`, `src/ruitong/registry.py`, `src/ruitong/backends/fake.py`, `src/ruitong/main.py`, `tests/test_router.py`

- `registry.py` — name → Backend; knows which models each backend serves.
- `router.py` — `backend="auto"` resolves via explicit policy: prefer healthy *and* serving the model; ties break by configured priority. **No silent cross-backend fallback mid-request** — if the chosen backend fails, error naming the backend. Silent failover makes equivalence claims unprovable.
- `fake.py` — deterministic `FakeCuda` / `FakeAscend` with injectable failure and latency.
- `main.py` — `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/health`.

**Acceptance:** all endpoints work end-to-end against fakes via `httpx.AsyncClient` · `auto` covered for both-healthy / one-unhealthy / both-unhealthy / model-absent · errors map to correct codes (404 model, 503 unavailable, 502 backend) · **coverage ≥ 90% on `router.py` and `registry.py`** · suite runs offline in < 5s.

---

## Phase 3 — vLLM HTTP backend (serves BOTH targets)

One client class, instantiated twice with different base URLs. CUDA and Ascend both expose the same OpenAI-compatible API.

**Files:** `src/ruitong/backends/vllm_http.py`, `tests/test_vllm_http.py`, `deploy/README.md`

- Configured by `RUITONG_CUDA_BASE_URL` and `RUITONG_ASCEND_BASE_URL`.
- Timeouts; one bounded retry on connection error only (never on a 4xx/5xx response).
- **Do not trust HTTP 200.** A known vllm-ascend defect returns an empty 200 body when the proxy swallows a decode error. Validate that the response body parses and contains non-empty choices; treat an empty or malformed 200 as `BackendError`. Cover this with a test.
- Tests mock the HTTP layer (`respx` or `httpx.MockTransport`). **Never start a real vLLM server in tests.**
- `deploy/README.md` documents exact launch commands for both sides — documentation only, not executed:
  - CUDA: standard vLLM with `Qwen3-8B`
  - Ascend: `quay.io/ascend/vllm-ascend` Docker image on an Atlas A2 (910B) host. Pin one complete row of the vllm-ascend compatibility matrix (CANN + torch + torch_npu + vLLM + vllm-ascend move as one unit). **910A is not supported.**

**Acceptance:** conforms to `Backend` · every failure mode mapped (timeout, connection refused, malformed body, empty-200, 404) · no live network in tests · one class serves both backends with no branching on backend name.

---

## Phase 4 — Equivalence harness (CLI-first) ⭐ the actual product

> **Decision: CLI-first.** Tests the business hypothesis with ~20% of the code. API wrapper comes later if needed.

Build this **before** renting any hardware. Fully testable on CPU.

**Files:** `src/ruitong/cli.py` (entry point), `src/ruitong/equivalence/runner.py`, `src/ruitong/equivalence/metrics.py`, `tests/test_equivalence.py`, `tests/test_cli.py`

**CLI shape:**
```
ruitong port <model> --target ascend  [--output report.json]
ruitong port <model> --target cuda    [--output report.json]
ruitong port <model>                  # auto — compare both
```

Produces structured equivalence report (JSON + human-readable terminal output).

Two comparison modes — do not conflate them:

1. **Teacher-forced logprob comparison (the real gate).** Send both backends the *same fixed token prefix* and compare per-position logprobs. Because both sides are OpenAI-compatible, this likely needs **no custom backend capability** — vLLM exposes `logprobs` / `top_logprobs`, and `echo=true` (or vLLM's `prompt_logprobs` extension) returns prompt-token logprobs. **Verify which of these `vllm-ascend` actually honors before building on it** — if neither works, fall back to mode 2 and say so in the report. Thresholds: cosine ≥ 0.99, max abs diff ≤ 0.05 (BF16), top-1 agreement ≥ 99%, top-5 set agreement ≥ 95%.
2. **Task-level parity (the sanity gate).** Fixed prompt suite, compare aggregate quality within a configured tolerance.

**Do not gate on token-by-token free generation matching.** Once one argmax differs the sequences diverge permanently — expected across different kernels, not a defect. Report generation divergence as an observation, never pass/fail.

**Acceptance:** `ruitong port` CLI runs and produces structured report · metrics unit-tested against synthetic tensors with *known* answers, including deliberately-failing pairs · runs entirely against fakes · report states which mode ran and why · CLI help text installed via `pip install -e .` entry point.

---

## Phase 5 — Ascend bring-up (BLOCKED — legal gate)

> 🛑 **CHECK WITH COUNSEL BEFORE COMMERCIALISING — not before experimenting.**
> BIS GP10 guidance (2025-05-13) names Ascend 910B/910C/910D as presumptively produced in violation
> of the EAR, and GP10 covers "use" and "otherwise service." **It has never been codified and has
> never been enforced in 14 months.** The unresolved question is whether *selling* Ascend migration
> tooling counts as "servicing" — that is a commercial-launch question, not a lab question. The
> larger risks are discretionary Entity List designation and US customers simply refusing.
> See RESEARCH.md. Phases 0–4 need no Ascend hardware and are unaffected.

**S1 is resolved — there is no research spike left.** This phase is: get an Ascend instance, run the `quay.io/ascend/vllm-ascend` image with Qwen3-8B, point `RUITONG_ASCEND_BASE_URL` at it, run the Phase 4 harness against the live pair.

**Sourcing (corrected):** **Huawei Cloud International has a CN-Hong Kong region with Ascend (Snt9B)** — signup needs no mainland entity and no real-name verification for non-mainland resources; contracting party is Huawei Cloud Services (Hong Kong) Limited. Pricing is sales-contact (unpublished). Alternative with published pricing: China Telecom 天翼云 at **¥38.45/hr** per 910B2 card — not the ¥15–25/hr originally assumed. Most cheaper figures online are fabricated.

No new backend code should be required. If it is, that is a finding — report it rather than working around it.

Budget 1–3 days for environment assembly via Docker on an A2 host. Expect version-pinning friction; do not deviate from a single matrix row to "fix" it.

**Acceptance:** Phase 4 report generated from two *live* backends · every deviation from the documented setup recorded in `RESEARCH.md`.

---

## Phase 6 — Deployment (NOT APPROVED — do not begin)

Blocked pending Boss's security gate. Constraints when it opens:

- **Ruitong does not deploy to the ShangQiao VPS.** That box carries ShangQiao production traffic and paying customers. Ruitong gets its own host.
- Security review before any public exposure, including temporary tunnels.
- Pricing/billing, if added, is CNY-native.

---

## Open items for Boss

1. **API-first or CLI-first MVP** — recommend CLI-first (`ruitong port <model> --target ascend` → artifacts + equivalence report). Tests the business hypothesis with ~20% of the code; API becomes a wrapper later. Plan currently assumes API-first as written.
2. **Landing page vs. prototype** — recommend landing page, given ShangQiao is mid-launch.
3. **Pricing shape** — recommend flat per-model porting fee over per-token.
4. **Open-source split** — router open, porting/equivalence proprietary?
5. **Entity/compliance posture** — pending research thread 3 (Entity List exposure, ModelArts access from HK).
