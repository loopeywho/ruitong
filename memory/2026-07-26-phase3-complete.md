# Ruitong Bridge — Project Journal

## 2026-07-26 — Phase 3 ✅, Phase 4 🚀

**Phase 3 (vLLM HTTP Backend)**: Delivered and audited.
- 125/125 tests pass, mypy clean, 96% coverage
- Claude audit: ✅ PASS — zero P1 blockers
- Files: `src/ruitong/backends/vllm_http.py`, `tests/test_vllm_http.py`, `deploy/README.md`
- Retry with backoff and SSE warning logging noted as P2 items (low priority)

**Phase 4 (Equivalence Harness — CLI-first)**: Dispatched to Qwen.
- Shape: `ruitong port <model> --target ascend [--output report.json]`
- Metrics: cosine sim, max abs diff, top-1/5 agreement
- Testable on CPU via fakes (no hardware needed)
- Delegation: `deleg_33cbb4da` via Qwen 3.6-35b-A3B

**Known issues:**
- Deployment NOT approved yet — security gate + CNY-native pricing required
- Ascend work gated behind Spike S1
- ruitong.io domain at Porkbun, DNS not configured
- CNIPA trademark filing not initiated
