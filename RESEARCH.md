# Ruitong — Research Notes (reference only, not re-read each round)

Last verified **2026-07-26**. Re-verify version-specific claims before acting.

⚠️ **Meta-warning:** three of the four "research findings" in the original plan traced to **AI-generated SEO content farms** (aimadetools.com, andrew.ooo, and a fake `deepseek.ai` that is *not* DeepSeek's domain), which search engines surface as authoritative. Ascend pricing pages are similarly polluted. Verify anything in this space against a primary source.

---

## S1 — RESOLVED: how to serve LLMs on Ascend

**Answer: `vllm-ascend`. Not ONNX Runtime.**

[github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) — official `vllm-project` org, docs on docs.vllm.ai. 446 contributors, ~110 commits/week, pushed 2026-07-26. Stable **v0.18.0** (2026-04-30), RC **v0.23.0rc1** (2026-07-19). Continuous batching + PagedAttention are **inherited from vLLM's V1 engine** — the plugin implements only the platform/attention/worker layer. Caveat: mostly RC releases; `latest` docs are labelled "developer preview."

### ❌ ONNX Runtime CANN EP — rejected (kept so nobody re-proposes it)

- **Nobody has publicly done it.** Zero success reports, EN or CN. The one 7B attempt ([ORT #22229](https://github.com/microsoft/onnxruntime/issues/22229)) hung on FP16, fell back to CPU on FP32, ran slower than CPU, closed stale after 12 months with no maintainer reply.
- **No dynamic shape support** — [ORT #24498](https://github.com/microsoft/onnxruntime/issues/24498) open since Apr 2025, unanswered. The EP compiles a per-shape `.om`; decoding changes shape every token.
- **No generation loop for Ascend.** `onnxruntime-genai` (KV-cache mgmt, sampling, batching) supports CPU/CUDA/DirectML/TensorRT/OpenVINO/QNN/WebGPU — **not CANN**.
- Fair to the other side: the EP is *not* abandoned, and ONNX export of Qwen/Llama via Optimum works fine. The blocker is dynamic shapes + no serving layer, **not** op count or export difficulty.

### Alternatives

| Path | Verdict |
|---|---|
| **MindIE / MindIE-LLM** | Huawei's own; benchmarks *faster* than vllm-ascend. Open-source face thin (16 GitHub stars, dev on Gitee). Revisit if throughput binds. |
| vllm-mindspore | Far behind (vLLM 0.9.1 vs 0.23). Only if committed to MindSpore. |
| AscendSpeed | **Not a serving path** — training tooling. Don't plan around it. |

---

## Hardware — specs corrected

⚠️ **The original plan's memory figures were wrong for both chips.**

| | Ascend 910B | Ascend 910C | NVIDIA A100 80GB |
|---|---|---|---|
| HBM | **64 GB** (32 GB on 910B4) | **128 GB** | 80 GB |
| Bandwidth | 1,600 GB/s (800 on B4) | ~3.2 TB/s | 2,039 GB/s |
| FP16 peak | ~400 TFLOPS top bin (280–376 shipping) | ~800 TFLOPS | 312 TFLOPS |

Sources: [CSET](https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/), [SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72); corroborated by vllm-ascend's own node specs (Atlas 800 A2 = 64G×8, A3 = 128G×8).

- The claimed "96 GB" is the **Atlas 300I Duo — an Ascend 310P inference card**, not a 910. Conflation.
- "910B comparable to A100" holds **on compute** (400 vs 312 TFLOPS) but **not memory** — the A100 has *more*.
- ⚠️ **910B/910C are now a generation behind.** Ascend **950PR shipped Q1 2026**; **950DT lands on Huawei Cloud Aug 2026** (144 GB HBM, 2 PFLOPS FP8). Targeting 910B/C today is targeting last-gen.

### vllm-ascend hardware support

| Hardware | Status |
|---|---|
| **Atlas A2 (= 910B)** | ✅ Primary target. 7–8B BF16 on **one 32 GB card at TP=1** — best-trodden config |
| Atlas A3 (≈ 910C-class) | ✅ Supported — but **no A3 wheel on PyPI** ([#11160](https://github.com/vllm-project/vllm-ascend/issues/11160)); build from source or use Docker |
| Atlas 300I Duo (310P) | 🔵 Experimental — different CANN, FP16 only |
| **Ascend 910A / 910 Pro B** | ❌ **Unsupported, "unplanned."** Do not rent. |

---

## Model choice — Qwen2.5-7B is deprecated here

[PR #8452](https://github.com/vllm-project/vllm-ascend/pull/8452) (merged **2026-04-21**) declared Qwen2.5-7B "out-of-date," **deleted its tutorial and removed its e2e CI configs**. On current `main` it has **zero CI coverage**, plus:

- [#2239](https://github.com/vllm-project/vllm-ascend/issues/2239) — crash in `SelfAttentionOperation`, open ~12 months
- [#5269](https://github.com/vllm-project/vllm-ascend/issues/5269) — **garbled output on prefix-cache hit**
- [#10604](https://github.com/vllm-project/vllm-ascend/issues/10604) — W8A8 gives no speedup and *worse* accuracy

**→ Use Qwen3-8B.** Core tier, tutorial, two live CI configs (BF16 + W8A8).

## Environment assembly

Treat **CANN + torch + torch_npu + vLLM + vllm-ascend as one compatibility set** — pick one complete row of the [version matrix](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/community/versioning_policy.md) and honor it exactly. Strict 1:1 vLLM pairing. One stable row pins torch_npu to a **git-hash build**; PyPI metadata contradicts the docs ([#12404](https://github.com/vllm-project/vllm-ascend/issues/12404)).

**Use `quay.io/ascend/vllm-ascend` Docker on an A2 host** → 1–3 days. Hand-install CANN → 1–2 weeks. Current CANN line is **9.0.x** (not 6.3 — that's a 2023 line).

## Known hazards affecting our design

- **A backend can return HTTP 200 with an empty body** — v0.23.0rc1: *"the load-balance proxy can swallow decode errors and return an empty HTTP 200 response."* Validate response *content*, never trust status code.
- KV-cache transfer can cause precision issues / TP-shard inconsistency on reformat.
- Streaming correctness bugs: [#12030](https://github.com/vllm-project/vllm-ascend/issues/12030), [#9767](https://github.com/vllm-project/vllm-ascend/issues/9767).
- LoRA, pooling, beam search 🔵 Experimental. TP>4 + graph mode is an open gap.

**Calibration:** [arXiv:2607.08215](https://arxiv.org/html/2607.08215v1) (2026-07-09) is the harshest evidence — twelve startup patches, "essentially undebuggable" kernel faults. **Don't over-apply it:** they ran 300–540 GB MoE/multimodal across 16 devices and attribute failures to *feature interaction* absent in a single-card dense model. Upper bound on pain, not forecast.

---

## Ascend access — ✅ Hong Kong works

**Correction to an earlier note in this file:** a first sweep concluded no non-mainland Ascend rental existed. **That was wrong.**

**Huawei Cloud's international site has a CN-Hong Kong region with Ascend (Snt9B).** Huawei's own docs, updated **2026-06-27**:

> "For inference deployment, it is advised to use the notebook and Snt9B resources in the **CN-Hong Kong** region."
> — [ModelArts best practice](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_aigc_infer_nb5907.html)

Corroborated by the [Ascend-vLLM Lite Server guide](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906004.html) listing Snt9B images for CN-Hong Kong (AP Southeast-1).

**Signup needs no mainland entity.** Intl registration = email + local phone + payment method + billing address. Real-name verification is required **only if buying mainland-region resources**. HK/Macau/Taiwan customers are explicitly directed to the international site. Default contracting party: **Huawei Cloud Services (Hong Kong) Limited**. Intl and mainland accounts do not interoperate.

*Friction:* Ascend software packages are permission-gated — expect to need a named Huawei contact even with a valid account.

### Pricing

**Huawei Cloud does not publish Ascend hourly rates** — the calculator is a JS SPA with no fetchable price data; all billing examples use CPU flavors. It is a **sales-contact enterprise product**.

The only verified public itemised Ascend price is **China Telecom 天翼云: ¥38.45/hr** for one 910B2 card (¥18,454/mo) — [ctyun price page](https://www.ctyun.cn/document/10029787/10047957), updated 2026-06-05. Not the ¥15–25/hr originally assumed.

⚠️ **Third-party Ascend prices span ¥0.85–¥19.8/card-hour — a 20× spread, mostly fabricated.** The widely-cited "¥19.8/hr 星宇智算" is fake; that platform's actual catalogue is RTX 4090/5090 only, no Ascend, all sold out. Discard prices from suanlix.com, anygpu.cn, 36171.com, idcsp.com, jygpu.com, china2077.com, mornai.cn.

| Other route | Ascend? | Price |
|---|---|---|
| Gitee AI 模力方舟 | ✅ documented 910B | console-gated |
| AutoDL | ✅ 910B2 + Kunpeng ARM | not published |
| SiliconFlow | Runs on Ascend, **token API only** (no SSH) — accepts 回鄉證 | per-token |
| OpenI 鹏城云脑 | ✅ free, points-gated, mainland-student oriented | free |
| OrangePi AIpro dev board | Ascend 310-class, by application | n/a |

No free Ascend tier on Huawei Cloud. Unresearched: whether Ascend hardware can lawfully be exported mainland → HK.

---

## Export controls — recalibrated (less categorical, still real)

**Two earlier overstatements in this file, corrected:**

1. ❌ *"Using Ascend anywhere in the world violates US export controls."* The original 2025-05-13 BIS press release said this; **BIS removed the phrase ~2025-05-21**. Current wording is *"alerting industry to the **risks** of using"* — a retreat from per-se violation to enforcement posture.
2. ❌ *"No non-mainland Ascend access exists."* Wrong — see CN-Hong Kong above.

### What's actually true

- [BIS GP10 guidance, 2025-05-13](https://www.bis.gov/media/documents/general-prohibition-10-guidance-may-13-2025.pdf) names **Ascend 910B, 910C, 910D** as presumptively produced in violation of the EAR. GP10 (15 CFR §736.2(b)(10)) covers "use" and **"otherwise service."**
- **Never codified.** Federal Register queries return **zero** documents mentioning General Prohibition 10 since 2025-01-01 and **zero** ever mentioning Huawei Ascend. This is sub-regulatory guidance — no notice-and-comment.
- **Never enforced.** No BIS action or DOJ charge under it in 14 months. (Weak evidence — a first case would be a policy decision, not a surprise.)
- **HK = China under the EAR since 2020** ([85 FR 83765](https://www.federalregister.gov/documents/2020/12/23/2020-28101)). **Incorporating in HK is not a mitigation.**
- Huawei is **not** on the OFAC SDN list (verified against live `sdn.csv`). It is Entity-Listed (2019, presumption of denial, "all items" incl. EAR99) and on NS-CMIC (securities-trading only).
- **50% Affiliates Rule** ([90 FR, 2025-09-30](https://www.federalregister.gov/documents/2025/09/30/2025-19001/expansion-of-end-user-controls-to-cover-affiliates-of-certain-listed-entities)) auto-extends Entity List restrictions to ≥50%-owned affiliates — operationally the biggest change for anyone in Huawei's orbit.

### 🔑 The open-source escape hatch

**15 CFR §734.7 removes "published" software from EAR jurisdiction entirely.** Genuinely open-sourcing the router/toolchain is a *real* jurisdictional exit, not a fig leaf. Commercial licensed software gets no such treatment. **This makes the open-source question a compliance decision, not just a GTM one** — see PLAN.md open item 4.

### Where the real risk sits

Not the statute — the discretion and the market:

1. **Discretionary Entity List designation.** BIS's own policy statement warns foreign parties *"may be added to the Entity List, **even where no violation of the EAR occurs**."* For a company whose public identity is "we make Ascend easier to adopt," this is arguably the single largest risk **and no compliance program fixes it.**
2. **US customers will simply refuse.** A US buyer operating Ascend is squarely within the guidance; §764.2(b)/(e) facilitation theories apply to *any person*. Precedent: Malaysia announced an Ascend-based national AI stack on 2025-05-19 and **retracted within a day** under US pressure. Assume the US enterprise market is closed regardless of legality.
3. **The US/PRC pincer.** China's **Anti-Foreign Sanctions Law (2021)** and Blocking Statute create liability for *complying* with US measures. You may face genuinely irreconcilable obligations. **Needs HK and PRC counsel, not only US counsel.**
4. **Payment processors.** Neither Stripe nor PayPal prohibits this on paper (China/HK are not restricted jurisdictions for either), but both terminate at sole discretion, and HK Stripe merchants get **no notice-period protection**. ⚠️ Stripe+Advent bid for PayPal (2026-07-15) — if it ever completes, "Stripe primary, PayPal backup" stops being redundancy. HK alternatives: Checkout.com, Adyen, PayDollar/AsiaPay.

### Probably *not* a problem

- **CFIUS** — structurally inapplicable to outbound; matters only on a US-inbound acquisition or exit.
- **31 CFR Part 850 (Treasury outbound)** — covered activities are IC design/fab/packaging, **EDA for designing ICs**, SME, supercomputers, quantum, and AI trained >10²³/10²⁵ ops. Migration tooling designs no ICs and trains no models, so **on the face of the regulation it appears outside Part 850**. Caveats: "AI system" is defined broadly (§850.202(b)) and "develop" includes "substantive modification." Only bites if US capital is taken anyway. **Get a written opinion; don't rely on the plain reading.**
- **EAR §744.6(c)(2)** targets semiconductor *fabrication*, not using AI chips — likely outside. **But** a separate [BIS policy statement (2025-05-13)](https://www.bis.gov/media/documents/ai-policy-statement-training-ai-models-may-13-2025) may require a license where a US person performs *"any contract, service, or employment"* knowing it may assist AI training for parties **headquartered in D:5 countries** — and an HK company is D:5-headquartered. **Put this to counsel before hiring any US person.**

### Enforcement precedent worth studying

**[Robert Bosch GmbH — $36.2M + ~$11.4M disgorgement, 2026-06-17](https://www.bis.gov/press-release/robert-bosch-gmbh-bosch-pay-36-million-penalty-bis-violations-pertaining-shipments-huawei).** German company, foreign-produced sensors **and automotive software**, shipped entirely outside the US to Huawei, reached via the **FDP rule**. Penalty materially reduced by voluntary self-disclosure. This is the closest analogue to a non-US software company serving Huawei. (Also: Seagate $300M, 2023.)

### Counsel checklist

(1) ECCN classification of the software — consider a BIS Commodity Classification Request or Advisory Opinion under §748.3(c). (2) Whether the product is "servicing" under GP10 — **the entire commercial premise turns on this and there is no guidance on point.** (3) Whether any US person may lawfully be employed. (4) Part 850 analysis before accepting US capital. (5) **Conflict-of-laws under China's Anti-Foreign Sanctions Law.** (6) HK's own Cap. 60 strategic-commodities regime — unresearched.

*Caveats: Treasury pages timed out repeatedly; some regulation text came from Cornell LII, not official eCFR. Verify quoted regulatory text against eCFR before relying on it.*

---

## Claims from the original plan — final status

| Claim | Verdict |
|---|---|
| openPangu 2.0 = 505B MoE, all-Ascend-**910B**, zero NVIDIA | **PARTLY TRUE.** 505B/18B Pro is real (announced HDC 2026-06-12, weights not yet released); Flash 92B/6B is out. **"910B" and "zero NVIDIA" are unsourced SEO fabrication** — Huawei says only "trained on Ascend." The *credible* citation is [Pangu Ultra MoE, arXiv 2505.04519](https://arxiv.org/abs/2505.04519): **718B trained on 6,000 Ascend NPUs**, 30% MFU. Use that instead. |
| DeepSeek V4-Pro benchmarked Ascend 910C vs A100 | **Model real** ([arXiv 2606.19348](https://arxiv.org/abs/2606.19348), Apr 2026, 1.6T/49B). **Benchmark FALSE** — only deployment guides exist. The one real Ascend-vs-NVIDIA number is **910C ≈ 60% of H100** for **R1/V3** (Feb 2025). A100 is nobody's baseline. |
| CANN SDK v6.3 current | **FALSE** — current is **9.0.x** (9.0.0-beta.1, 2026-03-02). 6.x is a 2023 line. |
| 910B ~96GB / 910C ~192GB | **FALSE** — 64 GB / 128 GB. See table above. |
| vLLM issue #6368 open for Ascend support | **FALSE** — closed 2024-12-13, marked stale, body now redirects to vllm-ascend. |
| "Llama 3.2 8B" | **Does not exist** (3.2 = 1B/3B/11B/90B). Moot — use Qwen3-8B. |

## Positioning

A backend flag on an OpenAI-compatible proxy is not a moat. The defensible product is **automated porting + proof of equivalence**. Plan is sequenced accordingly.
