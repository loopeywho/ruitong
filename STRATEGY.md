# 瑞通 Ruitong — Strategic Findings · 2026-07-27

**Status: FOR BOSS'S DECISION. Not a decision record.** Four independent research threads
(market, competitors, regulatory demand, security) converged on one conclusion, and it is not the
one the plan assumed. I am not deciding this — the evidence is below, sharpened.

---

## The single most important fact

**Huawei already ships this product. Free. Open source. As a CLI.**

It is called **`msprobe`** (`pip install mindstudio-probe`, Apache-2.0, part of `Ascend/mstt`):

```
msprobe compare -tp <npu_dump_path> -gp <gpu_dump_path> -o <output_path>
```

It emits `compare_result_{timestamp}.xlsx` with per-API **pass / warning / error** verdicts and an
`advisor_{timestamp}.txt` naming the culprit operators. It computes cosine similarity, MaxAbsErr,
MaxRelativeErr, and 双千/双万 error ratios. **It publishes official pass/fail thresholds** —
`cosine > 0.99` and `MaxAbsErr < 0.001`.

And Huawei Cloud's migration guide, **updated 2026-07-04**, states our problem statement in our
words: *"同一模型，从CPU或GPU移植到NPU中存在精度下降问题"* — the same model, ported from CPU or GPU
to NPU, has accuracy degradation.

**We cannot sell "we diff the tensors."** That is a commodity, funded by the hardware vendor, with
published thresholds, three weeks fresher than our plan.

## The demand does exist — it is just documented as a bug, not a budget

The best evidence in this whole file is Huawei's own issue tracker. **vllm-ascend issue #31**, opened
2025-02-11, **still open seventeen months later with no maintainer resolution**:

> "The inference results on the GPU are significantly different from those on the NPU. We used the
> same code and set temperature=0 to ensure reproducibility."

That is a customer writing our problem statement, unprompted, in public. Seven more open
correctness issues landed in the last month alone (garbled output, precision drops, accuracy
regressions across GLM-5.2, Qwen3.5, Kimi-K2.7, DeepSeek v4pro).

And `vllm-ascend`'s own accuracy CI accepts **5% drift against a hardcoded YAML number**, with
**no GPU baseline column at all**. Its docs tell you to run `lm_eval`, show you a score, and give
you nothing to compare it against.

## But nobody is buying, and that is the finding that matters

- **China Telecom's flagship** 384-card Ascend DeepSeek-671B deployment press-releases throughput,
  TTFT and TPOT — and **never mentions accuracy**.
- **CAICT's national adaptation-verification programme** (AISHPerf) scores five dimensions:
  适配易用性、功能完备性、优化效果、性能、成本. **Accuracy equivalence is not one of them.**
- **中科加禾** — a CAS compiler-institute spinout building precisely this (operator translation and
  **对齐**, alignment), with state backing and 20-year compiler pedigree — cleared only
  **数千万元** in orders in two years and was **acquired by Zhipu** on 2026-07-21.
- **Distributional**, the best-funded company selling AI behavioural testing ($19M), **pivoted three
  days ago** (2026-07-24, now Talaria Scientific). **Kolena** pivoted to document workflow.
  **Patronus** left eval entirely.
- Every model-optimisation vendor that ever published a price was acquired and sunset — **OctoML,
  Deci, Neural Magic**. **ZLUDA** lost two commercial sponsors. **Spectral** exited consulting.
  **Modular** gives its porting engineers away free to sell compute.
- Every well-capitalised Chinese player (无问芯穹 >¥2.2B, 趋境 >¥1B, 硅基流动) converged on the same
  model: **sell tokens and capacity, hide the chip.** SiliconFlow charges the *same* ¥/M-token
  whether the request lands on Ascend or NVIDIA.

> The absence of a discoverable price for this work anywhere in the world, East or West, is more
> plausibly evidence that **the market clears at zero** than evidence of a green field.

## The crux: compulsion and market do not overlap

Voluntary AI-testing spend does not convert — three named companies pivoted out proving it. This
only works where the buyer is **compelled**. Two regulators compel it, quotably:

**EASA** (aviation), AI Concept Paper Issue 02, Objective **IMP-05** — almost our product spec:

> "The differences between the software and hardware of the platform used for training and those
> used for the inference model verification should be identified and assessed for their possible
> impact on the inference model behaviour and performance."

Paired with **IMP-04**: *"the performance metrics for this verification and the associated
acceptable bounds of variation should be documented."*

**FDA** (medical devices), PCCP final guidance, **18 Aug 2025** — binding in practice. Hardware
change is a *named* modification category, and the required evidence is a differential against the
pre-change baseline, with worked examples finding performance *"statistically equivalent to the
baseline performance."*

**Where compulsion does NOT exist:**
- **Financial services.** SR 11-7 and PRA SS1/23 have the change-control machinery but contain
  **zero** mentions of hardware, precision, or accelerators. SS1/23 contains zero mentions of "AI"
  or "machine learning" at all. Vendor blogs claiming otherwise are quoting themselves, not the Fed.
- **China.** The CAC/TC260 regime (GB/T 45654-2025) is **content-safety scoped** — corpus legality,
  31 risk categories, ≥90% safety pass rate. **Nothing on numerical fidelity.**

So: **Ascend has the market but no compulsion. Aviation and medical have compulsion but no Ascend.**

## And the buyer we can reach is not the buyer with the pain

From the market thread, all verified:

- **A HK company cannot issue a fapiao** (Golden Tax System is PRC-entity-only), so SOE and telco
  vendor-onboarding systems cannot register us as a supplier at all.
- **安可测评 is closed** — 《安全可靠测评工作指南 V4.0》 requires the submitting entity to be
  registered in China.
- **SASAC Document 79** mandates strategic SOEs eliminate foreign IT software by **2027**, with
  quarterly progress reporting. That is a rule against our category, not just against NVIDIA.
- **网络安全审查办法** Art. 10 risk factors are written against precisely a HK-domiciled supplier of
  AI infrastructure.
- **No precedent found** of a small foreign or HK software company selling directly to a mainland
  SOE. Every documented path routes through a PRC intermediary at 20–40%.

Mechanics that *do* work: outbound payment only needs tax filing above **USD 50,000** per payment;
the buyer can deduct on a foreign invoice without a fapiao (公告2018年第28号 Art. 11); total
withholding ≈ **12.3%** (6% VAT + 7% EIT under the HK Arrangement); **no ICP filing** needed for a
customer-run CLI. So the *plumbing* is fine. The *access* is not.

## And GP10 makes the Ascend version the highest-risk version

The primary BIS document, extracted verbatim, notifies persons **"in the United States and abroad"**
that GP10 activities — the enumerated verbs include **"use"** and **"otherwise service"** — on
Ascend 910B/910C/910D risk criminal and administrative penalties. The safe harbour is narrow and
explicit: obtaining a chip *"solely for the purpose of technical analysis or evaluation (such as
destructive testing)."* **A commercial tool for making Ascend usable is not covered.**

Note: the **950 series is not on the list**, and no enforcement action exists in fourteen months.
Neither is a clean bill of health, and this fails customer, investor and bank diligence long before
it fails legally.

---

## The genuine gaps, stated honestly

If the project continues, these are the only defensible wedges found:

1. **Communication operators.** msprobe explicitly `不支持通信算子`. That is exactly where the real
   faults are: a CUHK-Shenzhen field study of DeepSeek-V4-Flash on 16 Ascend devices
   ([arXiv:2607.08215](https://arxiv.org/abs/2607.08215)) needed **twelve source patches** and
   disabled features "to preserve numerical correctness" — and **they did not use msprobe**. They
   built their own layer-by-layer oracle diff. That is the strongest signal in the file that the
   free tool does not cover the paying case.
2. **Inference-serving equivalence.** msprobe's GPU-vs-NPU path is **training-only**, dump-based,
   requires manual hook insertion and both boxes with determinism pinned. Nothing validates a
   *served* model over an OpenAI-compatible endpoint.
3. **The signed artifact.** Every existing tool emits a *developer* artifact — CSV, xlsx, console
   table, thrown assertion. **None is signed, timestamped, provenance-bound (weights hash, CANN /
   CUDA / driver versions, seeds), or designed to hand to a procurement officer or an auditor.**
   That is a real gap, and it is the only one a regulator would care about.
4. **Vendor neutrality.** msprobe is Ascend-centric. Nothing compares MetaX vs Ascend vs Moore
   Threads vs NVIDIA on equal terms.

And one hard constraint on any claim we make: **the noise floor is ~9% on the same vendor's
hardware** (arXiv:2506.09501 — bf16 accuracy varies up to 9% from GPU count and batch size alone).
Bitwise equivalence across CUDA and CANN is **not achievable**, and claiming it would be fraud. Any
product must be statistical and tolerance-banded, with a defensible tolerance. MLPerf's
**99% / 99.9% of FP32** is the only publicly defensible precedent in the Western ecosystem.

## Options — Boss's call

**A · Re-aim at compelled buyers (aviation / medical).** Follow the compulsion. EASA IMP-05 and FDA
PCCP are quotable, and no product serves them. Cost: certification-adjacent businesses are slow,
credential-heavy, and need domain partners. Upside: the buyer is *required* to buy, and the
published comparables are certification-body rate cards (BSI €4,356/day), not tool licences.

**B · Drop Ascend from the identity; sell an accelerator-agnostic harness, prove it on NVIDIA↔AMD.**
Both sides rentable by the hour, no export control, and AMD's own docs already instruct users to
*validate numerical correctness between CUDA and ROCm outputs before moving traffic to production*.
Ascend becomes a community plugin the customer runs themselves. Deletes the GP10 gate, the
unreachable-buyer problem, the no-hardware-access problem, and Boss's own QC blind spot in one move.
Cost: no compulsion, so it is voluntary spend — the thing that killed Distributional.

**C · Stay Ascend-first, attack the msprobe gaps** (comm operators + serving-stack + signed report).
Honest assessment: thin reeds, and the buyers who need it cannot transact with a HK entity. Requires
counsel on GP10 and a mainland reseller or WFOE before modelling any revenue.

**D · Stop, and keep the harness as internal tooling for BVI/ShangQiao.** The calibration work,
fault injection and tolerance methodology are genuinely good and transfer to any numerical-output
regression problem.

**My recommendation, for what it is worth:** the cheapest next act is nearly free and works under
B or C — **publish the artifact literally nobody publishes.** Not Huawei, not vllm-ascend, not
CAICT, not MLCommons: *a dated, reproducible, public GPU-vs-alternate-accelerator accuracy delta
table for the top open models, with stated tolerances and full methodology.* Whoever publishes that
first defines the category and becomes the reference everyone cites. On AMD it can be done this
month. On Ascend it cannot be done legally or practically.

## Research caveats

Chinese mandate specifics (the 30% rip-out rule, 50% cap, 80%-by-2028) are **Reuters/FT/SCMP from
anonymous sources — never published as government documents.** Everything US-side (BIS GP10, the Jan
2026 Federal Register rule, NVIDIA's filing, Kessler's testimony) is primary. Not established:
whether PBoC **JR/T 0221-2021《人工智能算法金融应用评价规范》** is a Chinese hook — that title is the
most likely candidate and nobody has read it. Chinese SI pricing (软通动力, 神州数码, 中软国际) remains
unpriced. Policy here is oscillating monthly; a July 2026 read has a short shelf life.
