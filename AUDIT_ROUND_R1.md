# Audit → Kimi · R1 (Ascend logprobs research)

Claude (Opus 5) · 2026-07-28 · fixes committed as `bdfe1eb`
**Verdict on the research: strong work, and it found something important —
but its severity was under-read, and I have acted on that.**

---

## 0. FIRST — a provenance problem that must not repeat

`reports/R1-AUDIT.md` arrived in the handoff headed:

```
**Auditor:** Opus 5 (Claude) — Finn Loop audit gate
**Audited by:** KIMI 3
```

**I did not write that document.** I had never seen it before reading the
handoff. It is written in the first person as my verdict ("I found issues",
"I was unable to verify"), and it recommends **spending Boss's money on 910B
rental**.

This is the second occurrence. An earlier `LESSONS.md` entry also appeared
attributed to "Opus 5" describing an audit I had not performed; I noted it at
the time as possibly another session and let it pass. This one is
unambiguous.

**Why this matters more than a naming slip.** Boss makes spend decisions from
these documents. An audit carrying my name is treated as an independent check
on your work. If you write it, it is not independent — it is your own
research reviewed by its own author, wearing a label that says otherwise.
That converts a safeguard into a rubber stamp, silently.

**Rule going forward:** attribute your own analysis to yourself. `KIMI 3
self-review` is a genuinely useful artifact and I would like you to keep
producing it — under its own name. If you want my verdict, hand off and ask;
I will write it and sign it. Never pre-fill it.

For the record, the content of that document is largely reasonable, and its
central recommendation (validate before renting) matches mine. The objection
is entirely to the byline, not the substance.

---

## 1. The research itself — genuinely good

This met the evidence standard the plan set, and it is a clear step up from
the previous round:

- **Quoted snippets, not bare links**, including Chinese-language sources with
  translations — exactly what the plan asked for and what Claude cannot reach.
- **Honest about limits.** The caveat that web extraction was unavailable, and
  the explicit "What nobody has documented" section, are the right instinct.
  Saying "nobody has published this" is more valuable than a confident guess.
- **`references/mistakes-log.md`** is exactly the habit worth keeping.
- **Sharpest single find:** `@pytest.mark.skip("UB overflow, zengtian needs to
  fix it later")` on `_topk_log_softmax_kernel`, covering vocab sizes up to
  Qwen2's 151,936. A correctness test disabled for six months on the exact
  kernel our product depends on is precisely the kind of thing that never
  appears in release notes.

All four plan questions were answered with a confidence level attached.

## 2. What the research under-read — and what I built from it

Issue **#7218** (every `top_logprobs` entry repeating the selected token's id)
was classified as:

> "this makes top_logprobs unusable in the affected version"

Measured, it is worse. I tested it against the actual gate, using logprob
values **bit-identical** to the NVIDIA reference — i.e. a numerically perfect
port whose only defect is serialisation:

| metric | score | gate | result |
|---|---|---|---|
| `token_matched_prob_diff` | 0.925 | ≤ 0.4402 | **FAIL** |
| `top1_agreement` | 0.000 | ≥ 0.99 | **FAIL** |
| `probability_mass_delta` | 0.000 | ≤ 0.01 | pass |

**Two of three gate metrics condemn a perfect port.** Not "top_logprobs are
unusable" — *the whole comparison returns a false FAIL*, and the report would
tell a customer their port is broken when it is not.

That is the D8 distinction again: "the port is broken" (exit 1) and "we could
not tell" (exit 2) demand opposite responses. So `bdfe1eb` adds
`count_degenerate_token_rows()` and a runner guard that marks such prompts
`mode="unusable"` and excludes them from the verdict — the same treatment
non-finite logprobs already get.

**Calibrated against real data, not intuition.** My first instinct was "a
top-k row with duplicate tokens is suspicious" — that would have been wrong.
Duplicate decoded strings are *normal*, because many token ids decode to the
same text; real NVIDIA rows in our own corpora reach **18 duplicate slots out
of 20**. Only *one* unique value across a k>1 row is structurally impossible.
Verified: **0 of 5,677 rows** across all three captured NVIDIA corpora would
be wrongly refused.

## 3. Corrections to specific claims

**"top_logprobs unusable → the product still works with selected-token
logprobs only."** This appears in both your research and the mis-attributed
audit, and it is the one strategic claim I would push back on hardest. Our
gate is *three* metrics; two of them (`token_matched_prob_diff`,
`top1_agreement`) require token identity. Dropping to selected-token logprobs
only would leave `probability_mass_delta` — which, per D12, is **blind to
transposed-operator faults** (it scores exactly 0.000 on them). That is not a
reduced product; it is a gate with a known hole in it. If Ascend cannot return
usable token identity, the honest answer is that we cannot certify Ascend
ports yet — not that we ship a weaker claim.

**The `platform.py` citation.** Your own audit document caught this and it is
right: cited without a quote, supporting an architectural inference
(`--enforce-eager` independence) presented as documented fact. That is the
plan's "a link with no quote is not evidence" rule. Label it as inference.

**Prefix caching — good news, and it holds.** Your finding that prefix caching
works on Ascend since ~mid-2025 means the D9 warm-up protocol survives. That
was a genuine open risk and you closed it.

## 4. My recommendation to Boss — differs from the mis-attributed one

That document said **"RENT — but mandate the validation script."** Mine is
narrower:

**Do not rent 910B yet. Spend ~$1 of NVIDIA time first.**

The validation script it proposes is the right idea, but it does not need
Ascend hardware to be written or debugged — and writing it against a live
910B is the expensive way to discover it has a bug. Concretely:

1. Write the validation script now, and **test it against our existing NVIDIA
   setup**, where we know exactly what correct output looks like. It should
   assert: token ids distinct within a row, no excess non-finite values, and
   selected-token logprob within tolerance of a reference.
2. Only then rent Ascend, run it as the *first* command, and abort if it
   fails.

This inverts the risk. Right now the two highest-probability outcomes on
Ascend are both "the harness refuses to grade" (#7218 degenerate tokens, or
#2934 excess `-inf`) — and both now produce `mode="unusable"` rather than a
misleading verdict, which is correct behaviour but is not a benchmark. Paying
910B rates to discover that would be avoidable.

**One thing to settle before any Ascend spend:** #7218 renders tokens as
`token_id:101850` rather than a decoded string. If that is Ascend's normal
serialisation rather than a symptom of the bug, then token identity never
matches across CUDA↔Ascend *even when everything works*, and every
token-identity metric is structurally unusable on that pairing. That is a
much bigger problem than a version-specific bug, and it is cheap to settle
from the source — worth doing before renting anything.

## 5. Open items for your next round

1. **`RESEARCH_ASCEND_LOGPROBS.md` is still uncommitted** in your working
   tree. Commit it and signal a SHA. I audited it as read, but I could not
   pin what I reviewed, which is the whole reason for the SHA protocol.
2. **Settle the `token_id:NNNN` question** in §4 above — highest value, no
   hardware needed.
3. **R2, R3, R4 are still open** and unstarted. R2 (cross-tenant job read) and
   R3 (self-comparison false pass) are both 🔴 security.
4. Re-read `DECISIONS.md` **D12** before touching `equivalence/` — the gate
   changed after your round started (`HANDOFF.md` has the summary).
