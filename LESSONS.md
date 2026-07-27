# 瑞通 Ruitong — Lessons for the Implementer

**Audience:** Qwen, Kimi, and any model writing code in this repo.
**Read this before each round.** It is short on purpose.

This is not a list of complaints. Every entry below is a bug that **passed a green test suite and a
code review** before an audit caught it. That is the point: these are the failure classes your
current process does not catch, so they are the ones worth internalising. Each entry gives the
**rule**, the **why**, and the **tell** — how to spot it yourself next time.

At the bottom there is a log. **Append your own mistakes there.** A mistake you record once and
generalise is worth more than ten you fix silently.

---

## L1 · Python evaluates function arguments eagerly — including "defaults"

```python
# WRONG — from_env() runs on EVERY call, even when the attribute exists
config = getattr(request.app.state, "config", BridgeConfig.from_env())

# RIGHT
config = getattr(request.app.state, "config", None)
if config is None:
    config = BridgeConfig.from_env()
```

**Why:** `getattr(a, b, c)` is a normal function call. Python evaluates `a`, `b`, **and `c`** before
`getattr` runs. The third argument is not lazy. This pattern appeared in three middlewares, so a
full `os.environ` read and dataclass construction happened **three times per HTTP request** while
a cached config sat unused.

**The tell:** any function call written inside a "default" or "fallback" position — `getattr`,
`dict.get`, `next(it, expensive())`. If the fallback is expensive or has side effects, it must be
guarded by an `if`, not passed as an argument.

## L2 · Dataclass field defaults are evaluated once, at import

```python
@dataclass(frozen=True)
class BridgeConfig:
    cuda_base_url: str = os.environ.get("RUITONG_CUDA_BASE_URL", "")   # WRONG
```

**Why:** the class body executes at **import time**. That `os.environ.get` runs once, ever. Any env
var set afterwards is invisible forever — config becomes untestable (`monkeypatch.setenv` does
nothing) and unconfigurable at runtime. It *appears* to work in Docker, where env is set before the
process starts, which is exactly why it survives review.

**Right:** read the environment inside a `from_env()` classmethod, at call time.

**The tell:** any function call on the right-hand side of a class-level assignment. Also: mutable
defaults (`list`, `dict`) — use `field(default_factory=...)`, or better a `tuple`, since a frozen
dataclass holding a `list` is not actually immutable.

## L3 · Never compare secrets with `==` or `!=`

```python
if provided != config.api_key:                       # WRONG — leaks via timing
if not hmac.compare_digest(provided, config.api_key): # RIGHT
```

**Why:** `str.__eq__` returns as soon as bytes differ. Response latency therefore reveals how many
leading bytes were correct, and an attacker recovers the key one byte at a time. Rate limiting does
not save you — key recovery needs patience, not volume.

**Applies to:** API keys, tokens, HMAC signatures, password hashes, webhook secrets, CSRF tokens.

**Note:** this codebase already had the right pattern in ShangQiao's `/preview` gate. It regressed
here. **Reusing a known-good pattern from a sibling project is cheaper than rediscovering it.**

## L4 · A test that cannot fail is worse than no test

```python
def test_invalid_role_rejected(self):
    with pytest.raises(ValidationError):
        Message(role=123, content="hi")     # an int — tests the TYPE, not the VALUE
```

This passed while `role: str` accepted `"banana"`. The name promised value validation; the test
only proved pydantic rejects non-strings. It would pass identically with the bug present.

The same class of error appeared again in the metrics tests: every case compared **exactly equal
floats** (`10.0` vs `10.0`), so the suite proved the implementation agrees with itself but never
that it measures anything. The one input that occurs in production — two backends differing by
`1e-4` — was never tested.

**The rule — ask before writing the assertion:**
> *Would this test still pass if I deleted the code it claims to test?*

If yes, it is decorative. For comparison functions specifically: **perturb the input.** Identical
inputs tell you nothing.

## L5 · `async def f() -> AsyncIterator[X]` is not an async generator

```python
class Backend(Protocol):
    async def stream(self, req) -> AsyncIterator[Chunk]: ...   # WRONG
    def stream(self, req) -> AsyncIterator[Chunk]: ...         # RIGHT
```

**Why:** with a `...` body, `async def ... -> AsyncIterator[X]` describes a **coroutine that
returns an iterator** — callers would need `it = await b.stream(req)` before iterating. But every
implementation is an async *generator* (`async def` + `yield`), which returns the iterator directly
and is used as `async for chunk in b.stream(req)`. mypy rejects the mismatch:

```
Expected: def stream(...) -> Coroutine[Any, Any, AsyncIterator[str]]
Got:      def stream(...) -> AsyncIterator[str]
```

The tests passed because `Protocol` is **not enforced at runtime** without `@runtime_checkable`.

**The tell:** protocol/ABC signature errors are invisible to runtime tests by construction. Run
`uv run mypy src/ruitong` every round. If mypy reports noise (e.g. pydantic false positives),
**fix the tooling** — a type checker people have learned to ignore catches nothing.

## L6 · Guard the inner structure, not just the outer

`_ensure_lists` checked `len(a) == 0` but let `[[]]` through — an outer list containing an empty
inner list. Validation passed, then `max()` on an empty sequence crashed downstream.

**The tell:** when validating nested data, write a test for the *empty inner* case explicitly.
`[]`, `[[]]`, and `[[], [1.0]]` are three different inputs.

## L7 · Coverage you cannot measure is coverage you do not have

`tests/test_cli.py` exercises the CLI properly — via `subprocess`. But `coverage.py` cannot
instrument a subprocess, so `cli.py` reports **0%** and the ≥90% gate cannot be applied to the
product's primary surface. Nobody can see which error paths are covered.

**Right shape:** make the entry point callable in-process — `def main(argv: list[str] | None = None)
-> int` returning an exit code instead of calling `sys.exit()` internally. Test that function
directly (fast, measurable), and keep **two** subprocess tests to prove packaging works end to end.

## L8 · Security: never trust a client-supplied header for a limit or an identity

Two live examples in `main.py` (see `SECURITY_AUDIT.md` S2, S4):

- The payload cap reads `Content-Length`. A `Transfer-Encoding: chunked` request has none, so the
  cap is skipped entirely. **Enforce limits while reading the body, not from a header.**
- The rate limiter keys on `request.client.host`. `ruitong.io` is Cloudflare-proxied, so that is
  always a Cloudflare IP — every user shares one bucket and the limiter becomes an outage.
  Read `CF-Connecting-IP` **only** when the peer is a trusted proxy; trusting it unconditionally
  makes the limiter spoofable, which is worse than having none.

**Also:** any unbounded `dict` keyed by something an attacker controls (IP, user-agent, job id) is
a memory-exhaustion vector. Bound it or evict it.

## L9 · Do not build what the plan does not contain — and verify the plan actually says it

`POST /v1/port` was built with a job store, auth, and rate limiting. It was never in `PLAN.md`. It
entered the project because an agent **described a plan it had not read**, inventing the endpoint to
make an argument vivid — and the invented detail became a build target.

**Two rules:**
1. **Before citing a file, read it.** Quote or grep the line. Never cite a symbol, line number, or
   endpoint from memory. Three fabricated citations survived review here; one got implemented.
2. **Scope comes from `PLAN.md` and `DECISIONS.md`.** If you believe something is missing, propose
   it as a decision entry — do not build it and let the plan catch up.

## L10 · Verdicts come from execution, not from reading

An autonomous audit reported "7 failing tests" that did not exist, and separately produced eight
citations whose line numbers were **all wrong** (`to_dict` cited at 180–195, actually at 64) —
because it described code it had not opened. Meanwhile it missed a metric that would have failed
every real comparison, because catching that required *running* the function.

**Before any status claim or audit verdict:**

```bash
uv run --extra dev pytest -q
uv run mypy src/ruitong
```

Quote the real output. And **commit first, then audit, and cite the SHA** — a tree being written to
cannot be audited. Two false findings this session came from sampling a repo mid-write.

---

# Mistake log — append your own

Format: `## [date] model — one-line title`, then **What I did**, **Why it was wrong**,
**Generalised rule**. Keep each entry under ten lines. Recording the *class* of error matters more
than the instance.

## [2026-07-26] qwen — Copied a parent `__init__.py` into a subpackage
**What:** `src/ruitong/jobs/__init__.py` was given the parent module's docstring plus
`from . import jobs`, which imports the package from inside itself. Circular import; **all** test
collection failed.
**Why wrong:** the line belonged in `src/ruitong/__init__.py` (to expose the subpackage), not in the
subpackage's own `__init__`.
**Rule:** after copying a file as a template, re-read every line and ask whether it is true *in the
new location*. `from . import X` inside `X/__init__.py` is always self-referential.

## [2026-07-27] claude — Filed a finding from a stale sample, and nearly filed a second
**What:** reported "7 failing tests, 9 mypy errors" from a tree read minutes earlier while Qwen was
mid-write; all were transient. Separately, almost reported "the CLI is untested" when it is tested
via subprocess, and almost reported a metric as broken when my own test used a degenerate case
(`k=5` with only 5 elements makes the top-5 set trivially complete).
**Why wrong:** both came from acting on a snapshot instead of re-verifying at the moment of writing.
**Rule:** re-run the check immediately before you write the claim. When a result looks alarming,
first ask whether the *test* is wrong — construct the realistic case before escalating.

## [2026-07-27] claude — Two false-pass paths that a "green" suite could not see
**What:** (a) `--reference a=http://x --candidate b=http://x` — different names, same
URL — passed the self-comparison guard, compared an endpoint with itself, always agreed,
exit 0. (b) An unreachable backend returned exit 1 (gate failed) instead of 2 (cannot run),
because the runner swallows connection errors into warnings.
**Why wrong:** the guard checked the *label* rather than the *thing*; and "broken port" vs
"we could not measure" were collapsed into one code, so an outage reads as evidence.
**Rule:** when guarding against comparing a thing with itself, compare the **identity that
matters** (the URL/target), not the name someone typed. And never let "no data" share an
exit code with "bad data" — they demand opposite responses.

## [2026-07-27] claude — Measured the exit code through a pipe and got the wrong answer
**What:** reported "exit 0 despite Passed: NO" and nearly rewrote a working code path. The
command was piped through `head`, so `$?` was head's status, not the CLI's.
**Why wrong:** in a pipeline, `$?` is the *last* command's status.
**Rule:** test exit codes with the command alone, or `set -o pipefail`. Re-running it
properly is what exposed the two real bugs above — the false reading was hiding them.

## [2026-07-27] measurement-layer — Nearly rewrote main.py that already had the fixes
**What:** context compaction showed an 8,411-char main.py with auth-first registration and
defaultdict buckets. Opus 5 had already committed H2/S3/S4 fixes at 02:53 (2b28b3b). My
write_file of the "new" version matched the committed file byte-for-byte — no-op.
**Why wrong:** the compaction summary was a snapshot from earlier in the session; the tree
had moved on while I was blocked.
**Rule:** always `git log --oneline -3` and quick-read the relevant file before starting
work on it. The compaction summary is a reference, not the current state.

## [2026-07-27] claude — A mutation test left a stale .pyc and I believed the fake failures
**What:** temporarily loosened a threshold to prove the sensitivity suite wasn't vacuous, restored
the file, and kept seeing 3 failures. `grep` showed the file was correct (0.05); Python was
importing 99.0 from a cached `__pycache__` of the mutated module. I nearly started "fixing" tests
that were already right.
**Why wrong:** editing a module out-of-band (sed/cp rather than a normal write) can leave bytecode
that no longer matches source, and the mismatch is invisible to `grep`.
**Rule:** after any out-of-band edit or mutation experiment, clear `__pycache__` and `.pytest_cache`
before trusting a result. And when source and behaviour disagree, print the value **as imported**
(`python -c "from mod import X; print(X.FIELD)"`) rather than reading the file — the interpreter's
view is the one that matters.

## [2026-07-27] claude — Calibrated on full-vocab data, gated on top-k data
**What:** added a probability-mass check comparing `sum(exp(logprob))` against 1.0. Calibration used
a full 512-token vocabulary where that holds. But an OpenAI server returns only the **top-k** of a
~150k vocabulary, whose mass is legitimately ~0.16, so the check failed every correct run.
**Why wrong:** the calibration fixture and the production input were different shapes. A threshold is
only valid for the data shape it was measured on.
**Fix/rule:** compare the mass **between the two backends** instead of against an absolute — a
scaling fault shifts one side relative to the other and needs no full distribution. Measured
afterwards: 0.00048 for an equivalent BF16 port vs 0.0952 for a x1.05 scaling fault. **Always
calibrate on the exact shape the wire delivers.**

---

# Round 5 — first contact with real hardware (2026-07-27)

Notes for Qwen / Kimi / KIMI-coder. Every entry below is a mistake or blind
spot that **only a real GPU exposed**, after 202 green tests and clean mypy.
Log your own the same way — the point of this file is that the next model does
not repeat what the last one learned.

## [2026-07-27] claude — Synthetic fixtures agreed with themselves for five rounds
**What:** the gate was calibrated on synthetic logprobs because no GPU was
available. On a real A40 the primary metric scored **1.25 between two correct
executions** against its own **0.05** threshold — it rejected the reference
model compared with itself. The weakest real fault scores 0.406, so noise and
faults *overlapped*: no threshold could have worked.
**Why the tests missed it:** synthetic logprobs had a convenient distribution.
A real 8B model's top-10 spans logprob 0 to −29; the synthetic one did not.
Every test agreed with the fixture, and the fixture was wrong.
**Rule:** a threshold is only valid for the data distribution it was measured
on. **Synthetic calibration is a placeholder, never a result.** Say so in the
doc, and re-measure before anyone quotes the number.

## [2026-07-27] claude — Two token ids can decode to the same string
**What:** wrote `dict(zip(tokens, logprobs))` to look up a token's logprob.
One position in the real corpus contained **nine distinct token ids all
decoding to `""`**. The dict kept only the last, so rank 1 was compared against
rank 17 and the metric scored **0.67 comparing a tensor with itself**.
**Why wrong:** the OpenAI wire format carries the *decoded string*, not the
token id. Strings are not unique keys.
**Rule:** never key on a decoded token string. Pair repeated strings by order
of appearance, and **always test a comparison metric for reflexivity** —
`metric(x, x)` must be exactly 0. That one assertion catches a whole class of
bug in one line, and it is the first test I should have written.

## [2026-07-27] claude — `list[str]` and `list[list[str]]` are indistinguishable at runtime
**What:** passed already-extracted rank-0 tokens to a function expecting full
top-k rows. `row[0]` then yielded a *character* instead of a token, and the
metric silently returned 0.984 instead of 1.0 — plausible, and wrong.
**Why wrong:** a `str` is itself a sequence of `str`, so duck typing cannot
tell the two shapes apart, and no exception is raised.
**Rule:** when a function takes nested string sequences, **reject the flat
shape explicitly** (`isinstance(rows[0], str) -> raise`). Type hints do not run
at runtime. A silently-wrong number is worse than a crash.

## [2026-07-27] claude — Asserted the cause before measuring it
**What:** saw a 404 from the endpoint and told Boss the proxy was auth-gating
it. Then saw `{"error":"Unauthorized"}` from *inside* the pod — it was vLLM's
own API key all along. Later saw a 28.56 difference and was about to build a
theory on it; the actual worst case across three fresh prompts was ~0.5, and
the 28.56 was a single deep-rank token swap.
**Why wrong:** both times I generalised from one observation to a mechanism.
**Rule:** before naming a cause, get a second measurement that *discriminates*
between the candidate causes. For the cache finding I did this properly —
read vLLM's `prefix_cache_hits_total` across a cold and a warm call (+0 then
+16) — and that turned a plausible story into a proven one. Do that first, not
after being wrong once.

## [2026-07-27] claude — `list_models` had never parsed a real response
**What:** `response.json()` returns the OpenAI envelope
`{"object":"list","data":[...]}`, not a bare list. The code checked
`isinstance(ids, list)`, found `False`, and wrapped the *envelope* in a list —
so every model id resolved to `""` and `models_served` counted the envelope's
two keys.
**Why the tests missed it:** every fixture returned a bare list, which no
OpenAI-compatible server has ever emitted. Same root cause as the logprobs
wire-format bug in round 4.
**Rule:** **pin at least one test to a payload captured verbatim from a real
server**, with a comment saying where and when it came from. If you cannot
capture one, the code is unverified — say so rather than shipping green tests.

## The meta-lesson for this round

Four separate bugs, one shared cause: **a fixture that returned a shape no real
server produces.** Test count, coverage percentage and a clean type-check
cannot detect this, because they all measure agreement with the fixture.

The only thing that detects it is contact with a real system. So:

1. Get *something* real as early as possible, even one endpoint for one hour.
2. Capture its output to disk (`corpora/`) and commit it. GPU time is scarce;
   analysis against a saved corpus is free and repeatable.
3. Make the regression suite run against that corpus, not against fixtures.
4. Mutation-test the suite: break the thing on purpose and confirm the tests
   fail. A test that cannot fail is documentation, not a test.
