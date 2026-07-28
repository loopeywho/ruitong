# QA — `src/ruitong/auth/` audit (H1, per-customer API keys)

Claude · 2026-07-28 · reviewed against the working tree while `auth/` was still
uncommitted. **Re-run this against a committed SHA before acting on it** — I am
breaking my own rule about auditing a frozen tree because the pods were
billing and I had the time; treat the findings as real but the line numbers as
approximate.

Written to a separate file on purpose: `QA_FINDINGS.md` is modified and
uncommitted in your tree right now, and it has been clobbered once before.

**Verdict: the design is sound.** Fail-closed admin key, no data-plane
fallback, no singleton fallback for the store, hashes never returned by
`list_keys`, plaintext returned only at creation. You clearly read
`LESSONS.md` — the fail-closed reasoning and the "no legacy fallback" comment
are exactly right, and they close H1 properly.

Two findings, one confirmed bug and one design note. Both were found by
*running* the code, not reading it.

---

## F1 (confirmed bug) — `last_used_at` debounce compares two different time formats

`keystore.py:133-144`. The docstring promises "debounced to ~5 min". Measured
behaviour: **the timestamp can be stale by up to ~24 hours.**

The code writes Python's format and compares against SQLite's:

| | value |
|---|---|
| written by `datetime.now(timezone.utc).isoformat()` | `2026-07-28T01:00:00.000000+00:00` |
| compared against `datetime('now','-5 minutes')` | `2026-07-28 13:01:12` |

The predicate is a **string** comparison. Both start `2026-07-28`, then the
stored value has `'T'` (0x54) where the cutoff has `' '` (0x20). `'T' > ' '`,
so any same-day stored value sorts *greater* than the cutoff and the
`last_used_at < cutoff` branch is false. The row only updates once the stored
*date* differs — i.e. roughly once per calendar day.

Reproduced directly against your class:

```
stored (python fmt, same day)  : 2026-07-28T01:00:00.000000+00:00
after re-auth                  : 2026-07-28T01:00:00.000000+00:00
updated?  False   <- 12 hours had passed
```

It is not a security hole — auth still works, and `is_active` is unaffected.
But "when was this key last used" is the field you would rely on to spot a
leaked key or retire a dormant one, and it silently lies.

**Fix:** compare in one format. Simplest is to let SQLite own both sides:

```python
self._conn.execute(
    """UPDATE api_keys SET last_used_at = datetime('now')
       WHERE key_id = ?
         AND (last_used_at IS NULL
              OR last_used_at < datetime('now', '-5 minutes'))""",
    (key_id,),
)
```

Then both sides are `YYYY-MM-DD HH:MM:SS` and the comparison is meaningful.
Add a test that sets `last_used_at` to a same-day value **in the format the
code itself writes** and asserts it updates — a test using SQLite's format
passes against the broken code, which is how this survived.

## F2 (design note, not a vulnerability) — the HMAC has no secret

`keystore.py:81-85` and `98-102` call:

```python
hmac.new(plaintext.encode("utf-8"), b"", "sha256")
```

That is HMAC keyed on *the API key itself* over an *empty message* — a
deterministic hash of the plaintext with no server-side secret and no salt.
It reads like it was meant to be a keyed hash (a pepper), and it isn't one.

**This is acceptable as written**, and I am not asking you to change it
without a decision entry: the tokens are `rt_` + 48 hex = 192 bits of
entropy, so a fast unsalted hash is not brute-forceable, and salting is
pointless for random tokens. Password-hashing advice (bcrypt/argon2) does
*not* apply here.

The gap is what it buys you: with no server-side secret, anyone who steals the
database can verify guesses offline. Against 192-bit tokens that is worthless,
so the practical risk is nil — but if you ever shorten the token, this becomes
load-bearing. Either pass a real secret from config as the HMAC key, or
replace it with plain `hashlib.sha256` and a comment saying why an unsalted
hash is correct for high-entropy tokens. Right now the code says one thing and
does another.

## Checked and NOT a bug

`router.py:36` — `provided.encode("latin-1")`. I expected the non-ASCII crash
I introduced myself in `main.py` (`SECURITY_AUDIT.md` M1). **It is safe here.**
HTTP headers are latin-1 on the wire and Starlette decodes them with latin-1,
so every value reaching `request.headers.get()` is latin-1-encodable by
construction and the round-trip cannot raise. My test appeared to show a crash,
but the `UnicodeEncodeError` came from *httpx on the client side* — the header
never reached the server. Recording it because "verify before reporting"
applies to audits too; I nearly filed a false positive.

`ruitong-keys.db` is covered by `.gitignore:20` (`*.db`). A keystore created in
the CWD cannot be committed by accident.

## Minor

`keystore.py:31` — `default()`'s docstring says "in-memory when no path set",
but `__init__` defaults to `ruitong-keys.db` on disk. The class docstring
(line 23) is the correct one. Fix the `default()` docstring, or drop
`default()` entirely — `router.py` deliberately refuses to fall back to it,
which is the right call, so the singleton may now be dead code.

---

## The habit worth taking from this

F1 is the same failure that has bitten this repo four times now: **two formats
that look interchangeable and are not.** `logprobs: list[float]` vs the real
object; `/v1/models` bare list vs `{"data": […]}`; `list[str]` vs
`list[list[str]]`; and now ISO-8601-with-T vs SQLite-space-separated.

Every one passed its tests, because the test used the same wrong format as the
code. The check that catches all four is the same: **feed the function the
exact bytes the other system actually produces**, and assert on that.
