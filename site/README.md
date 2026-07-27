# ruitong.io — public site

Single self-contained `index.html`. No build step, no dependencies, no external requests
(everything inline — safe under a strict CSP, and it loads behind the Great Firewall because
nothing is fetched from a CDN).

Bilingual EN / 中文 via a toggle; light + dark; responsive.

## Deploy — Cloudflare Pages (recommended)

`ruitong.io` is already on Cloudflare (nameservers `chuck`/`donna.ns.cloudflare.com`).

1. Cloudflare dashboard → **Workers & Pages** → Create → **Pages** → Direct Upload
2. Upload the `site/` directory
3. Custom domain → `ruitong.io`
4. **Delete the A record** pointing at `72.62.255.195` — Pages supplies its own routing, and the
   VPS should not serve this.

Why Pages rather than the VPS: a static page needs no server, so it carries **none** of the open
findings in `SECURITY_AUDIT.md`. Putting a marketing page on the box that runs the API would drag
an unaudited HTTP surface into public DNS for no benefit.

## Before it goes live — Boss's calls

1. **Contact method.** Both language versions end with "get in touch" / "欢迎联系" and there is no
   address. Needs a real one, or the page has no conversion path at all.
2. **Company identity.** No legal entity is named. Decide what appears in the footer before
   publishing — this touches the HK-entity question in `STRATEGY.md`.
3. **Publish gate.** The measured fault-sensitivity table is real and reproducible today. The
   accuracy-delta table is explicitly marked *pending hardware runs* — it must stay that way until
   real numbers exist. **Never fill it with estimates.** The page's entire value is that it
   publishes only what was measured.

## Content rules

- No invented numbers. Every figure traces to a run.
- Ascend stays labelled `v3` until counsel clears GP10 (`RESEARCH.md`).
- Never claim bitwise equivalence — it is unachievable across vendors.
- Name no vendor as "better" or "worse". Report differences only.

## Set up the inbox — 2 minutes, free

The contact form composes a `mailto:` to **hello@ruitong.io**. That address needs to exist:

1. Cloudflare dashboard → `ruitong.io` → **Email** → **Email Routing** → Enable
2. Create address: `hello@ruitong.io` → forward to your real inbox
3. Verify the destination (Cloudflare emails you a link)

Cloudflare adds the MX and SPF records itself. Free, no mailbox to run, and **your personal address
is never published** — visitors only ever see `hello@ruitong.io`.

### Why a `mailto:` form and not a hosted form

No backend, so nothing to secure — none of the open findings in `SECURITY_AUDIT.md` apply to a
static page. And no third-party form service sits between a prospect and your inbox reading your
leads. The trade-off is that it opens the sender's mail client; the direct address is shown
alongside for anyone who prefers to write themselves.

**If you later want a true in-page form** (no mail client), the smallest step is a Cloudflare Pages
Function posting to an email API. That adds a backend and therefore a security surface — worth it
once volume justifies it, not before.
