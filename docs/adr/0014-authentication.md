# ADR-0014 — Self-hosted auth with a BFF token bridge

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`15-security.md`](../15-security.md) §2, [`11-api-spec.md`](../11-api-spec.md) §2, FR-ACC-01…09

## Context

Requirements: email/password and OIDC login, email verification, MFA, session management with
revocation, refresh rotation with reuse detection, and account deletion cascading to biometric data.

Structural constraint: the frontend (Next.js) and the API (FastAPI) are separate services
(ADR-0006). Tokens must not be reachable from client JavaScript, because the app renders
untrusted-derived content (ASR transcripts of user media).

Budget constraint: hobby/student tier — per-user pricing is not viable.

## Options considered

### A — Managed auth (Clerk, Auth0, WorkOS)
**Pros:** Fastest to build; MFA, session management and social login handled; strong security posture
maintained by specialists.
**Cons:** Per-user cost that scales with a free tier we intend to have. A third party holds the user
directory. Vendor lock-in on a core flow. Account deletion must coordinate across two systems, which
complicates the GDPR erasure guarantee.

### B — Supabase Auth / GoTrue
**Pros:** Open source, self-hostable, generous free tier.
**Cons:** Pulls in the Supabase ecosystem; awkward alongside our own Postgres and SQLAlchemy schema.

### C — Fully self-hosted, tokens in client JS
**Pros:** Full control, no cost.
**Cons:** Access token reachable by XSS. Unacceptable given the content we render.

### D — Self-hosted with a BFF token bridge ✅
Auth.js in Next.js owns the browser session (httpOnly cookie). Route handlers exchange it for a
short-lived RS256 JWT and attach it server-side. FastAPI verifies with the public key. User records,
refresh tokens and sessions live in our Postgres.

## Decision

**Option D.**

| Token | TTL | Storage | Reachable by client JS |
|---|---|---|---|
| Session cookie | 30 d | Hashed in DB | ❌ httpOnly |
| Refresh token | 30 d | Hashed in DB | ❌ httpOnly, `Path=/api/auth` |
| Access JWT (RS256) | 10 min | Server memory only | ❌ never sent to the browser |
| Signed media URL | 15 min | — | ✅ (by design — scoped, expiring) |

Passwords: Argon2id (`m=64MiB, t=3, p=4`), HIBP k-anonymity check.
Refresh: rotation with **reuse detection** — a replayed token revokes the entire family.

## Rationale

**No API token exists in the browser**, so XSS cannot exfiltrate one. This is the property that
decides between C and D, and it matters more here than in a typical app because we render text
derived from arbitrary user-uploaded media.

**RS256 rather than HMAC** means the web tier verifies but cannot mint. A compromise of the web tier
does not become a compromise of the API.

**Refresh rotation with reuse detection** is what makes a 30-day refresh token acceptable. Without
it, a stolen refresh token is a month of undetected access. With it, the thief's first use — or the
legitimate user's next use — reveals the theft and revokes everything.

Self-hosting over managed auth is driven by two things beyond cost. First, account deletion must
cascade to biometric derivatives with a verifiable receipt (NFR-PRIV-04); owning the user table makes
that a single transaction rather than a cross-system dance. Second, the free tier is a product
requirement, and per-user auth pricing is fundamentally at odds with it.

The cost is real: we are responsible for getting auth right, and auth is a domain where subtle
mistakes are severe. This is why [`15-security.md`](../15-security.md) §2 specifies the mechanisms
precisely and why the anti-enumeration behaviour (constant-time comparison, dummy hashing on unknown
accounts) is stated explicitly rather than left to implementation.

## Consequences

**Positive** — No per-user cost. Full control over the user model and deletion semantics. Tokens
unreachable from client JS. Refresh theft is detectable.

**Negative** — We own auth security, including MFA, verification emails, reset flows and their edge
cases. More code to test and maintain. Email deliverability becomes our problem.

**Neutral** — Auth.js handles the OIDC dance; we own the resulting user record.

## Revisit when

- Enterprise SSO (SAML, SCIM) is required → managed providers become clearly worth it.
- An auth vulnerability is found in our implementation that indicates a systematic problem rather
  than a single bug.
- Team size grows such that maintaining auth is no longer the best use of engineering time.
