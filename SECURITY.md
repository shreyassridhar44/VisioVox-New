# Security Policy

## Reporting a vulnerability

**Do not open a public issue.**

Email **security@visiovox.app** with:
- Description and impact
- Steps to reproduce
- Affected component and version
- Any proof-of-concept (please don't test against other users' data)

You'll get an acknowledgement within **48 hours** and an assessment within **5 working days**.

## Coordinated disclosure

We follow a 90-day disclosure timeline. If a fix ships sooner, we'll agree a disclosure date with
you. We'll credit you unless you'd rather stay anonymous.

## Safe harbour

Good-faith security research is welcome. We will not pursue legal action for research that:

- Stays within your own account and test data
- Does not access, modify or exfiltrate other users' data
- Does not degrade service for others (no DoS, no automated scanning at volume)
- Does not use social engineering or physical attacks
- Reports promptly and gives us reasonable time to fix

If you're unsure whether something is in scope, ask first.

## In scope

- `visiovox.app`, `api.visiovox.app`, `cdn.visiovox.app`
- Authentication, session handling, token rotation
- **Media processing sandbox** — escape, credential access, network egress
- Cross-tenant access (IDOR) on projects, artifacts, captions, shares
- Presigned URL scope, expiry and forgery
- Rate limiting and quota bypass
- CSP and header bypasses leading to XSS
- Injection of any kind

Particularly interested in the media processing path — it is the highest-risk component and the one
we most want independent eyes on.

## Out of scope

- Missing headers with no demonstrated impact
- Rate limiting on unauthenticated read-only endpoints
- Self-XSS, clickjacking on pages with no state-changing action
- Vulnerabilities requiring physical access or a compromised device
- Reports from automated scanners with no verified impact
- Social engineering
- Attacks against third-party services we use
- Model output quality issues — that's a bug report, not a vulnerability

## Severity and response

| Severity | Definition | Fix target |
|---|---|---|
| Critical | RCE, mass data exposure, auth bypass | 24 h |
| High | Single-tenant data exposure, privilege escalation | 7 d |
| Medium | Limited disclosure, DoS | 30 d |
| Low | Defence in depth, minor leakage | 90 d |

## Our commitments

- Encryption in transit (TLS 1.3) and at rest (AES-256)
- All media decoding runs in a hardened, credential-free, network-isolated sandbox
- Every artifact access is ownership-checked; media URLs are signed and expire in ≤ 15 minutes
- **Voiceprints and face crops are deleted when a job finishes** — there is no biometric database
- Breach notification within 72 hours where legally required
- Dependency scanning, SBOM and image signing on every build

Details: [`docs/15-security.md`](./docs/15-security.md) ·
[`docs/16-privacy-compliance.md`](./docs/16-privacy-compliance.md)

## Supported versions

Pre-release. Once launched, the current production release receives security fixes.
