# 15 — Security

Target: **OWASP ASVS Level 2**. This system has an unusual property that shapes everything below —
its core function is **decoding attacker-supplied binary media**.

---

## 1. Threat model (STRIDE)

| Threat | Vector | Impact | Control |
|---|---|---|---|
| **Spoofing** | Credential stuffing | Account takeover | Argon2id, rate limit, MFA, breach-password check |
| | Session hijack | Full access | httpOnly+Secure+SameSite, short JWT, rotation w/ reuse detection |
| | Share-link guessing | Private media exposure | 128-bit tokens, hashed at rest, expiry, rate limit |
| **Tampering** | Direct object mutation | Data corruption | Ownership checks on every access; presigned URLs are single-key, single-method |
| | Job manipulation | Resource abuse | Server-side state machine; workers trust only DB state |
| **Repudiation** | Denying an upload | Compliance / rights disputes | Append-only audit log with IP hash + timestamp + attestation |
| **Information disclosure** | IDOR on artifacts | ⭐ Private recordings leak | Ownership check + short-lived signed URLs + no public bucket |
| | Verbose errors | Recon | Problem Details only; no internals |
| | Timing on auth | User enumeration | Constant-time compare, dummy hash on unknown users |
| | Model inversion of voiceprints | Biometric leak | Ephemeral by default; encrypted; never returned by the API |
| **Denial of service** | ⭐ **Media decode bomb** | Worker crash / RCE | Sandbox + ffprobe pre-validation + hard caps |
| | GPU exhaustion | ⭐ **Cost DoS** | Quotas, concurrency cap, duration cap, budget alerts |
| | Upload flooding | Storage cost | Per-user quota, size cap, presign rate limit |
| **Elevation of privilege** | ⭐ **RCE via ffmpeg CVE** | Worker compromise | Sandbox: non-root, seccomp, no network, read-only rootfs, no cloud credentials |
| | SSRF via URL ingest | Internal network access | Feature not built; if added — allowlist, no redirects, block link-local/RFC1918 |
| | Container escape | Host compromise | gVisor/Kata, dropped capabilities, no privileged containers |

The three starred rows are the ones specific to this product. Everything else is standard web
security. The starred ones deserve most of the effort.

---

## 2. Authentication

### Passwords
- **Argon2id**, `m=64MiB, t=3, p=4` — tuned so hashing takes ~250 ms on production hardware
- Minimum 12 characters; no composition rules (they reduce entropy in practice)
- Checked against Have I Been Pwned via k-anonymity range query (only a 5-char SHA-1 prefix leaves us)
- Never logged, never in error messages, never in a URL

### Tokens
| Token | Lifetime | Storage | Transport |
|---|---|---|---|
| Access JWT (RS256) | 10 min | Memory, server-side only | `Authorization` header, API only |
| Refresh | 30 days | Hashed in DB | httpOnly cookie, `Path=/api/auth` |
| Session cookie | 30 days | Hashed in DB | httpOnly, Secure, SameSite=Lax |
| Share token | configurable | Hashed in DB | URL path |
| Signed media URL | 15 min | — | Query string |

**Refresh rotation with reuse detection.** Each refresh issues a new token and marks the old one
used. Presenting an already-used token means it was stolen: revoke the whole `family_id`
immediately, log an audit event, and notify the user. Without reuse detection, refresh tokens are
just long-lived credentials.

**Asymmetric JWT (RS256), not HMAC.** The API signs; the Web BFF verifies with the public key. A
compromise of the web tier cannot mint tokens.

### OIDC
Google and GitHub via Auth.js. PKCE, `state` and `nonce` validated. Email must be provider-verified
before linking to an existing account — otherwise account takeover by registering an unverified
email at the provider.

---

## 3. Authorization

Single rule, applied without exception:

```python
async def require_project(project_id: str, user: User, db) -> Project:
    p = await db.get(Project, project_id)
    if p is None or p.deleted_at or p.user_id != user.id:
        raise NotFound()          # 404, never 403 — 403 confirms the resource exists
    return p
```

- Enforced at the **data-access layer**, not in route handlers — a new endpoint cannot forget it
- `404` rather than `403` for cross-tenant access: a `403` is an existence oracle
- Share-link access resolves to a scoped read-only principal with its own permission set
- Workers authorise by `job_id` from the queue message, validated against the DB — never trusting
  a path supplied in the message

**Test enforcement:** an automated suite iterates every route in the OpenAPI spec and asserts that
User B receives `404` for every one of User A's resources. New routes are covered automatically
because the suite is generated from the spec.

---

## 4. Untrusted media handling ⭐

The highest-risk subsystem. FFmpeg is a large C/C++ codebase with a long CVE history, and we run it
on arbitrary attacker-supplied files.

### Defence in depth

```
1. CLIENT PRE-CHECK           advisory only, never trusted
2. PRESIGN GUARD              size and quota checked before a URL is issued
3. MAGIC-BYTE CHECK           first 16 bytes must match the declared container
                              — extension and Content-Type are ignored entirely
4. FFPROBE (sandboxed)        5 s timeout, 256 MB memory, no network
                              → reject: duration > 60 min, w/h > 4096,
                                fps > 120, streams > 10, channels > 8, no audio,
                                nb_frames inconsistent with duration
5. FFMPEG (sandboxed)         full decode under the constraints below
6. OUTPUT VALIDATION          expected artifacts exist, sane sizes, correct durations
```

Step 3 is what stops polyglot files. Step 4 is what stops decompression bombs — a 2 MB file
declaring 32000×32000 frames is rejected before any decode happens.

### Sandbox specification

```yaml
# every ffmpeg/ffprobe invocation, without exception
runtime: gvisor                    # or kata — syscall isolation, not just namespaces
user: 65534:65534                  # nobody
capabilities: { drop: [ALL] }
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  seccompProfile: { type: Localhost, localhostProfile: ffmpeg-restricted.json }
network: none                      # ⭐ ffmpeg has network protocol handlers — disable the network
resources:
  limits: { memory: 2Gi, cpu: 2, ephemeral-storage: 10Gi }
volumes:
  - { name: in,  readOnly: true }
  - { name: out, readOnly: false }
timeout: 900s                      # hard wall clock kill
env: {}                            # ⭐ NO cloud credentials in this container
```

```bash
# additionally, at the command level
ffmpeg -nostdin \
       -protocol_whitelist file \      # ⭐ no http/tcp/concat protocol handlers
       -f mp4 \                        # explicit demuxer — never let it probe freely
       -threads 2 \
       -i input.mp4 …
```

`-protocol_whitelist file` and `network: none` close the class of attacks where a crafted container
makes ffmpeg fetch a remote resource (SSRF) or read a local path.

**No cloud credentials in the sandbox container.** The orchestrating worker downloads the input to a
scratch volume, runs the sandboxed process on that volume, and uploads the output. A compromised
ffmpeg process gets a scratch directory and nothing else — no bucket access, no network, no
identity.

### Limits

| Property | Limit |
|---|---|
| File size | 2 GB |
| Duration | 60 min |
| Resolution | 4096×4096 |
| Frame rate | 120 fps |
| Streams | 10 |
| Audio channels | 8 |
| Decode wall clock | 15 min |
| Decode memory | 2 GB |
| Output size | 5× input |

---

## 5. Input validation

- **Pydantic v2** models on every endpoint; unknown fields rejected (`extra='forbid'`)
- Path parameters validated as ULIDs by regex before any DB call
- SQL exclusively through SQLAlchemy parameterised queries — no string interpolation anywhere
- Filenames sanitised and never used as storage keys; keys are server-generated from IDs
- Rich text: none accepted. All user text is plain and escaped on render.
- File uploads: only via presigned URLs to a dedicated bucket prefix; the API never receives bytes

---

## 6. Frontend security

```
Content-Security-Policy:
  default-src 'none';
  script-src 'self' 'nonce-{RANDOM}' 'strict-dynamic';
  style-src 'self' 'nonce-{RANDOM}';
  img-src 'self' data: blob: https://cdn.visiovox.app;
  media-src 'self' blob: https://cdn.visiovox.app;
  connect-src 'self' https://api.visiovox.app;
  font-src 'self';
  worker-src 'self' blob:;
  frame-ancestors 'none';
  base-uri 'none';
  form-action 'self';
  object-src 'none';
  upgrade-insecure-requests;
```

Notes:
- Per-request **nonce**; `strict-dynamic` so bundler-injected scripts work without `unsafe-inline`
- No `unsafe-eval` — this constrains three.js usage; verify shader compilation paths don't need it
- `blob:` in `media-src` and `worker-src` is required by Web Audio and hls.js
- `frame-ancestors 'none'` — clickjacking

Additional headers: HSTS with preload, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` denying camera, microphone
and geolocation (the app needs none of them), COOP `same-origin`, CORP `same-site`.

**XSS:** React escapes by default; `dangerouslySetInnerHTML` is banned by an ESLint rule.
Captions come from ASR of user media — untrusted — and are rendered as text nodes only.

**CSRF:** `SameSite=Lax` plus a double-submit token on state-changing requests. Media URLs are
presigned and require no cookie, so they are unaffected.

---

## 7. Secrets

| Secret | Storage | Rotation |
|---|---|---|
| DB credentials | Cloud secret manager, injected at runtime | 90 d |
| JWT signing key | Secret manager; public key distributed | 180 d, with overlap |
| Storage credentials | Workload identity where available, else secret manager | 90 d |
| OIDC client secrets | Secret manager | On provider rotation |
| API keys (3rd party) | Secret manager | 90 d |

- No secret in git, in an image layer, in a build arg, or in an env file committed anywhere
- `gitleaks` in pre-commit **and** in CI (pre-commit is bypassable)
- Local development uses `.env.local`, gitignored, seeded from `.env.example` with dummy values
- JWT rotation supports two active public keys during the overlap window so rotation is zero-downtime

---

## 8. Supply chain

| Control | Tool |
|---|---|
| Lockfiles committed and CI-verified | `uv.lock`, `pnpm-lock.yaml` |
| Dependency CVE scanning | Dependabot + `pip-audit` + `pnpm audit` |
| Container scanning | Trivy — build fails on CRITICAL |
| SBOM per build | Syft → CycloneDX, attached to the release |
| Image signing | cosign; admission verifies signatures |
| Base images | Distroless or Chainguard; pinned by digest, never by tag |
| GitHub Actions | Pinned by commit SHA, not by tag (tags are mutable) |
| Model weights | SHA-256 verified against a manifest before load |

The model-weights check matters: a compromised HuggingFace download is arbitrary code execution
inside the GPU worker via pickle deserialization. Prefer `safetensors`; verify checksums; never load
a `.pt`/`.pkl` from an unverified source.

---

## 9. Rate limiting and abuse

Layered:

| Layer | Control |
|---|---|
| CDN/WAF | IP reputation, DDoS absorption, geo rules if needed |
| API global | 1000 req/min per IP |
| API per-endpoint | Table in [`11-api-spec.md`](./11-api-spec.md) §10 |
| Auth | Exponential backoff per account after failures |
| Quota | Uploads/day, media-minutes/month, **GPU-seconds/month** |
| Concurrency | Max simultaneous jobs per user |
| Budget | Cloud spend alerts at 50/80/100% with automatic job admission pause at 100% |

The GPU-seconds quota and the budget cut-out are the controls that actually bound financial risk.
Rate limiting alone does not — a user within every rate limit can still queue 50 hour-long videos.

---

## 10. Logging and audit

**Logged:** authn success/failure, authz denial, upload init/complete/reject, job lifecycle,
artifact download, share create/access/revoke, deletion, settings changes, admin actions.

**Never logged:** passwords, tokens (even hashed), full IPs (hashed with a rotating salt), media
content, transcript text, speaker embeddings, email bodies.

Audit events are append-only. Retained 1 year, then archived. Account deletion nulls `user_id` but
preserves the row ([`10-data-model.md`](./10-data-model.md) §2) — erasure of identity, retention of
the security record.

---

## 11. Incident response

| Severity | Definition | Response | Comms |
|---|---|---|---|
| SEV1 | Data breach, RCE, mass unauthorised access | Immediate; page | Users within 72 h (GDPR Art. 33) |
| SEV2 | Auth bypass, single-tenant leak | < 1 h | Affected users |
| SEV3 | DoS, degraded service | < 4 h | Status page |
| SEV4 | Low-impact vuln | Next sprint | — |

Procedure: contain (revoke sessions, disable endpoint, block IP) → preserve evidence → assess scope
from audit logs → eradicate → recover → post-mortem within 5 working days, blameless, with action
items tracked to completion.

`SECURITY.md` publishes a disclosure address, a 90-day coordinated disclosure policy, and a safe
harbour statement for good-faith research.

---

## 12. Pre-launch checklist

- [ ] Cross-tenant access suite passes for every route in the spec
- [ ] Sandbox verified: escape attempt, network attempt, fork bomb, decode bomb, symlink escape
- [ ] Malformed-media corpus processed without a worker compromise
- [ ] CSP has no `unsafe-*`; report-only mode run for 2 weeks with zero legitimate violations
- [ ] Rate limits verified under load
- [ ] Secrets scan clean across full git history (not just HEAD)
- [ ] Trivy: zero CRITICAL
- [ ] TLS: A+ on SSL Labs; HSTS preload submitted
- [ ] Presigned URLs verified to expire and to be method- and key-scoped
- [ ] No public bucket access (verified by an IaC policy test, not by inspection)
- [ ] Deletion verified end-to-end with a receipt
- [ ] Dependency audit clean
- [ ] Independent penetration test (or, at minimum, a structured self-assessment against ASVS L2)
- [ ] `SECURITY.md` published with a working contact
