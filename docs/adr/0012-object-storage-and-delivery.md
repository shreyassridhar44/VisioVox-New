# ADR-0012 — Cloudflare R2 for object storage and delivery

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`17-infrastructure-deployment.md`](../17-infrastructure-deployment.md) §9, [`20-performance-cost.md`](../20-performance-cost.md) §4, NFR-SEC-02

## Context

The system stores and serves large media: source uploads (up to 2 GB), per-speaker audio tracks,
HLS segments, thumbnails, captions. Every playback session streams video plus N audio tracks.

**Egress is the dominant variable cost in any video product.** A 10-minute project with 3 speakers
serves roughly 500 MB per full viewing session.

Security requirement: no public bucket access; every read must be an ownership-checked, short-lived
signed URL (NFR-SEC-02).

## Options considered

### A — Local filesystem (the archived roadmap's choice)
**Pros:** Zero cost, zero setup.
**Cons:** No durability, no horizontal scaling, no CDN, no signed URLs, media served through the
application. Incompatible with a production deployment. Rejected under F4.

### B — AWS S3 + CloudFront
**Pros:** Industry standard; deepest feature set; excellent tooling.
**Cons:** **Egress costs.** At the modelled ~1000 jobs/month with typical viewing, egress adds roughly
$200–400/month — more than all other infrastructure combined, and it scales linearly with usage in a
way the free tier cannot absorb.

### C — Cloudflare R2 ✅
**Pros:** **Zero egress fees.** S3-compatible API, so tooling and code are unchanged. Integrated CDN.
Presigned URLs supported. Competitive storage pricing.
**Cons:** Smaller ecosystem than S3. Fewer regions for data residency. Some S3 features absent
(certain lifecycle and replication options). Vendor concentration with the CDN/WAF.

### D — Backblaze B2 + Cloudflare
**Pros:** Free egress via the Bandwidth Alliance; cheap storage.
**Cons:** Two vendors to manage; less integrated; weaker S3 compatibility in practice.

## Decision

**Cloudflare R2**, S3-compatible API, MinIO for local development and CI.

Layout, lifecycle and access control per
[`09-system-design.md`](../09-system-design.md) §6:
- No public access on any bucket — verified by an IaC policy test, not by inspection
- All reads via presigned URLs, ≤ 15 min, method- and key-scoped
- Direct-to-storage multipart upload; the API never proxies bytes
- `work/` artifacts lifecycle-deleted after 7 days

## Rationale

The egress line dominates the decision. Per
[`20-performance-cost.md`](../20-performance-cost.md) §4, cost per job is ~$0.062 on R2 and ~$0.107
on S3+CloudFront — nearly double, entirely because of egress. For a product with a free tier and a
student budget, that single line determines whether the unit economics work.

The lock-in risk is low because **R2 is S3-compatible**. Application code targets the S3 API; moving
to S3, B2 or MinIO is a configuration change plus a data transfer, not a rewrite. That is what makes
choosing the cheaper option safe here — the exit cost is bounded and known.

Direct-to-storage upload is not merely a cost optimisation. It removes the largest bandwidth path
from the application, the largest DoS surface, and the most common source of request-timeout
failures, all in one decision.

Data residency (NFR-PRIV-06) is the real constraint on this choice. R2's jurisdictional restrictions
cover EU and a subset of regions; if the India region requirement becomes binding, a per-region
storage backend may be needed. Flagged in the revisit conditions.

## Consequences

**Positive** — Egress cost eliminated. S3-compatible, so tooling and portability are preserved.
Integrated CDN. Direct upload removes load and risk from the application.

**Negative** — Vendor concentration (storage, CDN, WAF, DNS all Cloudflare) — an outage affects
multiple layers simultaneously. Fewer regions than S3. Some lifecycle features must be implemented
in application code rather than configured.

**Neutral** — MinIO locally means development and production behave the same; the S3 API is the
common contract.

## Revisit when

- Data residency requirements (particularly India) cannot be met by R2's available regions.
- Cloudflare concentration risk materialises as a correlated outage.
- Storage volume grows to where per-GB pricing outweighs the egress saving (unlikely at any plausible
  scale for this product).
