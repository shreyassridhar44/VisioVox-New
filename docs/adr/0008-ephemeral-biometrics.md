# ADR-0008 — Biometric derivatives are ephemeral by default

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`16-privacy-compliance.md`](../16-privacy-compliance.md), NFR-PRIV-01/02, R-20, R-23

## Context

The pipeline produces **speaker embeddings (voiceprints)** and **face crops** for every person in
every uploaded video. Under GDPR Art. 9 these are special-category biometric data when used for
identification; Illinois BIPA carries a private right of action with statutory damages per
violation.

Critically, **the people in an uploaded video are not our users**. They are third-party data subjects
who never agreed to anything.

The archived roadmap (§10) proposed cross-video speaker identification — *"this is the same Speaker 2
as in last week's video"* — as a natural extension. Implemented naively, that means storing durable,
matchable voiceprints indefinitely: a covert voice-identification database built from recordings of
people who never consented.

## Options considered

### A — Store voiceprints indefinitely
**Pros:** Enables cross-video identification, speaker libraries, faster reprocessing.
**Cons:** Creates the highest-value breach target in the system. BIPA exposure. Likely pushes the
product toward "biometric identification" under the EU AI Act. Third parties cannot exercise rights
over data they don't know exists.

### B — Store with retention limits and encryption
**Pros:** Enables the features with reduced exposure.
**Cons:** Still a biometric database, still requires the full BIPA compliance apparatus (written
consent, published retention schedule, destruction within 3 years) — none of which we can obtain from
people we have never met.

### C — Ephemeral by default, opt-in persistence ✅
Embeddings and face crops live only for the duration of the job; deleted at S9. The DB column is
nullable and normally null. Persistence requires an explicit, revocable, per-user opt-in that is off
by default.

## Decision

**Option C.** Cross-video speaker identification is **not built in v1**
([`00-charter.md`](../00-charter.md) §4 non-goals).

Enforced structurally:
- `speakers.voiceprint` is nullable, written only when `users.persist_voiceprints` is true
- Artifacts are tagged `class='biometric'` and swept every 15 minutes
- A nightly assertion test verifies zero persisted voiceprints for non-opted-in users

## Rationale

This **eliminates a risk category rather than managing it.** There is no biometric database to
breach, to subpoena, or to misuse. Options A and B require ongoing compliance work, access controls,
retention scheduling and breach exposure — permanently — for a feature the product does not need.

The cost is genuinely low: embeddings are needed *during* processing, not after. The product works
identically without persistence. The only loss is cross-video identification, and that feature was
never load-bearing.

There is a second, less obvious reason. Under the EU AI Act, remote biometric *identification* is
heavily restricted. VisioVox performs biometric **processing for separation** — distinguishing
speakers within one recording without determining who they are, and retaining nothing that could
identify them later. That distinction is legally meaningful, and ephemerality is what preserves it.
Adding cross-video identification would change the classification and pull the product into a far
more regulated category. That is a strong reason to stay on this side of the line.

Finally, it is defensible as a design position rather than merely as compliance overhead:
"privacy-preserving by construction" is a stated project principle
([`00-charter.md`](../00-charter.md) §7.5), and this is what makes it true rather than aspirational.

## Consequences

**Positive** — Eliminates the largest privacy risk. Materially reduces GDPR Art. 9 and BIPA exposure.
Keeps the product out of the AI Act's biometric-identification category. Safe behaviour is the
default and requires no user action (Art. 25). Strengthens the DPIA.

**Negative** — No cross-video speaker identification. Reprocessing a video re-mines enrolments
(cheap — S4 is ~15 s). No speaker library feature.

**Neutral** — The opt-in mechanism exists in the schema, so the feature can be built later if there
is real demand and the compliance work is done first.

## Revisit when

- A concrete, high-value use case for cross-video identification appears **and** the full BIPA/GDPR
  consent apparatus is built first — not after.
- Regulation clarifies the boundary in a way that changes the analysis.

**Not a valid reason to revisit:** "it would be a nice feature."
