# ADR-0003 — Self-enrolment from diarization

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** ADR-0001, [`04-novelty.md`](../04-novelty.md) §2, FR-PIPE-12

## Context

TSE (ADR-0001) requires a cue identifying the target speaker. The literature almost universally
assumes a **clean enrolment recording supplied externally**.

Our users have none. A journalist uploading an interview cannot produce a clean isolated recording
of the person they just interviewed — that is the thing they came here to get. This is the reason
TSE has strong research results and essentially no consumer products.

## Options considered

### A — Ask the user for an enrolment clip
**Pros:** Highest cue quality; standard approach.
**Cons:** Destroys the product. Users cannot supply it, and requiring it turns a one-step upload into
a manual per-speaker task. Non-starter.

### B — Let the user scrub to a region where each speaker talks alone
**Pros:** No external file needed; user has context the system lacks.
**Cons:** Manual work per speaker per video. Requires listening to the whole recording first, which
is precisely what the product exists to avoid. Bad UX; users will choose poorly.

### C — Mine enrolment automatically from diarization ✅
Use overlap-aware diarization to find single-talker regions, score their purity, aggregate embeddings
weighted by purity.
**Pros:** Zero user effort. No enrolment UI at all. Works on arbitrary uploads. Diarization is
already in the pipeline, so the marginal cost is small. **Becomes a research contribution.**
**Cons:** Cue quality depends on the recording containing usable single-talker regions. Fails on
recordings that are overlapped throughout. Adds a dependency on diarization accuracy.

### D — Blind separation to bootstrap enrolment, then TSE
**Pros:** Works even with no clean regions.
**Cons:** Reintroduces every problem ADR-0001 rejected, as a dependency of the solution to them.

## Decision

**Option C**, with purity-weighted aggregation over multiple regions and explicit fallbacks:

| Condition | Behaviour |
|---|---|
| ≥ 1.5 s of pure speech found | Audio cue from purity-weighted embedding |
| < 1.5 s pure speech, face available | Visual-only conditioning |
| < 1.5 s pure speech, no face | Diarization-masked passthrough, segment marked low-confidence |
| No usable evidence at all | Speaker dropped from the registry with a user-visible note |

## Rationale

Real conversations are 80–95% single-talker. The information needed for enrolment is almost always
present in the recording; it simply has not been extracted before. Options A and B ask the user to
supply information the system can find on its own.

**Purity weighting rather than longest-region selection** is the non-obvious part. The intuitive
choice — take the longest clean stretch — is worse, because one long region captures a single
prosodic context (one sentence, one register) and generalises poorly. Weighted aggregation over
several diverse regions lands closer to the speaker's true embedding centroid. This is measurable
and is ablated (A2).

The confidence scores produced here are not diagnostics — they are the routing signal for Novelty 2's
reliability gates. Self-enrolment and modality-adaptive conditioning are one mechanism, not two.

## Consequences

**Positive** — No enrolment UI. Works on arbitrary uploads. Provides the confidence signals that
drive modality adaptation. Constitutes Novelty 1.

**Negative** — Cue quality is bounded by what the recording contains. Heavily overlapped recordings
degrade to visual-only. Adds a dependency on diarization quality — a diarization error propagates
into a contaminated cue.

**Neutral** — Enrolment mining must be deterministic (fixed seeds, stable sort) so job re-runs
reproduce and caching stays sound.

## Revisit when

- More than 15% of speakers in production lack a usable enrolment → the purity thresholds are wrong,
  or the fallback path needs strengthening.
- A robust enrolment-free TSE formulation appears that needs no cue at all.
