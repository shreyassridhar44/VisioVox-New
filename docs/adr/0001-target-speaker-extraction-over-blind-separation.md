# ADR-0001 — Target Speaker Extraction as the primary architecture

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`02-approach-review.md`](../02-approach-review.md) §F1, [`03-research-landscape.md`](../03-research-landscape.md) §1, NFR-ML-01…03

## Context

The archived roadmap proposed **blind source separation** (SepFormer / Conv-TasNet fine-tuned on
Libri2Mix) as the core model. The product requires:

1. Full-length, video-aligned isolated tracks for recordings up to 60 minutes
2. Support for a variable number of speakers (2–4), determined at inference
3. A UX in which the user names a speaker and receives that speaker's audio
4. Near-zero interferer leakage (NFR-ML-03: SIR ≥ 20 dB)

Available signal includes video, and the recordings are conversational — real overlap is 5–20% of
the timeline, not 100%.

## Options considered

### A — Blind source separation (the original proposal)
PIT-trained separator emitting K streams; assign identities afterwards.
**Pros:** Large literature, strong published benchmarks, pretrained checkpoints available, no cue
needed.
**Cons:** Output channel order is arbitrary and unstable across inference windows (the permutation
problem) — building a coherent 60-minute track requires a 700-step assignment problem whose errors
put the wrong voice in the track. Fixed output count. Trained on fully-overlapped anechoic clean
mixtures unlike real input. Ignores video entirely.

### B — Audio-only target speaker extraction
Condition on a speaker embedding; extract one speaker per invocation.
**Pros:** No permutation problem — identity is bound to the cue. Any speaker count. Simple stitching.
**Cons:** Requires an enrolment cue. Weak on same-gender / similar voices — a common and
perceptually severe real-world case.

### C — Audio-visual target speaker extraction ✅
As B, plus lip-motion conditioning from the speaker's face track.
**Pros:** All of B's benefits. Lip motion is unaffected by acoustic overlap, so it is orthogonal
information rather than redundant — largest gains exactly where audio-only is weakest. Aligns with
the "click a face" UX. Handles variable speaker count naturally.
**Cons:** Needs face detection, tracking and ASD upstream. Degrades when the face is not visible.
Higher compute. More training complexity.

## Decision

**Audio-visual TSE (option C)** is the primary architecture, with graceful fallback to audio-only
(option B) when video is unusable. Blind separation (option A) is retained as an **evaluation
baseline**, not as a production path.

## Rationale

The decisive factor is not benchmark quality — it is **identity stability over long recordings**.
The product's core promise is "this track contains only this person." BSS cannot structurally
guarantee that over a 60-minute timeline; TSE can, because the conditioning cue *is* the identity.

Option A's weakness is also self-reinforcing: the standard mitigation for permutation drift is to
embed each chunk's output and match it to a speaker centroid — which means implementing speaker
embeddings and a matching step anyway, arriving at a worse version of TSE with an extra failure mode.

Option C over B: video is already being processed for the speaker-selection UI. Using the strongest
available disambiguating signal only to draw a button label, and not to condition the model, is
unjustifiable once the vision stack exists.

The cost — needing an enrolment cue — is turned into a contribution rather than absorbed as a
limitation (self-enrolment, ADR-0003).

## Consequences

**Positive** — Stitching becomes trivial. Variable speaker count for free. UX and model share one
mechanism. Same-gender case addressed. Enables Novelties 1–3.

**Negative** — Depends on a vision pipeline that can fail. Requires a modality-fallback design
(handled by ADR-0003 and Novelty 2). More components to build and maintain. Cannot quote WSJ0-2mix
numbers competitively — we do not optimise for that benchmark.

**Neutral** — Different literature and different baselines; the results section compares against BSS
rather than joining its leaderboard.

## Revisit when

- Phase 1 measurement shows blind separation maintains identity over 10-minute recordings with a low
  permutation-error rate. **This is an explicit Phase 1 exit criterion** — the experiment is designed
  to be able to falsify this ADR.
- A BSS architecture emerges with built-in identity tracking across arbitrary-length input.
- Visual conditioning shows no measurable gain after alignment is verified (→ fall back to option B;
  the rest of the design is unaffected).
