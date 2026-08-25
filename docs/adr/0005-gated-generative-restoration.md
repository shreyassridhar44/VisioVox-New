# ADR-0005 — Gated generative restoration with dual-track delivery

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`04-novelty.md`](../04-novelty.md) §5, [`05-ml-architecture.md`](../05-ml-architecture.md) §9, G3, R-05

## Context

Discriminative extraction output is **faithful but artifact-y** — the characteristic metallic and
watery texture of masked audio. Goal G3 requires isolated audio that is pleasant to listen to, not
merely transcribable.

Generative restoration (resynthesis via a speaker-conditioned vocoder) produces markedly cleaner
audio. It can also **hallucinate**: when the input is severely degraded, it may synthesise plausible
words that were never spoken.

The primary persona is a journalist quoting a source. A hallucinated word is a catastrophic failure —
strictly worse than an audible artifact, because the artifact is obviously a defect and the
hallucination is not.

## Options considered

### A — Discriminative only
**Pros:** Zero hallucination risk by construction. Simple.
**Cons:** Fails G3 on hard segments. Sounds processed.

### B — Always apply generative restoration
**Pros:** Best perceptual scores.
**Cons:** Hallucination risk on exactly the segments where it is applied most aggressively. WER
regression. Unacceptable for the primary use case.

### C — Apply restoration only where quality is poor
**Pros:** Better than A, safer than B.
**Cons:** Backwards on the risk axis — the worst inputs are both where restoration helps most *and*
where hallucination risk is highest. This maximises exposure.

### D — Gated restoration + dual delivery ✅
A fidelity estimator φ decides per segment:

| φ | Action |
|---|---|
| ≥ 0.75 | Passthrough — already clean, restoration can only hurt |
| 0.35 – 0.75 | Restore and blend, γ = f(φ) capped at 0.7 |
| < 0.35 | **Refuse to restore.** Mark the segment low-confidence instead. |

Both variants are packaged and shipped: **Faithful** (default) and **Natural** (labelled).

## Decision

**Option D.** Faithful is always the default and is always what gets transcribed.

## Rationale

The third branch is the whole decision, and it is deliberately the opposite of what a
quality-maximising system would do: **where restoration would improve the numbers most, we refuse to
apply it.** Below φ = 0.35 the model has too little real signal to work from, so what it produces is
substantially invention. Declining and disclosing is the correct behaviour; a low-confidence marker
is more useful to a journalist than a confident fabrication.

Dual delivery resolves the tension without guessing at the user's purpose. A journalist verifying a
quote wants Faithful. Someone listening to a lecture wants Natural. The system cannot know which, so
it ships both and labels them — in the UI, in the filename, and in file metadata.

Transcribing the Faithful track (never the Natural one) keeps captions grounded in what was actually
recovered. Without this, the safety argument would be undermined at its source.

## Consequences

**Positive** — Perceptual gain without WER regression. Honest handling of a genuine trade-off.
Becomes Novelty 4, with hallucination rate as a measurable safety claim. Satisfies EU AI Act
transparency obligations for generated content.

**Negative** — Two tracks to generate, store and serve (~2× audio storage). An extra pipeline stage.
The fidelity estimator must itself be trained and calibrated. UI complexity: another user-facing
choice to explain.

**Neutral** — S6 is feature-flagged. If it does not work, disable it and ship Faithful-only; nothing
else depends on it.

## Revisit when

- Measured hallucination rate cannot be brought under 0.5% → disable S6 and report the negative
  result.
- Users overwhelmingly pick one mode → make it the default, keep the other available.
- A restoration model appears with provable faithfulness guarantees → the gate may be relaxed.
