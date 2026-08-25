# ADR-0013 — Dual-track dataset licensing

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`06-datasets.md`](../06-datasets.md), R-24

## Context

Audio-visual training needs face-plus-speech data. The best available corpora carry restrictive
licences:

| Dataset | Licence | Commercial use |
|---|---|---|
| LRS2 / LRS3 | Signed BBC/Oxford agreement | ❌ Research only |
| VoxCeleb2 | CC BY 4.0 annotations; YouTube-sourced video | ⚠️ Research; underlying content varies |
| AVSpeech | YouTube-derived | ⚠️ Research |
| EasyCom | CC BY-NC 4.0 | ❌ Non-commercial |
| AMI | CC BY 4.0 | ✅ |
| LibriMix / LibriSpeech | CC BY 4.0 | ✅ |
| VVX (ours) | Our consent forms | ✅ within consent scope |

A model trained on research-only data cannot ship commercially. Discovering this after training is
an expensive mistake — and a very common one.

## Options considered

### A — Use everything, worry later
**Pros:** Best possible quality; simplest planning.
**Cons:** The resulting checkpoint is unshippable. Retraining on permissive data late in the project
costs weeks and lands after the results are written.

### B — Permissive data only
**Pros:** Everything is shippable.
**Cons:** Loses LRS3's quality, which is the cleanest AV speech data available, and weakens the
research results.

### C — Two checkpoint tracks ✅
- **Research checkpoint** — trained on everything including LRS3/EasyCom. Used for the paper and
  ablations. Never deployed.
- **Production checkpoint** — trained only on LibriMix, VoxCeleb2, AVSpeech and VVX. Deployed.

Both evaluated on VVX-Eval, so the gap between them is measured and reported.

## Decision

**Option C.** Every checkpoint carries a model card recording exactly which datasets it saw and
whether it is commercially usable. Deployment tooling refuses to deploy a checkpoint whose card is
not marked commercially clear.

## Rationale

The two tracks serve genuinely different purposes and have different constraints. The research
checkpoint answers *"how good can this method be?"* — for which using the best available data is
correct. The production checkpoint answers *"what can we ship?"* — for which licensing is a hard
constraint.

Running both is cheap because the difference is a dataset list in a config, not a code change. The
production checkpoint's C4 stage (VVX in-domain fine-tuning) is where most of the real-world quality
comes from anyway, and VVX is fully ours.

**Reporting the measured gap between the two checkpoints is itself a useful result.** It quantifies
what restrictive licensing costs a deployed system — a question practitioners face constantly and
that papers rarely answer, because papers only ever report the research checkpoint.

Making the model card a deployment gate is what keeps this from decaying into good intentions. A
licence audit that depends on someone remembering is not a control.

## Consequences

**Positive** — No unshippable-model surprise. Licence status is explicit and machine-checked. Yields
an additional reportable result.

**Negative** — Two training runs for the final model (~25 extra GPU-hours). Two sets of results to
keep straight. Model card discipline must be maintained from the first checkpoint, not retrofitted.

**Neutral** — Ablations run on the research track only, since they are for the paper.

## Revisit when

- The LRS3 agreement is denied → the research track collapses into the production track. No
  architectural change; the plan already assumes this is possible.
- The project is confirmed as non-commercial and permanently research-only → a single track suffices.
- A permissively-licensed AV corpus of comparable quality becomes available.
