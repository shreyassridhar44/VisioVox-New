# ADR-0015 — AMI replaces VVX as the in-domain evaluation set

- **Status:** Accepted
- **Date:** 2026-08-26
- **Related:** [`06-datasets.md`](../06-datasets.md), [`08-evaluation-protocol.md`](../08-evaluation-protocol.md), [ADR-0013](./0013-dataset-licensing.md), R-26
- **Supersedes:** the T4 tier as originally specified in `06-datasets.md` §1

## Context

The plan specified a self-recorded corpus, **VVX**, as Tier 4: 30 sessions with per-speaker
reference microphones, recorded by us. It was to be the basis of every headline number
(`06-datasets.md` §17, `08-evaluation-protocol.md` §3) because it matches the product's actual
input distribution, and `25-compute-and-hardware.md` §6 marked it the one dataset never to cut.

**VVX will not be recorded.** No participants have been collected and none are planned for this
project. That is a fixed constraint, not a scheduling slip, so the evaluation basis has to change
rather than wait.

Something must fill the role, because the role is real: without in-domain data there is no honest
"does this work on real conversation" number, only benchmark scores on simulated mixtures — which
is exactly the failure mode `02-approach-review.md` was written to avoid.

## Options considered

### A — Report only on simulated mixtures (Libri2Mix and similar)
**Pros:** Already available; literature-comparable.
**Cons:** Libri2Mix is fully overlapped, anechoic, level-balanced and two-speaker. Real conversation
is sparsely overlapped, reverberant and unbalanced — we measured 3–15% overlap on real meetings
against Libri2Mix's 100%. Reporting only on it would reproduce the precise overclaim the project
exists to avoid, and `03-research-landscape.md` §7 already commits us not to.

### B — Apply for EasyCom, MISP or LRS3
**Pros:** EasyCom and MISP are genuinely in-domain multi-party AV with close-talk references.
**Cons:** All are agreement-gated with unpredictable turnaround, and the Oxford LRS3 page is
currently 404. EasyCom is CC BY-NC, which ADR-0013 restricts to ablations only. Trading a certain
blocker for an uncertain one.

### C — ⭐ Promote AMI from Tier 3 to the primary in-domain benchmark
**Pros:**
- Already acquired, CC BY 4.0, commercially usable under ADR-0013.
- **Per-participant headset microphones give ground-truth per-speaker references** — the property
  that made VVX valuable, and the reason SI-SDR and SIR can be computed at all on real conversation.
- **Per-participant `Closeup` cameras give one clean face per speaker.** Measured: 100% frame
  coverage and 60–85 px median face, against 0 usable tracks on the `Corner` wide view.
- 154 meetings across four rooms (ES 60, IS 38, TS 40, EN 16), so the speaker- **and** room-disjoint
  split discipline in `06-datasets.md` §193 survives intact.
- Real multi-party conversation with realistic sparse overlap.

**Cons:** meeting-room domain only; 352×288 video; 2004–2006 English recordings; no control over
recording conditions. Detailed below.

### D — Record a token 2–3 sessions instead of 30
**Pros:** Some genuinely in-domain data.
**Cons:** Too small for a speaker-disjoint eval split, and a headline number computed on three
sessions invites more doubt than it resolves. Worse than a clean substitution honestly labelled.

**This fallback was designed in advance.** `21-implementation-plan.md` §66 states that if VVX
never happens, AMI's close-talking headset channels provide per-speaker references for real
overlapping speech, and `22-risk-register.md` carries it as **R-25**. This ADR executes that
contingency and settles the details it left open — which cameras, and how the splits work.

## Decision

**AMI becomes the primary in-domain evaluation set, as `AMI-Eval`.** It is built from per-participant
`Closeup` video paired with that participant's `Headset` audio, with the mixture formed by summing
the headsets — the same construction already validated in Phase 1.

Splits are room-disjoint by series: train on **ES + IS**, evaluate on **TS**, hold **EN** back as a
second unseen room. Speaker-disjointness is enforced within that, since AMI publishes participant
ids.

VoxCeleb2-Mix remains the Tier 2 corpus for AV training scale and speaker diversity once credentials
arrive. AMI is the evaluation basis, not the only training data.

## Consequences

### What this costs, stated plainly

Four things are genuinely lost, and none should be papered over in the report:

1. **Domain.** AMI is a meeting room with seated participants and table-mounted cameras. The product
   accepts arbitrary uploaded video. Headline numbers now describe *meeting recordings*, not the
   general case, and must be labelled that way.
2. **Video quality.** 352×288 at 25 fps is far below a modern phone. The visual pathway is being
   evaluated at a resolution no current user would produce — plausibly pessimistic for face
   detection, but simply unrepresentative rather than conservative.
3. **Controlled hard cases.** VVX was to be recorded deliberately containing same-gender pairs,
   specific overlap densities and specific room acoustics. AMI supplies whatever it happens to
   contain; those conditions can be selected for but not created.
4. **Consent scope.** ADR-0013 gave VVX the broadest usage rights of any corpus. AMI is CC BY 4.0,
   which is permissive but requires attribution and carries its own terms.

### What is unaffected

- The permutation-error result and ADR-0001 stand: they were measured on AMI already.
- ADR-0010 single-talker routing was measured on AMI already.
- The extractor still needs training; nothing here changes ADR-0001 or the SEAVE design.
- R-26 (back up VVX off-machine, irreplaceable) no longer applies — every dataset is now
  re-downloadable, which removes the project's single unrecoverable-data risk.

### Required documentation changes

`06-datasets.md` §1 and §17, and `08-evaluation-protocol.md` §3, both state that every headline
number is reported on VVX-Eval. Both must now read AMI-Eval, with the domain caveat attached at the
point of claim rather than in a footnote.

### Revisit if

Participants become available. Even 10 sessions would allow a genuine in-domain check against the
product's real input distribution, reported alongside AMI rather than replacing it. The pipeline for
building an eval set from (per-speaker video, per-speaker reference audio) already exists and would
accept VVX unchanged.
