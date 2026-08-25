# MEMORY.md

Durable project context — the things that aren't obvious from the code and would otherwise have to be
rediscovered. Append as decisions are made and lessons are learned.

---

## Current status

**Phase:** Documentation complete. **Phase 0 not started.** No application or ML code exists yet.

| | |
|---|---|
| Repo | `github.com/shreyassridhar44/VisioVox-New` |
| Branch | `main` |
| Contains | 26 design docs, 15 ADRs, 4 templates, repo tooling. No source code. |
| ML machine | College workstation — RTX A5000 24 GB, 128 GB RAM |
| App machine | Laptop, 16 GB RAM, no usable GPU (or the workstation, if access is unrestricted) |

### Immediate next actions

1. **Workstation pre-flight** — [`docs/25-compute-and-hardware.md`](./docs/25-compute-and-hardware.md) §1b.
   Especially: is `~` network-mounted, is the GPU idle, is HuggingFace reachable.
2. **HuggingFace token** — accept gated terms for `pyannote/speaker-diarization-3.1` and
   `pyannote/segmentation-3.0`; put the token in `.env.local`.
3. **Start the LibriMix download** — slow, unattended, blocks Phase 3.
4. **Phase 0** — monorepo scaffold, tooling, Compose stack, and the pretrained-model smoke test
   ([`docs/21-implementation-plan.md`](./docs/21-implementation-plan.md)).

### Decisions taken outside the docs

- **Ethics clearance is deferred.** It gates only Phase 3 (VVX recording), not Phases 0–2. Check with
  the supervisor whether it is required at all before treating it as a six-week process. Written
  participant consent is still mandatory whenever recording happens.
- **Commit granularity:** one commit per completed unit of work; push at phase boundaries. Roughly
  5–15 commits per phase. Not per file, not per phase.
- **Attribution:** commits are authored solely by the repository owner. No AI co-author trailers or
  generation footers anywhere. Policy in [`CLAUDE.md`](./CLAUDE.md).

---

## Origin

The project began as `speaker-isolation-captioning-roadmap.md` (preserved at
`docs/archive/original-roadmap.md`), a plan for a lab prototype: fine-tune SepFormer on Libri2Mix,
build a simple local player, single-user, no auth.

Reviewing it against the actual goal — a production website with high-accuracy isolation and novelty
— found the feasibility analysis, risk framing and pipeline skeleton sound, and the model
architecture, player design, novelty claim and application scope wrong. That review is
`docs/02-approach-review.md` and is the reason every other document exists.

---

## Decisions that took real thought

### Why not blind source separation
The seductive path: SepFormer has pretrained checkpoints, a mature SpeechBrain recipe, and strong
benchmark numbers. It looks like the obvious choice.

It fails on a property that benchmarks never test: **PIT-trained models have no stable output
identity across inference windows.** A 6-minute video is ~72 windows, each with independent channel
ordering. Building one coherent track means solving a 72-step assignment problem where a single error
puts the wrong person's voice in the output.

The tell is that the standard fix — embed each chunk and match to a centroid — reconstructs target
speaker extraction badly, with an extra failure mode. Better to do TSE properly.

**Phase 1 measures this empirically** (permutation-error rate on full-length blind output). If BSS
turns out to hold identity well, ADR-0001 is wrong and gets revised in week 3, not month 5.

### Why the player design changed
`audio.currentTime = video.currentTime` looks like synchronisation. It isn't — it's a seek request,
resolved asynchronously to the nearest decodable point. And two media elements have two decoder
clocks, so they drift regardless.

The insight that unlocked it: **put every track on one `AudioContext` clock and never stop any of
them.** All tracks play simultaneously; only gain changes. Inter-track drift becomes structurally
impossible, and switching becomes an 80 ms gain ramp with no seek, no decode, no network.

Correct video drift with `playbackRate` nudges (≤0.4%, inaudible) rather than seeks. The original
plan's "periodically re-sync currentTime" would have produced a periodic audible click.

### Why "novelty" was reframed
Fine-tuning a published checkpoint on its own dataset is a reproduction. The original plan even
predicted the outcome correctly — "little improvement over pretrained on Libri2Mix."

The five contributions were chosen by asking a different question: *what blocks real deployment that
benchmark-chasing doesn't address?* Enrolment acquisition, modality reliability, the SI-SDR/audibility
gap, faithfulness vs quality, and inference-time confidence. Each maps to one contribution. That's
not coincidence — it's the selection criterion.

### Why biometrics are ephemeral
This started as a compliance decision and became an architectural one. Storing voiceprints would:
build a covert identification database from people who never consented; create the highest-value
breach target in the system; incur BIPA exposure with statutory damages; and likely reclassify the
product under the EU AI Act from "biometric processing" to "biometric identification," which is a far
more regulated category.

The feature it costs (cross-video speaker ID) was never load-bearing. Deleting the risk beat managing
it.

### Why two tracks in the plan
The original plan was sequential — model, then app. That makes the app hostage to model convergence,
and ML timelines have high variance.

The mock pipeline (fixture manifests, realistic timing, no GPU) decouples them. It cost about two
days to build and takes the entire application off the critical path. This is probably the single
highest-leverage decision in the schedule.

It only works because the **artifact manifest is frozen in week 3**. Contract drift would collapse
the whole structure, which is why the contract check is a blocking CI gate.

---

## Things that will be tempting and are wrong

| Temptation | Why it's wrong |
|---|---|
| "Just use SepFormer, it has checkpoints" | Permutation instability over long recordings — the thing the product must not do |
| "Store voiceprints, it enables cool features" | ADR-0008. Legal, ethical and regulatory reasons, all pointing the same way. |
| "Always apply generative restoration, it sounds better" | It can invent words. The primary user is quoting a source. |
| "Optimise the extractor, it's the AI part" | Video analysis is 28% of runtime; extraction is 17% |
| "Process everything uniformly, it's simpler" | Separating already-clean audio degrades it. 80–95% of the timeline is single-talker. |
| "SI-SDR is the metric" | It can't distinguish leakage from artifact. SIR is what matches the requirement. |
| "Skip the C0 smoke test, the code looks right" | A model that can't overfit 100 samples has a bug. Finding it after 40 hours is the expensive way. |
| "Transcribe the Natural track, it's cleaner" | Undermines the entire hallucination-safety design at its source |

---

## Constraints that shape everything

- **Single consumer GPU (≥12 GB)** — drives model size, gradient checkpointing, bf16, the VRAM ladder
- **Solo maintainer** — favours managed services and boring technology; ADR-0007 chose Celery over
  Temporal largely for this reason
- **Student budget** — R2's zero egress is the difference between viable and not
- **WSL2 on Windows** — all data and code inside the WSL2 filesystem, never `/mnt/c` (9p is several
  times slower and bottlenecks the dataloader before the GPU saturates)
- **~24 weeks part-time** — the parallel-track structure is what makes this recoverable when ML slips

---

## Open questions

| Question | Resolved by | When |
|---|---|---|
| Does blind separation actually drift over long recordings? | Phase 1 permutation-error measurement | Week 3 |
| Does visual conditioning help on *our* data? | C2 exit gate | Week 12 |
| Can generative restoration be gated safely enough to ship? | Hallucination-rate measurement | Phase 5 |
| Is 3-speaker quality listenable? | VVX-Eval + listening test | Phase 7 |
| Is the 40 MB WebAudio threshold right? | Production telemetry | Post-launch |
| Will the LRS3 agreement be granted? | Application outcome | Phase 3 |

---

## Lessons

_Append as they happen. Include what was expected, what happened, and what changed as a result._

- *(2026-08-25)* The original roadmap's weakest section was the one that sounded most reasonable:
  "don't over-invest in infra, auth, or deployment." Good advice for a different project. The lesson
  is that scope advice is only valid relative to a stated goal — and the goal had shifted between the
  two documents without the advice being revisited.

---

## Conventions worth remembering

- Requirement IDs (`FR-PLAY-03`) are stable and referenced from tests — don't renumber them
- Findings `F1`–`F13` refer to `docs/02-approach-review.md`
- Stages `S0`–`S9` refer to `docs/05-ml-architecture.md`
- Media timing is integer milliseconds at the API boundary, samples internally, never float seconds
- ⭐ in the docs marks the decisions most central to the project
