# 00 — Project Charter

## 1. One-line statement

**VisioVox** turns a video of people talking over each other into a video where you can listen to
any one of them, alone, with their captions — by selecting them.

## 2. The problem

When two or three people speak simultaneously on a recording, the result is often unusable. Human
listeners perform "cocktail party" separation effortlessly in a room, but lose that ability on a
mono recording, because the spatial cues the brain relies on are gone. Meetings, interviews,
lectures, podcasts, courtroom and field recordings all suffer from this.

Existing tooling does not solve it:

| Category | Examples | Why it doesn't solve this |
|---|---|---|
| Meeting transcription | Otter, Fireflies, Zoom AI | Produces speaker-labelled *text*. The **audio** stays mixed and unlistenable. |
| Stem separation | LALAL.AI, Moises, Demucs | Separates *music* stems (vocals/drums/bass), not two simultaneous **speakers**. |
| Podcast cleanup | Adobe Podcast Enhance, Auphonic | Denoises and de-reverbs a mix. Does not decompose it by speaker. |
| Research separation | SepFormer, Conv-TasNet, TF-GridNet | Strong on synthetic benchmarks. Unstable channel identity over long recordings, fixed speaker count, no product surface, audio-only. |
| Video editors | Premiere, DaVinci | Multi-track only if you *recorded* multi-track. Cannot un-mix a single mic. |

**The gap:** nobody delivers *per-speaker isolated audio, time-locked to the video, switchable
during playback, with per-speaker captions.*

## 3. Goals

> **G2 is the primary objective.** Everything else in this document serves it. When two goals
> conflict, isolation accuracy wins — specifically, *the unselected speakers being inaudible*.
> Every model, metric and architectural choice here is instrumental and may be replaced if
> something else serves G2 better; none of them is an end in itself. See §7.7.

| # | Goal | Measure of success |
|---|---|---|
| G1 | Isolate each speaker into a full-length, video-aligned audio track | ≥ 12 dB SI-SDRi on the held-out real-world eval set (AMI-Eval) |
| **G2** ⭐ | **Make the *unselected* speakers effectively inaudible** | **≥ 18 dB SIR; silence-region leakage ≤ −30 dB; cross-stream leakage word rate ≤ 3%** |
| G3 | Isolated audio must be pleasant to *listen* to, not merely transcribable | DNSMOS-P.835 OVRL ≥ 3.2; informal MOS ≥ 3.5/5 |
| G4 | Speaker switching is instant and stays in sync | Switch latency ≤ 120 ms; A/V drift ≤ 40 ms sustained over 10 min |
| G5 | Per-speaker captions, accurate and word-aligned | Target-speaker WER ≤ 15% on AMI-Eval; word timing median error ≤ 120 ms |
| G6 | Production-grade, secure, deployable web application | Passes the release gate in [`19-testing-strategy.md`](./19-testing-strategy.md) §9 |
| G7 | Demonstrable novelty over prior art | Five contribution axes in [`04-novelty.md`](./04-novelty.md), each with an ablation |
| G8 | A landing page and player people describe as good-looking | Lighthouse ≥ 95 across all four categories; ≤ 2.5 s LCP on mid-tier mobile |

## 4. Non-goals (v1)

Stating these protects the timeline. Each is a deliberate exclusion, not an oversight.

- ❌ **Real-time / live-stream separation.** Offline batch only. Real-time causal extraction is a
  materially harder problem (no lookahead) and would compromise G1–G3.
- ❌ **More than 4 speakers.** Design target is 2–3, hard cap 4 with an explicit quality warning
  shown to the user.
- ❌ **Non-English ASR** in v1. The extraction model is language-agnostic; only captioning is gated.
  Whisper's multilingual capability makes this a config change, deferred only for evaluation scope.
- ❌ **Music/singing separation.** Speech only.
- ❌ **Mobile native apps.** Responsive web only.
- ❌ **Collaborative editing / multi-tenant workspaces.** Single-owner projects in v1.
- ❌ **Cross-video speaker identity.** Deliberately excluded on privacy grounds — see
  [`16-privacy-compliance.md`](./16-privacy-compliance.md). Opt-in, post-v1 at the earliest.
- ❌ **Off-screen speaker handling as a first-class feature.** Handled by audio-only fallback, but
  not optimised for.

## 5. Users and jobs-to-be-done

| Persona | Job | Critical requirement |
|---|---|---|
| **Journalist** (primary) | Isolate one interviewee from a noisy multi-person field recording to quote accurately | G2 — leakage would mean misquoting someone |
| **Researcher / qualitative analyst** | Transcribe focus-group recordings per participant | G5 — attribution accuracy above all |
| **Podcast editor** | Recover a co-host's track from a session where mics bled into each other | G3 — must be broadcastable |
| **Student / accessibility user** | Follow one lecturer or one participant in a crowded recorded seminar | G4 — switching must feel immediate |
| **ML reviewer / examiner** | Judge whether the contribution is real | G7 — reproducible ablations |

### Primary user story

> As a journalist with a 12-minute interview where my subject and their colleague talked over each
> other for 90 seconds, I want to click my subject's face and hear only them for the whole
> recording, with captions, so that I can quote them without guessing.

## 6. Scope boundaries: what "done" means

A release is done when all of these hold simultaneously:

1. **Pipeline** — upload → speaker registry → per-speaker isolated tracks → per-speaker captions,
   all sample-aligned to the source video, fully automated, no manual steps.
2. **Player** — selecting a speaker swaps audio and captions with no perceptible seam, correct after
   seeking, scrubbing, rate change and tab-backgrounding.
3. **Model** — trained AV-TSE model beating both (a) the blind-separation baseline and (b) the
   audio-only TSE baseline on AMI-Eval, with a published ablation per contribution axis.
4. **Application** — authenticated, rate-limited, sandboxed, observable, deployed via CI/CD to a
   live environment, with a runbook and rollback path.
5. **Evaluation** — a written report with quantitative results, degradation curves by speaker count
   and overlap ratio, and an honest limitations section.
6. **Security & privacy** — threat model addressed, DPIA completed, deletion path verified working
   end-to-end.

## 7. Guiding principles

1. **Honesty over demo-polish.** A confidence score that admits "this 4 s segment is unreliable" is
   worth more than hiding the failure. Failure states are designed, not swept up.
2. **Faithfulness is the default.** Generative restoration can invent words. The default track is
   always the faithful one; "Natural" is an explicit opt-in with a visible label.
3. **Degrade, never break.** No face → audio-only. No enrolment → visual-only. No overlap → passthrough.
   Every stage has a defined fallback. The pipeline must never return nothing.
4. **The contract is the boundary.** Frontend, API and ML worker are built against a versioned
   OpenAPI + artifact-manifest contract so they can progress independently.
5. **Privacy by construction, not by policy.** Biometric derivatives are job-scoped and deleted with
   the job. The safe behaviour is the default behaviour and requires no configuration.
6. **Measure the thing the user cares about.** SI-SDR is a proxy. Leakage audibility and perceptual
   quality are the product. Track both; when they disagree, believe the perceptual metric.
7. **Models are instruments, not commitments.** TF-GridNet, pyannote, LoCoNet, Whisper — every one is
   chosen because it currently serves G2 best, and every one is replaceable. A change that improves
   measured isolation accuracy is always in scope, whatever it does to the architecture diagram. The
   only things not up for replacement are the *properties* the architecture guarantees: stable
   speaker identity over long recordings, graceful modality degradation, and faithfulness by default.

## 8. Constraints

| Constraint | Value | Impact |
|---|---|---|
| **ML hardware** | **NVIDIA RTX A5000 24 GB + 128 GB RAM** (college workstation, shared) | All training, evaluation and real inference. 24 GB removes the need for gradient checkpointing — see [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §5 |
| **App hardware** | Laptop, 16 GB RAM, no usable GPU | Application development runs entirely against the mock pipeline; no GPU required |
| Inference hardware (prod) | GPU worker, scale-to-zero | Cold-start budget handled in UX (queued state) |
| Team size | 1 | Favours managed services over self-hosted infra; ADR-0007 |
| Timeline | ~24 weeks part-time | ML and app tracks run in parallel — see [`21-implementation-plan.md`](./21-implementation-plan.md) |
| Budget | Hobby/student tier | Scale-to-zero GPU, R2 (zero egress) over S3, free-tier observability |
| Dataset licensing | LRS2/LRS3 need research agreements | Fallback corpus path in [`06-datasets.md`](./06-datasets.md) §6 |
| Training scope | **Exactly one model is trained: the extractor.** Everything else is pretrained or algorithmic | [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §3 |

## 9. Key risks (top 5 — full register in [`22-risk-register.md`](./22-risk-register.md))

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-01 | 3-speaker extraction quality below listenable threshold | High | Report degradation curve honestly; cap demo at 3; surface per-segment confidence |
| R-02 | Real-world domain gap collapses benchmark gains | High | In-domain fine-tune on AMI-Train; dereverb front-end; realistic simulation |
| R-03 | Generative restoration hallucinates words | High | Faithful track is default; gate on fidelity; label clearly; ASR cross-check |
| R-04 | Dataset licensing blocks AV training | Medium | Tiered fallback: VoxCeleb2 + AVSpeech + self-recorded |
| R-05 | GPU cost overruns | Medium | Scale-to-zero, per-user quotas, duration caps, cost alerting |

## 10. Glossary pointer

Terms like SI-SDR, TSE, PIT, DER, ASD, LUFS, CMAF are defined in
[`24-glossary.md`](./24-glossary.md).
