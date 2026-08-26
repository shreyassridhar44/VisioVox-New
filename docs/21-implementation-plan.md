# 21 — Implementation Plan

**~24 weeks part-time.** Two tracks run in parallel — ML and Application — joined by a versioned
contract. Each phase has explicit **exit criteria**; do not advance until they are met.

---

## 1. Why two parallel tracks

The archived roadmap was sequential: build the model, then build the app. That has a fatal property
— **the app is blocked on model convergence**, so any ML slip (and there will be several) consumes
app time directly. With 7 weeks of ML and 3 of app in a 12-week plan, one bad training week eats the
entire frontend.

The fix is the **mock pipeline**: a service that returns a pre-computed, schema-valid manifest from
fixture data. Built in week 2, it lets the entire application be developed, tested and demoed
without a GPU or a trained model. The tracks converge at Phase 8.

```
 Week   1    4    8    12   16   20   24
        │    │    │    │    │    │    │
 ML     ├─P1─┼─P3─┼───P4────┼─P5─┼─P7─┤
        │    │    │         │    │    │
 APP    ├─P2─┼──────P6──────┼─P8─┼─P9─┤
        │    │              │    │
        └ contract frozen   └ integrate
```

---

## Week 0 — Long-lead items

Only two of these are worth doing before Phase 0. The rest can start later without cost.

| Item | Lead time | Blocks | When |
|---|---|---|---|
| **HuggingFace token + accept pyannote licences** | 5 minutes | Phase 0 smoke test | ⭐ **Now** |
| **Start LibriMix download** | 2–5 days unattended | Phase 3 | ⭐ **Now** — slow, needs no supervision |
| Workstation pre-flight | 1 hour | Everything | Now — [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §1b |
| ~~Recruit VVX participants~~ | — | cancelled (ADR-0015) | — |
| Ethics clearance (if required) | 0–6 weeks | Phase 3 only | See below |
| LRS3 research agreement | 2–8 weeks, may be denied | Nothing — optional | Anytime, or never |

### On ethics clearance — deferrable, and possibly not required

**It blocks nothing until Phase 3 (week 4).** Phases 0, 1 and 2 use only public datasets that carry
their own licences and ethics provenance. The baseline pipeline, the application, and Tier 1
training on LibriMix plus simulation all proceed without it.

Two things are being conflated and shouldn't be:

| | What it is | Required? | Cost |
|---|---|---|---|
| **Institutional ethics / IRB approval** | A formal review process | **Often not, for coursework** — many institutions exempt student projects recording consenting adults on non-sensitive topics | 0–6 weeks |
| **Written participant consent** | A form each person signs | **Always. No exceptions.** | 10 minutes |

**Do this instead of a six-week detour:** spend ten minutes asking your project supervisor whether
your institution requires review for a capstone that records consenting classmates on non-sensitive
topics. The answer is frequently "no" or "file an exemption form." If it does require full review,
submit it then and keep building — you have four weeks of runway.

**Use the consent form regardless.** [`templates/vvx-consent-form.md`](./templates/vvx-consent-form.md)
needs no approval to use, costs nothing, and is what actually protects the participants and makes
the data usable in a report. This is the part that is genuinely non-negotiable.

**VVX did not happen, and the fallback has been executed.** AMI's close-talking headset channels
provide per-speaker references for real overlapping conversational speech — weaker than
purpose-recorded data, but real and publishable. This was R-25 in
[`22-risk-register.md`](./22-risk-register.md); it is now
[ADR-0015](./adr/0015-ami-replaces-vvx.md).

---

## Phase 0 — Foundations (Week 1)

**Goal:** everything runs; nothing is built yet.

Split across both machines — see [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §8.

**On the laptop (app environment):**

| Task | Detail |
|---|---|
| Repo | Monorepo per [`09-system-design.md`](./09-system-design.md) §11; pnpm + uv workspaces |
| Tooling | ruff, mypy strict, eslint, prettier, pre-commit, gitleaks |
| CI skeleton | Lint, typecheck, empty test run |
| Docker Compose | Postgres, Redis, MinIO, mailhog |
| Docs | This set, committed |

**On the college workstation (ML environment):**

| Task | Detail |
|---|---|
| **CUDA verify** | ⭐ `nvidia-smi` → A5000 visible; `torch.cuda.is_available()` → `True`. Nothing downstream works without this. |
| Python env | PyTorch (CUDA build matching driver), SpeechBrain, pyannote, faster-whisper, ffmpeg |
| Model access | HuggingFace token for gated pyannote 3.x models |
| Storage | ≥ 510 GB on **local NVMe**, not a network home directory |
| **Start LibriMix download** | ⭐ Begin immediately — it is slow, unattended, and blocks Phase 3 |
| **Smoke test** | ⭐ Run every pretrained model once on one clip, end to end |
| tmux/screen | So long runs survive disconnection |

**Exit:** `make dev` brings up the full stack on the laptop with the mock pipeline. On the
workstation, every pretrained model produces output on a sample clip and PyTorch sees the GPU.

> The smoke test is the archived roadmap's Phase 0 advice, and it was right: proving the plumbing
> before building on it saves days later. Do it before writing any pipeline code.

---

## Phase 1 — Baseline pipeline, pretrained only (Weeks 2–3) · ML

**Goal:** an honest "before" baseline, and a working artifact contract.

| Task |
|---|
| S0 ingest with ffmpeg |
| S2A: pyannote diarization + overlap |
| S2B: SCRFD + ByteTrack + Light-ASD (fast model first) |
| S3: fusion → speaker registry |
| S5: **pretrained SepFormer**, blind separation — the baseline we intend to beat |
| S7: Whisper + WhisperX |
| S9: packaging, HLS, VTT |
| **Artifact manifest v1.0** ⭐ — frozen, JSON Schema committed |
| Run on 3 test videos; document every failure precisely |

**Exit:**
- End-to-end pipeline produces a schema-valid manifest
- **Measured permutation-error rate** on full-length blind-separation output ⭐
- Written baseline report: where it breaks, with audio examples

> The permutation-error measurement is the empirical test of
> [`02-approach-review.md`](./02-approach-review.md) §F1.1. If blind separation maintains identity
> well over 10 minutes, the architecture decision was wrong and must be revisited **now**, in week 3
> — not in month 5.

---

## Phase 2 — Application skeleton + mock pipeline (Weeks 2–4) · APP

Runs concurrently with Phase 1.

| Task |
|---|
| FastAPI: auth (register/verify/login/refresh + rotation), projects, jobs |
| Postgres schema + Alembic migrations |
| Presigned multipart upload to MinIO |
| Celery + Redis; job state machine; SSE progress |
| **Mock pipeline worker** ⭐ — emits fixture manifests with realistic stage timing |
| Next.js: auth pages, project list, upload, processing view |
| **Generated TS client from OpenAPI** |
| CI: unit + integration tests |

**Exit:** upload → mocked processing with live progress → a `ready` project with a valid manifest.
No GPU involved.

---

## Phase 3 — Data (Weeks 4–6) · ML

| Task |
|---|
| Generate Libri2Mix + Libri3Mix (`min`, 16 kHz, `mix_both`) — one config only |
| Acquire VoxCeleb2; pre-extract mouth ROIs to a packed format |
| Acquire AMI (headset + far-field + video) |
| Build the realistic mixture simulator ([`06-datasets.md`](./06-datasets.md) §5) |
| ~~Record VVX corpus~~ — **cancelled**; build **AMI-Eval** instead (`scripts/build_ami_eval.py`), per [ADR-0015](./adr/0015-ami-replaces-vvx.md) |
| Define splits with verified speaker- and room-disjointness |
| Build the evaluation harness ([`08-evaluation-protocol.md`](./08-evaluation-protocol.md) §2) |

**Exit:** dataloaders produce correct batches at ≥ 85% GPU utilisation. AMI-Eval built, aligned
and split room- and speaker-disjoint. Harness reproduces published SepFormer numbers on Libri2Mix within ~1 dB — which validates
the harness, not the model.

> **VVX is cancelled** ([ADR-0015](./adr/0015-ami-replaces-vvx.md)). What was the longest-lead
> item is now a download, which removes the largest schedule risk in the project — at the cost
> of every headline number becoming a meeting-domain result rather than a general one.

---

## Phase 4 — Core model (Weeks 6–14) · ML · **critical path**

| Sub-phase | Weeks | Content |
|---|---|---|
| 4a | 6–7 | Implement SEAVE: TF-GridNet backbone, FiLM conditioning, losses. **C0 smoke — overfit 100 samples** |
| 4b | 7–10 | C1: audio-only TSE on Libri2Mix → ≥ 13 dB SI-SDRi |
| 4c | 10–12 | C2: add visual pathway + reliability gates + modality dropout → ≥ +1.5 dB on same-gender |
| 4d | 12–13 | C3: realistic simulation |
| 4e | 13–14 | C4: AMI-Train fine-tune |

Per [`07-training-playbook.md`](./07-training-playbook.md).

**Exit:** meets NFR-ML-01/02/03/04 floors on AMI-Val. Checkpoint versioned with a model card.

**Checkpoints:** if 4b misses 13 dB by week 10, stop and diagnose rather than pushing forward —
every later stage builds on it. Contingency in §5.

---

## Phase 5 — Novelty stages (Weeks 14–17) · ML

| Task |
|---|
| S4 self-enrolment with purity scoring (Novelty 1) |
| S1 dereverb + denoise front-end |
| S5 single-talker passthrough router (F11) |
| S6 gated generative restoration (Novelty 4) — **feature-flagged** |
| S8 cross-stream leakage audit (Novelty 5) |
| Confidence head + calibration |
| Loudness normalisation + length assertion (F13, I3) |

**Exit:** full pipeline replaces the Phase 1 baseline. Each novelty stage independently toggleable
for ablation. RTF ≤ 2.0×.

> S6 is flagged because it is the one stage the product can ship without. If the timeline compresses,
> disable it and ship Faithful-only — nothing else depends on it
> ([`05-ml-architecture.md`](./05-ml-architecture.md) §9).

---

## Phase 6 — Application build-out (Weeks 5–16) · APP

Runs concurrently with Phases 3–5.

| Weeks | Content |
|---|---|
| 5–7 | **`WebAudioSyncEngine`** ⭐ + sync test harness. Built early — highest technical risk in the app |
| 7–9 | Player UI: speaker rail, captions, multi-lane timeline, transcript |
| 9–11 | `HlsSyncEngine`; export pipeline; share links |
| 11–13 | **Landing page**: 3D hero, live demo, scroll narrative |
| 13–14 | Settings, privacy controls, deletion with receipts |
| 14–15 | Security hardening: CSP, sandbox, rate limits, ownership suite |
| 15–16 | Accessibility pass; error and empty states; responsive |

**Exit:** full application working against the mock pipeline. Sync tests pass. Lighthouse ≥ 95.
Zero axe violations. Security checklist complete.

> Build the sync engine in week 5, not week 15. It is the part most likely to reveal a design flaw,
> and [`12-media-pipeline-and-sync.md`](./12-media-pipeline-and-sync.md) is a design, not a proven
> implementation. Find out early.

---

## Phase 7 — Evaluation & ablations (Weeks 16–19) · ML

| Task |
|---|
| Train baselines B1–B6 ([`07-training-playbook.md`](./07-training-playbook.md) §7) |
| Ablations A1–A7, 3 seeds each |
| Full AMI-Eval with all slices |
| **Listening test** — ≥ 15 participants |
| Degradation curves, calibration diagrams |
| Draft the report including the limitations section |

**Exit:** every novelty claim has a result — supporting or falsifying it. Report drafted.

> Ablations parallelise perfectly. If GPU-bound, rent 8 cloud GPUs for two days rather than serialise
> three weeks.

---

## Phase 8 — Integration (Weeks 19–21)

Where the tracks meet.

| Task |
|---|
| Replace the mock worker with the real pipeline |
| Verify manifest compatibility — the contract has been frozen since week 3 |
| GPU worker containerisation, model cache volume, checksum verification |
| End-to-end on real uploads |
| Performance tuning ([`20-performance-cost.md`](./20-performance-cost.md) §2 Tier 1) |
| Full observability wiring |
| Load and concurrency testing |

**Exit:** real uploads process end to end within NFR-PERF-02. Traces span browser → API → worker.

> Integration is short **because the contract was frozen in week 3**. If the manifest schema has
> drifted, this phase expands and the whole plan slips — which is exactly why the contract check is
> a blocking CI gate ([`11-api-spec.md`](./11-api-spec.md) §12).

---

## Phase 9 — Production hardening & launch (Weeks 21–24)

| Weeks | Content |
|---|---|
| 21–22 | Terraform, K8s, CI/CD, staging deploy |
| 22 | Security: pen test / ASVS self-assessment, malicious media corpus, sandbox escape tests |
| 22–23 | Privacy: DPIA, legal pages, deletion verification, retention jobs |
| 23 | Dashboards, alerts, runbooks; DR rehearsal |
| 23–24 | Device matrix testing; final listening check; demo reel |
| 24 | Release gate ([`19-testing-strategy.md`](./19-testing-strategy.md) §9); production deploy |

**Exit:** live, secure, observable, documented, with a rehearsed rollback.

---

## 2. Milestones

| # | Week | Milestone | Demonstrates |
|---|---|---|---|
| M1 | 3 | Baseline pipeline + frozen contract | The problem is real and measured |
| M2 | 4 | App skeleton on mock pipeline | Tracks decoupled |
| M3 | 6 | AMI-Eval built | In-domain evaluation set ready |
| M4 | 7 | **Sync engine proven** | Highest app risk retired |
| M5 | 10 | Audio-only TSE ≥ 13 dB | Model works |
| M6 | 12 | AV conditioning beats audio-only | Core novelty validated |
| M7 | 14 | Meets NFR-ML floors on AMI-Eval | Product-grade quality |
| M8 | 16 | Full app on mock | Application complete |
| M9 | 19 | Ablations complete | Research complete |
| M10 | 21 | Real end-to-end | Integrated |
| M11 | 24 | Production launch | Done |

---

## 3. Task board seed

Issues to open in week 1, labelled `track:ml` / `track:app` / `phase:N`:

**Phase 0** — repo scaffold · CI skeleton · WSL2/CUDA verify · compose stack · model smoke test ·
docs commit

**Phase 1** — ingest · diarization · face pipeline · ASD · fusion · SepFormer baseline · Whisper ·
packaging · **manifest schema v1.0** · baseline report

**Phase 2** — auth · schema+migrations · presigned upload · Celery+SSE · **mock worker** ·
Next.js shell · generated client · test harness

**Phase 3** — LibriMix gen · VoxCeleb2 ROIs · **AMI-Eval build** · simulator ·
splits · eval harness

**Phase 4** — TF-GridNet · FiLM · losses · **C0 smoke** · C1 · visual frontend · reliability gates ·
modality dropout · C2 · C3 · C4

**Phase 5** — self-enrolment · dereverb · passthrough router · restoration+gate · leakage audit ·
confidence calibration · loudness+assertions

**Phase 6** — **WebAudioSyncEngine** · sync tests · speaker rail · captions · timeline · transcript ·
HlsSyncEngine · exports · shares · **3D hero** · **live demo** · settings · deletion · CSP ·
sandbox · a11y · error states

**Phase 7** — baselines · A1–A7 · full eval · listening test · figures · report draft

**Phase 8** — real worker · GPU container · model cache · perf tuning · telemetry · load test

**Phase 9** — terraform · k8s · CI/CD · pen test · DPIA · legal · dashboards · DR drill · device
matrix · demo reel · launch

---

## 4. Dependencies

```
P0 ──┬─► P1 ──┬─► P3 ──► P4 ──► P5 ──► P7 ──┐
     │        │                             ├─► P8 ──► P9
     └─► P2 ──┴─► P6 ─────────────────────  ┘
                  ▲
                  └── depends on the manifest contract (frozen at P1), not on P4
```

The critical path is **P0 → P1 → P3 → P4 → P5 → P7 → P8 → P9**. P4 (model training) is the longest
segment and has the most variance.

Off the critical path: the entire application. That is the point of the parallel structure.

---

## 5. Contingencies

| If | Then |
|---|---|
| C1 misses 13 dB by week 10 | Initialise from a published TSE checkpoint (WeSep/SpeakerBeam) instead of training from scratch. Costs some novelty framing, saves 3 weeks. |
| Visual conditioning shows no gain | Debug ROI/audio alignment **first** — it is nearly always the cause. If genuinely no gain, ship audio-only TSE; Novelties 1, 3, 4, 5 remain intact. |
| ~~VVX recording slips~~ | **Realised and executed** — AMI headset channels are the evaluation basis (ADR-0015). The substitution is noted in the report. |
| LRS3 agreement denied | VoxCeleb2 + AVSpeech only. Already the plan for the production checkpoint ([ADR-0013](./adr/0013-dataset-licensing.md)). |
| GPU budget exhausted | Rent cloud GPUs for the ablation sweep only — it parallelises and is the bulk of the cost. |
| Restoration (S6) doesn't work | Feature-flag off. Ship Faithful-only. Drop Novelty 4; report the negative result honestly — a documented failed hypothesis is a legitimate result. |
| Timeline compresses | Cut in this order: (1) S6 restoration, (2) HLS engine (WebAudio covers typical durations), (3) exports, (4) share links, (5) 4-speaker support. **Never cut:** sync engine, security, privacy, limitations section. |
| Sync engine fails design review in week 7 | Fall back to HLS-only with a documented ~300 ms switch gap and revise NFR-PERF-03. Better to know in week 7. |

---

## 6. Weekly cadence

- **Monday** — pick the week's issues; confirm the phase exit criteria are still the target
- **Daily** — one focused block; commit daily; keep training runs going overnight
- **Wednesday** — check training curves against
  [`07-training-playbook.md`](./07-training-playbook.md) §5; kill runs that are clearly failing
- **Friday** — demo something runnable to yourself; update the risk register; write down what was
  learned
- **Phase boundary** — verify exit criteria honestly. A phase that "mostly" meets its criteria has
  not met them; the cost of pretending compounds.
