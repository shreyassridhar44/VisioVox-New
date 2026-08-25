<div align="center">

# VisioVox

**Upload a video where people talk over each other. Pick a speaker. Hear only them.**

Per-speaker isolated audio, locked to the video, switchable during playback, with per-speaker captions.

[Documentation](./docs/README.md) · [Approach Review](./docs/02-approach-review.md) · [Implementation Plan](./docs/21-implementation-plan.md) · [Roadmap](./ROADMAP.md)

</div>

---

## What it does

When two or three people speak simultaneously on a single microphone, the recording is often
unusable. VisioVox decomposes it:

```
  ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │  ▓▓▒▒▓▓▓▒▒▒▓▓▒▓▓▓▒▒▓▓▒▒▓▓▓  │        │  Speaker 1  ▓▓▒░░░░▓▓▒░░░░▓  │  ◄── select
  │                              │   ──►  │  Speaker 2  ░░░▓▓▒░░░▓▓▒░░░  │
  │   three voices, one track    │        │  Speaker 3  ░░░░░▓▓▒░░░░▓▓▒  │
  └──────────────────────────────┘        └──────────────────────────────┘
                                            + captions, per speaker, in sync
```

- **Isolates** each speaker into a full-length track aligned to the video
- **Suppresses** the others to near-inaudible
- **Switches** instantly during playback — no gap, no drift
- **Captions** each speaker separately, word-aligned
- **Discloses** where it is uncertain, instead of hiding it

## Why it's different

Meeting transcribers give you labelled *text* and leave the audio mixed. Stem separators split music,
not speakers. Research separation models produce waveforms, not products — and lose track of who is
who over long recordings.

VisioVox is built on **audio-visual target speaker extraction**, which binds each output track to a
specific person by construction, and adds five contributions targeting the gap between separation
research and something usable:

| # | Contribution | Problem it solves |
|---|---|---|
| 1 | **Self-enrolment** from diarization | Extraction normally needs a clean reference recording per speaker. Users don't have one. |
| 2 | **Modality-adaptive conditioning** | Faces get occluded, speakers turn away, some are off-camera entirely. |
| 3 | **Suppression-first objective** | SI-SDR doesn't punish audible leakage enough. "Almost zero" is the actual requirement. |
| 4 | **Gated generative restoration** | Clean-sounding restoration can invent words. Faithfulness is the default; both tracks ship, labelled. |
| 5 | **Cross-stream leakage repair** | If the same words appear in two transcripts at the same time, one is wrong — detect it without ground truth. |
| 6 | **Switchable multi-track delivery** | Nobody ships per-speaker audio a browser can switch between in sync with video. |

Full detail: [`docs/04-novelty.md`](./docs/04-novelty.md).

## Status

🚧 **Pre-implementation.** Documentation complete; build starting at
[Phase 0](./docs/21-implementation-plan.md#phase-0--foundations-week-1).

## Architecture

```
 upload → validate (sandboxed) → ingest → dereverb/denoise
     ↓
 analyse:  diarization + overlap  ∥  face detect/track + active speaker detection
     ↓
 fuse → speaker registry
     ↓
 self-enrol (mine clean reference regions per speaker)
     ↓
 extract:  single-talker → passthrough
           overlapped   → AV target speaker extraction   ── once per speaker
     ↓
 restore (gated) → transcribe → leakage audit → package (HLS + WebVTT)
     ↓
 play:  one AudioContext, all tracks, 80 ms equal-power crossfade
```

| Layer | Stack |
|---|---|
| Model | PyTorch · TF-GridNet backbone · pyannote · LoCoNet · Whisper large-v3 |
| API | FastAPI · PostgreSQL · Redis · Celery |
| Web | Next.js 15 · React 19 · Tailwind · React Three Fiber |
| Infra | Kubernetes · Cloudflare R2 · Terraform · OpenTelemetry |

Full design: [`docs/09-system-design.md`](./docs/09-system-design.md).

## Quick start

```bash
git clone https://github.com/shreyassridhar44/VisioVox-New.git ~/projects/visiovox
cd ~/projects/visiovox                  # inside WSL2, NOT /mnt/c — see docs/23 §2
cp .env.example .env.local
make dev        # postgres, redis, minio, api, web, mock worker
make seed       # demo user + fixture projects
```

→ http://localhost:3000 · `demo@visiovox.local` / `demo1234`

The **mock pipeline** (`PIPELINE_MODE=mock`) returns fixture results, so the entire application —
upload, player, speaker switching, captions, landing page — runs with **no GPU**.

### Two machines

| Work | Machine |
|---|---|
| Frontend, API, workers, docs, infra | Laptop — mock pipeline, no GPU |
| Training, evaluation, real inference | Workstation — RTX A5000 24 GB |

Only **one model is trained** in this project: the extractor, fine-tuned from a pretrained
separation checkpoint. Every other stage uses a pretrained model or a deterministic algorithm.
Details in [`docs/25-compute-and-hardware.md`](./docs/25-compute-and-hardware.md).

```bash
make test          # full suite
make eval-quick    # 30-item ML eval (GPU)
make lint fmt typecheck
```

## Repository layout

```
apps/web           Next.js — landing, app, player
apps/api           FastAPI — control plane
services/          CPU and GPU workers
packages/          contracts (OpenAPI + manifest schema), generated clients, UI
ml/                model, pipeline, training, evaluation
infra/             terraform, docker, k8s
docs/              ← documentation set
```

## Targets

| | Target | Measured on |
|---|---|---|
| SI-SDRi (2 speakers) | ≥ 14 dB | VVX-Eval (real recordings) |
| Interferer suppression (SIR) | ≥ 20 dB | VVX-Eval |
| Target-speaker WER | ≤ 15% | VVX-Eval |
| Speaker-switch latency | ≤ 120 ms p95 | Browser, production |
| A/V drift over 10 min | ≤ 40 ms | Browser, production |

Targets are set against **real recorded conversations**, not synthetic benchmarks. They are lower
than published benchmark figures because they are measured on harder data —
[`docs/03-research-landscape.md`](./docs/03-research-landscape.md) §7.

## What it won't do

Stated up front, because the alternative is letting people find out:

- Real-time or live separation — offline only
- More than 4 speakers, and quality drops noticeably at 4
- Music or singing
- Recover a speaker who is inaudible in the source — it separates, it doesn't invent
- Identify who someone is, or match voices across videos — deliberately not built
  ([ADR-0008](./docs/adr/0008-ephemeral-biometrics.md))

## Privacy

Voiceprints and face crops are **deleted when the job finishes**. There is no biometric database.
Cross-video speaker identification is not implemented, by design. Default retention is 30 days, and
deletion produces a verifiable receipt. [`docs/16-privacy-compliance.md`](./docs/16-privacy-compliance.md).

## Documentation

25 documents and 14 ADRs: [`docs/README.md`](./docs/README.md).

Most useful entry points:
- [Approach Review](./docs/02-approach-review.md) — what changed from the original plan and why
- [Media & Sync](./docs/12-media-pipeline-and-sync.md) — how switching stays seamless
- [Implementation Plan](./docs/21-implementation-plan.md) — the build sequence

## Contributing

[`CONTRIBUTING.md`](./CONTRIBUTING.md) · Security: [`SECURITY.md`](./SECURITY.md)

## Licence

TBD before public release. Model checkpoints carry per-checkpoint licence status in their model
cards — see [ADR-0013](./docs/adr/0013-dataset-licensing.md).
