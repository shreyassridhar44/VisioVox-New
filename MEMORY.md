# MEMORY.md

Durable project context — the things that aren't obvious from the code and would otherwise have to be
rediscovered. Append as decisions are made and lessons are learned.

---

## Current status

**Phase:** **Phase 0 complete.** Scaffold, tooling, Compose stack and the pretrained-model
smoke test all run. No application or pipeline code yet — that is Phase 1/2.

| | |
|---|---|
| Repo | `github.com/shreyassridhar44/VisioVox-New` |
| Branch | `main` |
| Contains | 27 design docs, 15 ADRs, monorepo scaffold, CI, Compose stack, smoke test |
| ML machine | College workstation — RTX A5000 24 GB, 128 GB RAM |
| App machine | Laptop, 16 GB RAM, no usable GPU (or the workstation, if access is unrestricted) |

**Workstation environment:** Windows 11 + WSL2 Ubuntu 24.04, distro named `VisioVox`, rootfs on
`E:\wsl\Ubuntu\ext4.vhdx`. Repo at `~/visiovox/VisioVox-New` on ext4. Docker Engine runs inside
the distro (not Docker Desktop). Full record and reasoning in
[`docs/26-workstation-as-built.md`](./docs/26-workstation-as-built.md).

**Phase 1 in progress — the starred exit criterion is met.** ADR-0001 has been tested empirically
and holds: permutation-error rate is **29.8%** pooled over three AMI meetings (three sites,
full-length, 1123 scored windows). Naive stitching lands **10.7 dB below returning the mixture
untouched**; the same separator output with oracle assignment and gain alignment lands **+2.55 dB
above** it. The separator is not the bottleneck — stitching is. Full write-up in
[`docs/27-phase1-baseline-report.md`](./docs/27-phase1-baseline-report.md).

Artifact manifest v1.0 is frozen (`packages/contracts/schemas/manifest.schema.json`, 13 contract
tests), which unblocks the application track (Phase 2) against a stable interface.

**Phase 1 complete** (2026-08-26). All nine tasks and all three exit criteria:
S0 ingest · S2A diarization+overlap · S2B SCRFD+ByteTrack · S3 fusion · S5 separation ·
S7 transcription · S9 packaging · manifest v1.0 frozen · 3 AMI test videos.
`uv run python scripts/run_pipeline.py` produces a **schema-valid manifest on 3/3 clips**.

ADR-0010 routing is implemented and measured: **93–97% of the AMI timeline is single-talker** and
passes through untouched. That is the concrete explanation for the un-routed Tier 0 number coming
out 10.7 dB below doing nothing — the separator was being run over audio that never needed it.

**Phase 2 complete** (2026-08-26). Exit criterion verified against the running stack, not
in-process: register -> presigned PUT to MinIO -> job queued to Celery -> a real worker picks it up
-> ready project with a schema-valid manifest and 11 stage rows, in ~11 s.

Delivered: Postgres schema + Alembic · auth with refresh rotation and family reuse detection ·
ownership as a dependency (invariant 4) · presigned multipart upload · Celery job state machine ·
SSE progress · mock pipeline worker · generated TS client with a CI drift gate · Next.js app
(auth, project list, upload, processing view).

**Phase 0 exit evidence:** `make check` green (ruff, mypy strict, eslint, tsc, pytest); `make dev`
brings up Postgres, Redis, MinIO and mail with every service answering; `make smoke` reports
7 ok / 0 skipped / 0 failed. GPU baseline: A5000 sm_86, bf16 native, 48.4 TFLOP/s bf16 matmul.

### Immediate next actions

1. ✅ **HuggingFace token** — done (2026-08-26). In `.env.local`; `make smoke` reports
   **7 ok / 0 skipped / 0 failed**. VoxCeleb2 download key is in the same file.
2. **Dataset pipeline** — running unattended in tmux. `librimix` fetches; `dataset` waits for it
   and then generates Libri2Mix (16k / min / mix_both only). Logs in `~/logs/`. Both resumable.
   Generation aborts if the host volume has under 260 GB free.
3. **Phase 3** — data. LibriMix generation is chained behind the download; VoxCeleb2 needs the
   credentials the user has; VVX recording is the longest-lead item and should start now that
   ethics is waived. Consent forms must be signed **before** recording, not after.
4. **Reclaim `C:`** — `wsl --unregister Ubuntu` removes the superseded distro (~13.5 GB); `C:` is
   down to ~2 GB. The live distro is `VisioVox` at `D:\wsl\VisioVox`; a backup of the original
   sits at `F:\wsl-backup\ext4.vhdx.bak`.
5. **Push access** — commits are authored `sharanmalali` but that account has no write permission
   on `shreyassridhar44/VisioVox-New` (403). Resolve before the next push.

### Decisions taken outside the docs

- **Ethics clearance waived** *(decided 2026-08-26)*. Proceeding without institutional review;
  VisioVox is a college project, and docs/21 §44 already notes coursework recording consenting
  adults on non-sensitive topics is commonly exempt. VVX recording is therefore unblocked and no
  longer has a six-week lead time. **Written participant consent remains mandatory** — that is a
  separate requirement from institutional review, and ADR-0008 depends on it, so the consent form
  stays in the VVX session checklist.
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

- *(2026-08-26)* **Correct per-window identity is not enough to stitch blind separation.** F1.1
  predicts permutation drift, and that is real (29.8%). But a separator also gives no guarantee
  that the same speaker emerges at the same *scale or polarity* from two independent calls, and
  SI-SDR is scale invariant so the matching never notices. Overlap-add does — adjacent windows
  cancel where they overlap. The tell was an oracle scoring *worse* than naive, which is impossible
  for an upper bound. Correcting scale was worth most of a 13.25 dB gap. When a bound is violated,
  the bound is usually the bug.

- *(2026-08-26)* **Three of the five measurement bugs in Phase 1 produced plausible numbers that
  favoured the wrong conclusion**, and none of them threw. A per-track VAD counted headset bleed as
  speech and reported 79.7% overlap where the truth was 5.6%; an absolute dB floor made a meeting
  recorded 30 dB hotter score every window as overlapped; a truncated download was accepted because
  the check was "larger than 100 KB". The habit that caught all three was checking whether a number
  was *plausible for the physical situation*, not whether the code ran.

- *(2026-08-26)* **"HuggingFace is blocked" was wrong, and the doc's remedy would have been
  wrong too.** `docs/25` §1b maps a failed reachability check to "download weights elsewhere, set
  `HF_HUB_OFFLINE=1`". The actual cause was the host's single DNS resolver timing out on A queries
  for that one domain while resolving everything else normally; connecting by IP with SNI returned
  200. Adding secondary resolvers inside WSL took it from 0/10 to 10/10. **Lesson:** distinguish
  "does not resolve" from "is blocked" with `curl --resolve host:443:<ip>` before accepting an
  offline workflow — the two have opposite remedies, and the expensive one looks plausible.

- *(2026-08-26)* **`speechbrain/sepformer-wsj02mix` is an 8 kHz model and says so only in a log
  line.** Fed 16 kHz it silently resamples down and returns two plausible-looking sources. Every
  other part of this project is 16 kHz. A Tier 0 baseline measured this way would have been
  quietly incomparable with the Tier 1 numbers it exists to be compared against. Assert a model's
  sample rate; do not assume the input rate is honoured. Caught by the Phase 0 smoke test, which
  is exactly the argument for running it before writing pipeline code.

- *(2026-08-26)* **The `~`-is-network-mounted check passed and the machine still failed the storage
  requirement.** All volumes were local, but `C:` had 1.3 GB free and the drive with the most free
  space was a USB HDD — which bottlenecks the dataloader for the same reason a network home does,
  while being invisible to `df`. Check `MediaType`/`BusType`, not just free bytes.

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
