# 20 — Performance & Cost Engineering

---

## 1. Where the time and money go

For a 10-minute, 2-speaker video (baseline from [`05-ml-architecture.md`](./05-ml-architecture.md) §13):

| Stage | Time | % | Cost driver |
|---|---|---|---|
| S2B Video analysis | 150 s | 28% | ⭐ Per-frame face detection — the single largest cost |
| S5 Extract | 90 s | 17% | GPU inference × speakers |
| S7 Transcribe | 70 s | 13% | Whisper × speakers |
| S2A Audio analysis | 55 s | 10% | Diarization |
| S6 Restore | 45 s | 8% | Gated generative |
| S9 Package | 45 s | 8% | ffmpeg encode |
| S1 Enhance | 40 s | 8% | |
| S0 Ingest | 25 s | 5% | I/O |
| S8, S4, S3 | 37 s | 7% | |

**Two observations that shape all optimisation:**

1. **Video analysis costs more than extraction.** The "expensive AI part" is not the expensive part.
   Optimising the extractor first would be optimising 17% while ignoring 28%.
2. **Cost scales with speaker count** in S5 and S7 but not in S2B. A 4-speaker video costs roughly
   1.6×, not 2×, a 2-speaker one.

---

## 2. Optimisations, by return

### Tier 1 — large, cheap, already in the design

| Optimisation | Saving | Risk |
|---|---|---|
| **Single-talker passthrough (F11)** | ⭐ 60–80% of S5 | None — it *improves* quality |
| **VAD-gated ASR** | 40–60% of S7 | None — also suppresses hallucination |
| **Face detection at 12.5 fps + interpolation** | ⭐ 50% of S2B | Small ASD accuracy cost; measure it |
| Skip S8 when overlap < 2% | 100% of S8 | None |
| Skip S6 when φ is high everywhere | Most of S6 | None |
| Skip S1 denoise when SNR gain < 1 dB | Most of S1 | None |

The first three together take the reference job from ~9 min to roughly **5 min** — comfortably
inside NFR-PERF-02 — with no quality loss and, in two cases, a quality gain.

The pattern is worth naming: **the biggest performance wins here come from not processing things
that don't need processing**, not from making the processing faster. Adaptive routing beats kernel
optimisation.

### Tier 2 — moderate

| Optimisation | Saving | Cost |
|---|---|---|
| Batch multi-speaker extraction in one forward pass | 30% of S5 | More VRAM |
| Half-precision inference throughout | 20–30% GPU | Verify no quality change |
| Pin models in VRAM across jobs (warm worker) | ~15 s/job cold start | Idle VRAM |
| torch.compile on the extractor | 10–20% | Compile time on first run |
| Parallelise S7 across speakers | Wall clock only | More GPU |
| Frame decoding via NVDEC | 30% of S2B | Codec-dependent |

### Tier 3 — later

Distil the extractor (2–3× faster, some quality cost); ONNX/TensorRT export; quantise Whisper to
int8 (already partly via CTranslate2); replace the face detector with a lighter model at
lower resolution.

---

## 3. Frontend performance

| Lever | Effect |
|---|---|
| RSC by default | Ships less JS |
| Route-level code splitting | Player code absent from the landing page |
| Lazy 3D chunk | 400 kB excluded from initial load |
| InstancedMesh, one draw call | Hero holds 60 fps |
| `dpr={[1, 1.5]}` | ~2× fragment cost saved on retina, no visible difference |
| `frameloop="demand"` | Idle GPU usage → ~0 |
| Zustand selectors | Avoids 60 Hz re-renders — the main hazard |
| rAF direct-DOM for scrubber/captions | Hot path bypasses React |
| AVIF/WebP + `next/image` | Large image savings |
| ISR on marketing routes | Near-static delivery |

The two starred hazards for this app specifically: an over-broad Zustand subscription
(60 fps → 10 fps), and an un-disposed `AudioContext` on navigation (browsers cap concurrent
contexts; leaking them breaks playback after several project visits).

---

## 4. Cost model

### Per job (10 min, 2 speakers)

| Component | Amount | Cost |
|---|---|---|
| GPU (L4-class @ ~$0.60/h) | ~5 min after Tier-1 | $0.050 |
| CPU worker | ~2 min | $0.003 |
| Storage (source + outputs, 30 d) | ~600 MB | $0.007 |
| Egress (R2) | ~500 MB | **$0.000** |
| DB/Redis/API | — | $0.002 |
| **Total** | | **≈ $0.062** |

Against the free-plan allowance (30 media-minutes/month ≈ 3 jobs) that is ~$0.19/user/month — a
sustainable free tier.

### Sensitivity

| Change | Effect on cost/job |
|---|---|
| Speaker count 2 → 4 | +60% |
| Duration 10 → 60 min | +500% |
| Tier-1 optimisations removed | +80% |
| S3+CloudFront instead of R2 | **+$0.045** (egress) — nearly doubles it |
| GPU spot instead of on-demand | −60% |

The R2 line and the Tier-1 line are the two decisions that determine whether the unit economics work
at all.

### Controls

1. `maxReplicaCount` on the GPU pool — hard ceiling
2. Per-user GPU-second quota
3. Duration and size caps
4. Budget alerts at 50/80/100% with **automatic admission pause at 100%**
5. 7-day lifecycle on work artifacts
6. Scale-to-zero on non-production environments

Control 4 is the one that turns a runaway into a queue rather than an invoice.

---

## 5. Capacity

| Users | Jobs/mo | GPU-h/mo | GPU replicas | Monthly |
|---|---|---|---|---|
| 100 | 300 | 25 | 0–1 | ~$250 |
| 1,000 | 3,000 | 250 | 0–2 | ~$450 |
| 10,000 | 30,000 | 2,500 | 0–8 | ~$2,400 |
| 100,000 | 300,000 | 25,000 | 0–40 | ~$20,000 |

At 10k users the architecture holds. Beyond that: reserved GPU capacity, a distilled model, regional
worker pools, and Temporal in place of Celery ([ADR-0007](./adr/0007-job-orchestration.md)) become
worth the migration cost.

---

## 6. Regression prevention

| Guard | Threshold | Where |
|---|---|---|
| Bundle size | Per-route budget | CI |
| Lighthouse | ≥ 95 | CI |
| API latency | p95 < 200 ms at 100 rps | k6, nightly |
| Pipeline RTF | ≤ 2.0×, no > 20% increase | CI-EVAL |
| Cost per job | Alert on > 20% week-over-week | Dashboard |
| Player switch latency | p95 < 120 ms | Production RUM |

Cost per job is tracked as a first-class engineering metric, not a finance report. It responds
directly to code changes — disabling passthrough or raising the face-detection frame rate shows up
here within a day.
