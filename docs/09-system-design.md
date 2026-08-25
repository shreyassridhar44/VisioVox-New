# 09 — System Design

---

## 1. Design drivers

| Driver | Consequence |
|---|---|
| Jobs take minutes, not milliseconds | Asynchronous everywhere; no request holds a connection while work happens |
| GPU is the scarce, expensive resource | Dedicated worker pool, scale-to-zero, queue-based admission, per-user quotas |
| Media files are large (up to 2 GB) | Direct-to-storage upload; the API never proxies bytes |
| Media is attacker-controlled | Decoding is isolated in a sandbox with no network and no persistent identity |
| Media is private and often sensitive | Ownership checks + short-lived signed URLs; no public buckets, ever |
| Solo maintainer | Managed services over self-hosted; boring, well-documented technology |
| ML and app must progress in parallel | A versioned contract (OpenAPI + artifact manifest) separates them from day one |

---

## 2. C4 Level 1 — Context

```
   ┌──────────┐        ┌──────────────┐        ┌──────────────┐
   │  Viewer  │        │   Creator    │        │  Researcher  │
   │ (shared  │        │  (uploads,   │        │  (runs eval, │
   │  link)   │        │   listens)   │        │   trains)    │
   └────┬─────┘        └──────┬───────┘        └──────┬───────┘
        │                     │                       │
        └──────────┬──────────┘                       │
                   ▼                                  ▼
        ┌─────────────────────────┐        ┌──────────────────────┐
        │      VisioVox           │        │   ML Workbench       │
        │  (web application)      │◄───────│  (offline: train,    │
        │                         │ models │   eval, model reg.)  │
        └───┬────────┬────────┬───┘        └──────────────────────┘
            │        │        │
            ▼        ▼        ▼
     ┌────────┐ ┌────────┐ ┌──────────┐
     │  IdP   │ │ Email  │ │ Object   │
     │(Google,│ │(Resend)│ │ storage  │
     │ GitHub)│ │        │ │  + CDN   │
     └────────┘ └────────┘ └──────────┘
```

---

## 3. C4 Level 2 — Containers

```
                            ┌─────────────┐
                            │   Browser   │
                            └──────┬──────┘
                                   │ HTTPS
                    ┌──────────────▼───────────────┐
                    │  Edge / CDN  (WAF, TLS,      │
                    │  rate limit, cache)          │
                    └───┬──────────────────────┬───┘
                        │                      │  signed URLs
        ┌───────────────▼──────────┐           │  (media, direct)
        │  WEB  — Next.js 15       │           │
        │  SSR/RSC · landing ·     │           │
        │  app shell · auth UI     │           │
        │  BFF route handlers      │           │
        └───────────┬──────────────┘           │
                    │ JWT (RS256, 10 min)      │
        ┌───────────▼──────────────┐           │
        │  API — FastAPI           │           │
        │  authn/z · projects ·    │           │
        │  jobs · presign · SSE    │           │
        │  ── NO media bytes ──    │           │
        └───┬───────┬──────────┬───┘           │
            │       │          │               │
    ┌───────▼──┐ ┌──▼─────┐ ┌──▼──────────┐    │
    │ Postgres │ │ Redis  │ │  Object     │◄───┘
    │ (state,  │ │(queue, │ │  storage    │
    │  meta)   │ │ cache, │ │  (S3/R2)    │
    │          │ │ pubsub)│ │             │
    └──────────┘ └───┬────┘ └──────▲──────┘
                     │ Celery      │ read/write
         ┌───────────▼─────────────┴──────────────┐
         │  WORKER FLEET                          │
         │  ┌──────────────┐  ┌────────────────┐  │
         │  │ CPU workers  │  │  GPU workers   │  │
         │  │ ingest,      │  │  S1–S8         │  │
         │  │ package (S0, │  │  (scale-to-    │  │
         │  │ S9)          │  │   zero)        │  │
         │  │ ┌──────────┐ │  │                │  │
         │  │ │ SANDBOX  │ │  │  model cache   │  │
         │  │ │ ffmpeg   │ │  │  (volume)      │  │
         │  │ └──────────┘ │  │                │  │
         │  └──────────────┘  └────────────────┘  │
         └────────────────────────────────────────┘
```

### Container responsibilities

| Container | Tech | Owns | Explicitly does not |
|---|---|---|---|
| **Web** | Next.js 15, React 19, TS | Rendering, landing, player UI, session cookies, BFF | Business logic, media bytes, direct DB access |
| **API** | FastAPI, Python 3.12 | AuthZ, projects, jobs, presigning, SSE, quotas | Media processing, serving media bytes |
| **CPU workers** | Celery, Python | S0 ingest, S9 package — all ffmpeg work | GPU inference |
| **GPU workers** | Celery, PyTorch | S1–S8 inference | Untrusted decoding (that stays in the CPU sandbox) |
| **Postgres** | 16 | Users, projects, jobs, speakers, artifacts, audit | Media |
| **Redis** | 7 | Broker, result backend, rate limits, SSE pub/sub, cache | Durable state |
| **Object storage** | S3 / R2 | All media and artifacts | Public access |

**Why Web and API are separate.** Next.js gives the best rendering and animation story; the ML stack
is Python-only. Splitting them means each is idiomatic. The Next.js BFF layer keeps tokens in
httpOnly cookies and never exposes the API surface directly to the browser.
[ADR-0006](./adr/0006-service-topology.md).

**Why untrusted decoding is on CPU workers, not GPU workers.** GPU workers hold model weights and
are expensive to restart. Keeping the highest-risk code (ffmpeg on attacker media) in a separate,
cheap, disposable, network-isolated container limits blast radius.
[ADR-0009](./adr/0009-sandboxed-media-processing.md).

---

## 4. Request flows

### 4.1 Upload

```
Browser                Web(BFF)        API           Storage        Queue
  │  select file          │             │               │             │
  │──validate locally────►│             │               │             │
  │  POST /uploads/init   │             │               │             │
  │──────────────────────►│────────────►│               │             │
  │                       │             │ quota check   │             │
  │                       │             │ create Project(pending)     │
  │                       │             │ presign multipart──────────►│
  │◄─────── upload_id + part URLs ──────│               │             │
  │                                                     │             │
  │══════ PUT parts directly to storage ═══════════════►│             │
  │       (bytes NEVER traverse Web or API)             │             │
  │                                                     │             │
  │  POST /uploads/{id}/complete       │                │             │
  │──────────────────────►│───────────►│ complete MPU──►│             │
  │                       │            │ enqueue validate ───────────►│
  │◄─── 202 { job_id } ───│            │                │             │
```

Bytes never traverse the application. This removes the largest bandwidth cost, the largest DoS
surface, and the largest source of request-timeout problems in one decision.

### 4.2 Processing

```
Queue ─► CPU worker: VALIDATE (sandboxed ffprobe)
             │ reject → job.failed(reason), audit log
             ▼
         CPU worker: S0 INGEST (sandboxed ffmpeg)
             │ artifacts → storage
             ▼
         GPU worker: S1 → S2A ∥ S2B → S3 → S4 → S5 → S6 → S7 → S8
             │ each stage: publish progress to Redis → SSE → browser
             │ each stage: idempotent, checkpointed to storage
             ▼
         CPU worker: S9 PACKAGE
             │ manifest → storage, row → Postgres
             ▼
         job.ready  ─► SSE ─► browser
```

Modelled as a **Celery chain** with per-stage retry policy. Every stage writes its outputs to
storage and records completion in Postgres, so a worker crash resumes from the last completed stage
rather than restarting the job (NFR-REL-03).

### 4.3 Playback

```
Browser ── GET /projects/{id}/manifest ──► API
                                            │ ownership check
                                            │ mint signed URLs (15 min)
        ◄────── manifest + signed URLs ──────┘

Browser ══ GET media (signed) ══════════════► CDN ──► Storage
Browser ── select speaker ─► client-side only (tracks already loaded) ── no server round trip
```

**Speaker switching involves no network request.** That is what makes ≤ 120 ms achievable
(NFR-PERF-03). See [`12-media-pipeline-and-sync.md`](./12-media-pipeline-and-sync.md).

---

## 5. Job orchestration

### State machine

```
      ┌─────────┐
      │ pending │ (upload in progress)
      └────┬────┘
           ▼
      ┌──────────┐   invalid    ┌────────┐
      │validating├─────────────►│ failed │
      └────┬─────┘              └────────┘
           ▼                         ▲
      ┌────────┐                     │ any stage, permanent error
      │ queued │                     │
      └────┬───┘                     │
           ▼                         │
   ┌───────────────┐                 │
   │  processing   │─────────────────┘
   │ (stage, pct)  │
   └───────┬───────┘         ┌───────────┐
           ▼                 │ cancelled │◄── user cancel (any active state)
      ┌────────┐             └───────────┘
      │ ready  │
      └────┬───┘
           ▼ retention window elapsed
      ┌─────────┐
      │ expired │
      └─────────┘
```

Transitions are guarded in the DB (`CHECK` constraint + application-level state machine), so an
out-of-order worker message cannot corrupt state.

### Retry policy

| Failure | Policy |
|---|---|
| GPU OOM | Retry once with reduced chunk size; then fall back to the audio-only path |
| Worker crash / preemption | Retry from last completed stage, max 3, exponential backoff + jitter |
| Model download failure | Retry 5×; alert if persistent (worker image is misconfigured) |
| Corrupt media | Permanent failure, no retry |
| Quota exceeded | Permanent failure with a user-actionable message |
| Timeout (> 3× expected) | Kill, retry once, then fail with a diagnostic |

### Idempotency
Every stage is keyed by `(job_id, stage, pipeline_version)`. If its output artifacts already exist
and their checksums match, the stage is skipped. This makes retries free and makes pipeline
re-runs after a partial failure cheap.

---

## 6. Data flow and storage layout

```
s3://visiovox-media/
  u/{user_id}/p/{project_id}/
    src/original.{ext}                 ← uploaded, never mutated
    work/                              ← intermediates, lifecycle-deleted after 7 days
      analysis.wav  enhanced.wav  frames/  diarization.rttm  face_tracks.json
      enrolments/spk_{k}.npz
    out/                               ← delivered artifacts
      video/  audio/spk_{k}.{faithful,natural}.m4a  hls/  captions/
      thumbs/ peaks/  manifest.json
```

| Class | Retention | Lifecycle |
|---|---|---|
| `src/` | user retention window (default 30 d) | delete with project |
| `work/` | 7 days | auto-delete — large and reproducible |
| `out/` | user retention window | delete with project |
| Biometric derivatives (`enrolments/`, face crops) | **job duration only** | deleted at S9 unless opt-in (NFR-PRIV-01) |

Bucket policy: **no public access, ever.** Every read is a short-lived signed URL issued after a
server-side ownership check.

---

## 7. Scaling

| Component | Scaling | Bottleneck |
|---|---|---|
| Web | Horizontal, stateless | CPU on SSR |
| API | Horizontal, stateless | DB connections → PgBouncer |
| CPU workers | Horizontal by queue depth | ffmpeg CPU |
| **GPU workers** | **Queue depth, scale-to-zero, cap** | **VRAM — the real constraint** |
| Postgres | Vertical + read replicas | Writes |
| Redis | Vertical; cluster if needed | Memory |
| Storage | Managed | — |

**GPU capacity policy:** a hard concurrency cap with a queue in front, not autoscale-to-infinity.
Unbounded GPU autoscaling is an unbounded bill. When the cap is reached, jobs queue with an ETA
shown in the UI (FR-JOB-07) — visible waiting is a better outcome than a surprise invoice
(NFR-REL-04).

Separate queues by expected duration (`short` < 5 min, `long` ≥ 5 min) so a 60-minute job cannot
head-of-line block a stream of 2-minute jobs.

---

## 8. Failure modes and degradation

| Failure | Behaviour |
|---|---|
| GPU fleet down | Jobs queue; UI shows maintenance state; existing projects remain fully playable |
| Redis down | New jobs rejected with 503; playback unaffected (CDN-served) |
| Postgres down | Full outage of the app; CDN-cached media still resolves for open sessions |
| Object storage down | Uploads and playback fail; app shell still renders |
| Restoration model unavailable | Faithful track only; `natural` omitted from manifest; UI hides the toggle |
| ASD model fails | Audio-only path; labelled chips instead of thumbnails |
| One speaker's extraction fails | Other speakers still delivered; that speaker marked failed in the registry |

The last two are the important ones: **partial pipeline failure must yield partial results, never a
failed job.** A job that produces 2 of 3 speakers is far more useful than one that produces nothing
(Charter principle 3).

---

## 9. Environments

| Env | Purpose | Data | GPU |
|---|---|---|---|
| `local` | Development | Seeded fixtures | Optional; mock pipeline by default |
| `ci` | Automated tests | Ephemeral | No — mock pipeline |
| `staging` | Pre-production | Synthetic + demo clips | 1 worker, scale-to-zero |
| `production` | Live | Real user data | Autoscaled to cap |

Staging mirrors production configuration exactly, differing only in scale and secrets. A change that
has not run in staging does not go to production.

**The mock pipeline is a first-class deliverable**, not a testing afterthought. It returns a
pre-computed manifest from fixture data, letting the entire frontend and API be developed and tested
without a GPU. This is what makes the parallel-track schedule in
[`21-implementation-plan.md`](./21-implementation-plan.md) possible.

---

## 10. Technology decisions

| Layer | Choice | Rejected | Reason |
|---|---|---|---|
| Frontend | Next.js 15 + React 19 | SvelteKit, plain Vite | RSC, ecosystem, React Three Fiber |
| Styling | Tailwind + shadcn/ui | MUI, Chakra | Own the components; no runtime CSS-in-JS cost |
| 3D | React Three Fiber (three.js) | raw three.js, Spline | Declarative, SSR-safe, code-splittable |
| Animation | Motion (Framer Motion) + GSAP for scroll | CSS only | Complex orchestration; respects reduced-motion |
| API | FastAPI | Django, Express | Async, Pydantic, OpenAPI generation, Python for ML parity |
| Queue | Celery + Redis | RQ, Dramatiq, Temporal | Mature canvas/chains; Temporal is the documented scale path — [ADR-0007](./adr/0007-job-orchestration.md) |
| DB | PostgreSQL 16 | MongoDB | Relational data, transactions, JSONB where needed |
| ORM | SQLAlchemy 2 + Alembic | Prisma, raw SQL | Async, mature migrations |
| Storage | S3-compatible (R2 prod, MinIO local) | Local FS | Durability, signed URLs, **zero egress on R2** |
| Auth | Auth.js (web) + RS256 JWT (API) | Clerk, Auth0 | Self-hosted, no per-user cost — [ADR-0014](./adr/0014-authentication.md) |
| Observability | OpenTelemetry → Grafana stack | Datadog | Cost; vendor-neutral instrumentation |
| IaC | Terraform | Pulumi, manual | Standard, reviewable |
| CI/CD | GitHub Actions | — | Integrated |

---

## 11. Repository layout

```
visiovox/
├── apps/
│   ├── web/                 Next.js — landing, app, player
│   └── api/                 FastAPI — control plane
├── services/
│   ├── worker-cpu/          ingest, packaging (sandboxed ffmpeg)
│   └── worker-gpu/          S1–S8 inference
├── packages/
│   ├── contracts/           OpenAPI spec + manifest JSON Schema (source of truth)
│   ├── ts-client/           generated TS client
│   └── py-client/           generated Python client
├── ml/
│   ├── seave/               model, losses, conditioning
│   ├── pipeline/            stage implementations
│   ├── training/            curriculum, configs
│   ├── eval/                harness, ablations
│   └── notebooks/
├── infra/
│   ├── terraform/
│   ├── docker/
│   └── k8s/
├── docs/
└── scripts/
```

`packages/contracts` is the **source of truth** for the Web↔API↔Worker boundary. Clients are
generated, never hand-written; a contract change that breaks a client fails CI. This is the
mechanism that keeps the parallel tracks honest.
