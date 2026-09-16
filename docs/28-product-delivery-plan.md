# 28 — Product Delivery Plan (Track W)

> **What this is.** The working plan to take VisioVox from "ML pipeline works, app is a skeleton"
> to a deployable product: authenticated web app, large-video upload, engaged waiting, preview,
> quality-tiered download, sharing, and the abuse controls that keep it from being expensive to run.
>
> **How to use it.** Phases are **W1–W9** — deliberately lettered so they never collide with the
> ML phase numbers in [`21-implementation-plan.md`](./21-implementation-plan.md). Track W is the
> concrete build-out of that document's **Phase 6 (Application build-out)** and
> **Phase 9 (Production hardening & launch)**.
>
> Each phase has **Exit criteria** that are testable. Do not advance on "looks done".
> Tick the boxes as work lands. This file is the resume point — read §2 and §3 first,
> then jump to the lowest phase with unticked boxes.

- **Created:** 2026-09-16
- **Audit basis:** working tree at `6eb8724`, branch `feat/phase6-playback-engine`
- **Related:** 09 (system design) · 11 (API spec) · 12 (media/sync) · 13 (frontend) · 14 (design system) · 15 (security) · 17 (infra) · 21 (implementation plan) · ADR-0009, ADR-0011, ADR-0012, ADR-0014

---

## 1. Scope

**In.** Auth UI + hardening; large-video upload with a **computed, displayed** size limit rather
than a fixed one; a processing experience that explains itself and gives an honest ETA; in-browser
preview; 1080p/720p export and download; sharing; rate limits, quotas and abuse controls; and a
**zero-cost public deployment** on the existing workstation.

**Two constraints set by the project, which shape everything below:**
1. **No fixed 5 GB cap.** Accept whatever the machine can actually process; compute the limit from
   live disk and queue state; show it up front; tell the user the ETA before they commit. (D2)
2. **No money.** Final-year project — no subscriptions, no card, nothing rented. The workstation is
   the server, reached over a free tunnel. (D5)

**Out (already done, do not rebuild).** The ML pipeline S0–S9, the dual playback engine, the
manifest contract, the data model core, the auth *service* layer. These exist and work.

**Out (deliberately deferred).** Payments/billing, teams and org accounts, mobile apps, i18n,
real-time/live processing (explicitly a non-goal per README).

---

## 2. Ground truth — what actually exists (audited 2026-09-16)

Recent commit history is almost entirely `feat(ml)`. The product surface stalled after the Phase 6
playback engine while the model work ran ahead. The honest state:

### Built and working

| Area | State |
|---|---|
| ML pipeline | `ml/pipeline/` S0–S9 complete: ingest, VAD, audio/video analysis, fuse, self-enrol, extract, transcribe, align, package, HLS |
| Playback | `apps/web/src/lib/playback/` — dual engine (HLS + WebAudio), sync, captions, manifest parsing. ~1200 lines, tested, Playwright e2e in `e2e/player.spec.ts` |
| Auth service | `auth_service.py` — register/login/refresh/logout, rotating refresh tokens, **family reuse detection**, hashed IPs |
| API control plane | `main.py` — health, auth, projects CRUD; `routes_media.py` — multipart upload init/complete, manifest, **SSE progress** |
| Data model | `models.py` — `users`, `sessions`, `refresh_tokens`, `projects`, `jobs`, `job_stages`. CHECK constraints, ULID PKs, TIMESTAMPTZ, ms-integer timing |
| Storage | `storage.py` — S3/R2 multipart, presigned URLs, MinIO locally |
| Contracts | `packages/contracts/openapi.json` + `manifest.schema.json`, generated TS client |
| Local stack | `make dev` → Postgres, Redis, MinIO, MailHog via `infra/docker/compose.yaml` |

### Specified but NOT built — these are the gaps

| # | Gap | Spec exists at | Reality |
|---|---|---|---|
| G1 | **Rate limiting** | 11 §10, 15 §9 | **Zero code.** `grep -riE "rate.?limit\|slowapi\|quota"` across `apps/ services/ packages/ ml/` returns nothing |
| G2 | **Quotas / concurrency / GPU-seconds / budget cut-out** | 11 §10, 15 §9 | No tables, no counters, no admission control |
| G3 | **Frontend** | 13, 14 | Landing page is **16 lines of unstyled HTML**. No Tailwind, no R3F/three.js in `package.json` despite ADR-0011. No upload UI at all. No processing UI. |
| G4 | **Export / download** | — | No endpoint, no muxed MP4, no quality ladder. Pipeline emits audio HLS + passthrough video only |
| G5 | **Sharing** | 11 §10 references `/shared/{token}` | No share table, no route, no OG/preview page |
| G6 | **Idempotency-Key** | 11 §1 | Declared mandatory on creating POSTs; not implemented |
| G7 | **Security headers / CSP** | 11 §11, 15 | Not implemented. CORS is the only middleware |
| G8 | **Source validation sandbox** | ADR-0009 | Not wired into the upload path |
| G9 | **Real worker** | — | `services/worker-gpu/` is a **3-line `__init__.py`**. Only `mock_pipeline.py` is wired to the API |
| G10 | **Deploy** | 17 | `infra/k8s/` and `infra/terraform/` are `.gitkeep` only |
| G11 | **5 GB uploads** | — | `max_upload_bytes = 2 GiB`; quota table caps at 2 GB |

**Read G1, G2 and G9 together:** today, an authenticated user can POST `/upload/init` in a loop
with no limit, and there is no real worker behind it. The abuse surface is wide open *and* the
thing it would abuse is not connected yet. Both get fixed in Track W.

---

## 3. Decisions taken (and the ones still open)

### D1 — Design direction: split the surfaces ✅

There is a real conflict between the brief ("funky, modern, mostly 3D, it should pop") and
[`14-design-system.md`](./14-design-system.md) §1 ("studio instrument, not consumer app… colour as
signal rather than decoration"). Both are right, about **different screens**. The resolution:

| Surface | Direction | Why |
|---|---|---|
| Landing, auth, upload, **processing/waiting** | **Expressive.** 3D ribbon separation (ADR-0011), depth, motion, glow, the full pop | These sell the idea and hold attention. Nobody is doing precision work here |
| **Player, transcript, speaker cards, export** | **Restrained studio instrument.** Dark, precise, tabular figures, colour = speaker identity only | Here the user is judging audio quality. Decoration competes with the signal |

The transition between them is itself the product metaphor: the tangled 3D ribbon *resolves* into
the clean, quiet workspace. That is a feature, not a compromise. Existing OKLCH tokens in 14 §2
stay authoritative for both — the expressive surfaces use more of the palette, not a different one.

**Action:** amend ADR-0011 to record this split, and add the expressive-surface tokens
(gradient meshes, glass/blur layers, glow) to 14 §2 rather than inventing ad-hoc colours.

### D2 — No fixed size cap. A *computed, displayed* limit ✅

The brief: don't hard-limit to 5 GB — accept any size we can actually process, tell the user
whether we can process it and how long it will take, and show the current maximum up front.

That is the right design, and it means the limit is **derived from live machine state, not a
constant in a config file**. Three independent constraints bind, and the displayed limit is the
tightest of them:

| Constraint | Formula | Why it binds |
|---|---|---|
| **Storage headroom** | `free_bytes_on_media_volume / peak_multiplier` | A job needs source + working copy + outputs on disk at once. Measure `peak_multiplier`; budget ~2.5× the source until measured |
| **Processing time** | `duration × RTF × speaker_count` | GPU time scales with *duration*, never with bytes |
| **Upload time** | `size / measured_uplink` | A 20 GB upload over a college uplink is hours. A limit the user cannot practically reach is not a limit, it is a trap |

**Bytes and minutes are different currencies, and only one of them costs GPU.** A 5 GB ProRes file
can be 12 minutes; a 400 MB H.264 file can be 3 hours. So admit on **probed duration ×
speaker count**, server-side, never on client-declared metadata — while sizing the *upload* cap
from disk.

**The model does not need the big file.** Extraction runs on 16 kHz audio plus mouth ROIs. A 4K
5 GB source is ~99% bytes the model discards. S0 derives a compact working copy immediately; the
original is touched again only at the final mux. This is what makes large uploads affordable —
and it means `peak_multiplier` is much closer to 1.3× than to 3×, once measured.

**What the user sees.** Two numbers on the upload screen, live, never stale:

> **Up to 34 GB per file · up to 60 minutes of video**
> *Based on current free space and queue. Updated just now.*

Then, the instant a file is chosen — from browser-readable metadata, before a single byte
uploads:

> `interview.mp4` — 4.2 GB, 18 min, likely 3 speakers
> **Upload ~6 min · processing ~12 min · ready by 14:35**

And if it does not fit, say exactly why and what to do:

> This file is 41 GB; the current limit is 34 GB. Free space is the constraint, not the format.
> Trimming to under 45 minutes would also work.

**ETA must be measured, not guessed.** Benchmark real RTF per stage on the A5000 once (W8), store
it as a config, then refine continuously — `job_stages.duration_ms` already records every stage of
every past job, so p50-per-stage scaled by `duration_ms × speaker_count` is available for free and
gets more accurate with use. Show a **narrowing range**, never a fake countdown.

**Hard ceiling still required.** "Any size" cannot mean unbounded: a single absurd upload must not
be able to fill the disk and take the service down. Keep a sanity cap (start at 50 GB) that sits
above the computed limit, and let the computed limit do the real work.

### D3 — Sharing: be honest about what browsers allow ✅

**You cannot post directly to Instagram from a web app.** There is no public API for it —
the Graph API requires a Business/Creator account, app review, and it does not accept arbitrary
third-party uploads. Any button claiming "share to Instagram" in a web app is a lie or a redirect.

What actually works, in order of quality:

1. **Web Share API Level 2** (`navigator.share({ files: [...] })`) — on mobile this opens the
   native sheet with Instagram, WhatsApp, Telegram, Signal, everything the device has. This is the
   real answer and it shares the **file**, not just a link.
2. **`wa.me/?text=` deep link** for WhatsApp specifically, desktop and mobile, link-only.
3. **A share link** (`/s/{token}`) with Open Graph + Twitter Card tags so the unfurl shows a
   thumbnail and duration. Revocable, optionally password- and expiry-gated.
4. Copy-link fallback everywhere, because 1 and 2 are unavailable on desktop Chrome for files.

Ship 1 + 2 + 3 + 4 and label them accurately. Do not build a fake Instagram button.

### D4 — Yes, you need the database, for more than passwords ✅

The brief asked whether a DB is needed beyond user IDs and passwords. It is, and one already
exists with six tables. Concretely, each of these is a table because it must survive a process
restart and be queried across requests:

| Need | Why it cannot live anywhere else |
|---|---|
| `projects`, `jobs`, `job_stages` | A job outlives every request and every worker process. Resume-after-crash reads `job_stages` |
| `usage_counters` | Quota enforcement must be atomic across N API replicas. Redis alone loses it on eviction |
| `share_links` | Revocation must be durable — a "revoked" share that comes back after a restart is a privacy incident |
| `exports` | An export is an artifact with a lifecycle, cost and expiry |
| `audit_events` | 15 §10 requires append-only, 1-year retention |
| `upload_sessions` | Resuming a 5 GB upload after a browser refresh needs server-side part state |

Redis stays for **hot counters, rate-limit windows and SSE progress** — fast, disposable.
Postgres is the ledger. Losing Redis should degrade performance, never correctness.

### D5 — Deployment: zero cost, self-hosted, publicly reachable ✅

**Constraint: final-year project. No subscriptions, no card, no recurring cost. Must demonstrably
be deployed and working on the public internet.**

This rules out the managed stack in [`17-infrastructure-deployment.md`](./17-infrastructure-deployment.md)
and Kubernetes. It rules *in* the thing that is already running: **the workstation is the server.**
Postgres, Redis, MinIO and Docker are live in the `VisioVox` distro today. Nothing needs renting.

```
   examiner's phone                    your workstation (WSL2 "VisioVox")
        │                                        │
   HTTPS│                                        │  outbound-only, no open ports
        ▼                                        ▼
   ┌──────────────────┐   free tunnel    ┌──────────────────────────────┐
   │ Cloudflare edge  │◄─────────────────│ cloudflared                  │
   │ TLS · WAF · CDN  │                  │  ├─ Next.js (web)            │
   └──────────────────┘                  │  ├─ FastAPI (api)            │
                                         │  ├─ Celery + GPU worker      │
                                         │  ├─ Postgres · Redis         │
                                         │  └─ MinIO  ← media on E:     │
                                         └──────────────────────────────┘
```

| Layer | Choice | Cost |
|---|---|---|
| Public ingress | **Cloudflare Tunnel** (`cloudflared`) — outbound-only, real TLS, no port forwarding, no static IP, survives CGNAT | £0 |
| Alternative ingress | **Tailscale Funnel** — `*.ts.net` HTTPS, no domain needed at all | £0 |
| Domain | Free `.me` via **GitHub Student Developer Pack** (a `bmsce.ac.in` address qualifies), or the tunnel's own subdomain | £0 |
| Landing page | **Cloudflare Pages**, static — stays up even when the workstation is off | £0 |
| App + API + workers | Workstation, Docker Compose | £0 |
| Object storage | **MinIO on E:** — S3-compatible, so ADR-0012's R2 migration path stays open for later | £0 |
| Database / cache | Postgres + Redis in the distro | £0 |
| Transactional email | Brevo or Resend free tier (~300/day) for verification mail | £0 |
| Errors / metrics | Sentry free tier, or self-hosted Grafana + Prometheus | £0 |
| CI | GitHub Actions, free on public repos | £0 |

**The honest caveat, stated up front:** the workstation must be powered on and online to process a
job. For a final-year demonstration that is fine — an examiner opens a real HTTPS URL on their own
phone, uploads a video, and gets a result. That *is* a deployment. Two things make it robust:

1. **Landing page on Cloudflare Pages**, so the URL is always up and never shows a connection
   error — it shows "processing is offline right now" if the tunnel is down.
2. **A health banner** driven by tunnel reachability, so the state is always visible rather than
   mysterious.

**Consequence for quotas (D4, W1):** with nothing rented, over-use no longer costs money — it costs
*your GPU being busy* and *your disk filling up*. So the controls stay, but their purpose shifts
from billing protection to **queue fairness and disk safety**. The "budget cut-out" in 15 §9 becomes
a **disk-headroom cut-out and a queue-depth cut-out**, which is the same mechanism pointed at the
real resource.

### D6 — Still open ⏳

1. **Tiering.** With no billing, plans are probably unnecessary — one generous limit for everyone,
   bounded by the computed cap in D2 and per-user concurrency. Confirm before W1 wires plan checks.
2. **Retention.** Default is 30 days. On a fixed disk, shorter (7 days) is safer and is itself a
   good privacy story. Recommend 7 days with a visible countdown.
3. **Public signup, or invite-only for the demo?** Invite codes remove most abuse surface for a
   project that does not need strangers.

---

## 4. Phases

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

### W0 — Storage headroom ⚠️ **BLOCKER — do this before anything else**
*Measured 2026-09-16. Nothing in W2, W6 or W8 can work until this is fixed.*

**The problem.** Real free space on the host:

| Drive | Size | Free | Note |
|---|---|---|---|
| `C:` | 477 G | **20.6 G** | Already known to be tight; nothing project-related goes here |
| `D:` | 500 G | **5.6 G** | ⚠️ **The WSL vhdx lives here** — `D:\wsl\VisioVox\ext4.vhdx`, 360 GB |
| `E:` | 454 G | **334 G** | NVMe, effectively empty. This is where media must live |

Inside the distro, `df` reports **675 GB available**. That number is fiction — it is the vhdx's
virtual maximum. The vhdx can only actually grow into the **5.6 GB** left on `D:`.

**So today the machine cannot store a single 5 GB upload, let alone process one.** This is the same
trap recorded in the workstation notes ("`df` inside WSL reports the vhdx's virtual maximum, not
real host free space… this already caused one bug"). It will present as a confusing `ENOSPC` deep
inside ffmpeg, not as a clear disk error.

- [ ] Create a dedicated media vhdx on `E:` (~250 GB) and mount it in the distro at `/srv/media`
- [ ] Point the MinIO Compose volume at `/srv/media` instead of the default Docker volume (which sits on the `D:` vhdx)
- [ ] Move `~/runs`, `~/logs` and dataset caches off the `D:` vhdx if they are growing
- [ ] Reclaim space on the `D:` vhdx afterwards (`fstrim` + `Optimize-VHD`, or `wsl --manage VisioVox --move`)
- [ ] **A disk-space check that reads `/mnt/e`, never `df /`** — wire it into the computed upload limit (D2) and into CI so the wrong check cannot come back
- [ ] Alert + automatic admission pause below a configured headroom floor (this is the D5 "budget cut-out", pointed at disk)

**Exit:** `E:`-backed media volume mounted; MinIO writing to it; the upload-limit endpoint returns a
number derived from real `/mnt/e` free space; deliberately filling the volume pauses job admission
instead of crashing a worker.

---

### W1 — Foundation: limits, quotas, headers, idempotency
*Closes G1, G2, G6, G7. Do this first — every later phase adds attack surface.*

- [ ] `apps/api/src/visiovox_api/ratelimit.py` — sliding-window Redis limiter, exact table from 11 §10
- [ ] `RateLimit-Limit` / `-Remaining` / `-Reset` headers on **every** response; `429` + `Retry-After`
- [ ] RFC 9457 Problem Details error handler with `correlation_id` (11 §1). Assert no stack traces, SQL, storage keys or model names leak into `detail`
- [ ] Exponential backoff on repeated auth failure, keyed on account, not just IP
- [ ] Migration: `usage_counters`, `audit_events`, `upload_sessions`
- [ ] Quota service: uploads/day, media-minutes/month, **GPU-seconds/month**, concurrent jobs. Enforced at **admission**, not after work is done
- [ ] Security headers middleware + CSP (11 §11). CSP must be written to allow R3F/WebGL without `unsafe-eval`
- [ ] `Idempotency-Key` on creating POSTs, Redis-backed, 24 h replay window
- [ ] Audit logging for the 15 §10 event list, with hashed IPs

**Exit:** a test suite that hammers each limited endpoint past its limit and asserts `429` +
correct headers; a test that a free-plan user cannot start a 4th upload in a day, a 2nd concurrent
job, or exceed the monthly minute budget. `make test` green.

---

### W2 — Large upload, resumable, validated, with a computed limit
*Closes G8, G11. Implements D2. Depends on W0.*

- [ ] **`GET /v1/limits`** — returns the live computed cap (free space on `/mnt/e`, queue depth, measured RTF) plus a sanity ceiling. Cached ~60 s. This is what the upload screen displays
- [ ] Replace the constant `max_upload_bytes` with that computed value; keep a 50 GB hard ceiling as a backstop, not as the user-facing number
- [ ] **Pre-flight estimate from client metadata** — read duration and resolution via `<video>` metadata before uploading a byte, and show upload ETA + processing ETA + "ready by" (D2). Server re-derives authoritatively after probe
- [ ] Admit on **probed duration × speaker count**, not on bytes
- [ ] **Reserve disk headroom at init** and release on complete/abort/timeout, so N concurrent uploads cannot jointly overcommit the volume
- [ ] **Batched part-URL issuance.** Current `upload_init` presigns all parts at once — at 30 GB that is thousands of URLs in one response. Issue in batches of ~50 via `POST /upload/{id}/parts`
- [ ] Part sizing: 16 MiB target, scaled up for very large files so part count stays under the 10,000-part S3 limit (16 MiB tops out at ~160 GB, so this is mostly headroom)
- [ ] Persist `upload_sessions` + per-part state so a browser refresh **resumes** rather than restarts
- [ ] Client uploader: `File.slice`, 4-way concurrency, per-part retry with exponential backoff, pause/resume, accurate throughput + ETA. Do **not** SHA-256 the whole file in the browser — use per-part checksums and let the server compose
- [ ] Abort/cleanup path + **S3 lifecycle rule for incomplete multipart uploads** (abandoned parts are billed storage — this is a real, silent cost leak)
- [ ] **S0 probe in a sandbox** (ADR-0009): `ffprobe` in a locked-down container, no network, CPU/mem/time capped. Reject on real duration, resolution, codec, stream count. Never trust client metadata
- [ ] Derive the compact working copy (16 kHz audio + downscaled video for ROI) immediately; the original is touched again only at final mux. **Measure the real peak disk multiplier here** and feed it back into the D2 formula
- [ ] Reject the malformed-media class: absurd duration, thousands of streams, decompression bombs

**Exit:** a file larger than the old 2 GB cap uploads, survives a mid-upload refresh and resumes, is
probed and admitted or rejected **on duration with a stated reason**; the displayed limit visibly
drops when the volume fills; a deliberately malformed file is rejected without the probe escaping
its sandbox.

---

### W3 — Design system build-out
*Closes half of G3. Implements D1.*

- [ ] Tailwind v4 + the OKLCH tokens from 14 §2 as CSS custom properties (single source of truth)
- [ ] Add expressive-surface tokens: gradient mesh, glass/blur, glow, elevation for the 3D surfaces
- [ ] Amend ADR-0011 to record the expressive/restrained split
- [ ] Typography per 14 §3 — Inter Variable, JetBrains Mono, Instrument Serif; **tabular figures on every timecode**
- [ ] Component primitives: Button, Card, Field, Toast, Dialog, Progress, Skeleton, Tooltip
- [ ] CI contrast gate — parse tokens, compute APCA/WCAG for every pairing, fail the build (14 §2 says contrast is a test, not a review comment)
- [ ] Dark default, light fully supported, `prefers-reduced-motion` honoured throughout

**Exit:** Storybook (or a `/styleguide` route) renders every primitive in both themes; contrast CI
job passes; no hard-coded colour outside the token file.

---

### W4 — Auth UI and account surface
*Depends on W3.*

- [ ] Register / login / logout / forgot-password / reset flows against the existing service layer
- [ ] Email verification (MailHog locally, real provider in prod)
- [ ] httpOnly + Secure + SameSite cookie handling for refresh; access token in memory only — **never `localStorage`**
- [ ] Silent refresh, and correct handling of the reuse-detection revocation (the service already revokes the family — the UI must log out cleanly, not loop)
- [ ] Session list + "sign out everywhere" (the `sessions` table already supports it)
- [ ] Settings: retention days, `allow_training_use`, `persist_voiceprints`, delete account + deletion receipt (16)
- [ ] Route protection, and an auth-aware server-side redirect that does not flash content

**Exit:** Playwright covers register → verify → login → refresh → logout → reuse-detection
lockout. No token in `localStorage`, asserted by a test.

---

### W5 — Upload UI and the engaged wait
*The heart of the brief.*

- [ ] Landing page: R3F tangled-ribbon-separates-on-scroll (ADR-0011), `InstancedMesh`, `dpr` cap, `frameloop="demand"`, IntersectionObserver pause, FPS watchdog → degrade → poster
- [ ] **Every fallback path from ADR-0011 tested**: reduced-motion, no WebGL2, Save-Data, slow device. WebGL must never initialise under reduced-motion
- [ ] Drag-and-drop upload with 3D depth response, per-part progress, throughput, ETA, pause/resume/cancel
- [ ] Rights attestation checkbox wired to the existing `rights_attested` check
- [ ] **Processing view driven by the existing SSE stream** — stage-by-stage, explaining the real work: "Finding faces and matching them to voices", "Learning each speaker's voice from clean moments", "Separating speaker 2 of 3"
- [ ] **Honest ETA.** `job_stages.duration_ms` is already recorded for every stage of every past job — compute p50 per stage, scale by `duration_ms × speaker_count`, and show a narrowing range, not a fake countdown. Widen the estimate when a stage runs long; never show a bar that sticks at 99%
- [ ] Ambient 3D visual that reflects genuine progress (ribbons separating as stages complete)
- [ ] Notify-on-complete: tab title, favicon badge, optional Web Push — people will switch tabs during a 20-minute job
- [ ] Reconnect on SSE drop with backoff; recover state from `GET /projects/{id}/job` on reload
- [ ] Failure states that say what to do next, per error code

**Exit:** upload → live stage narration → ready, with the tab backgrounded and the connection
dropped mid-job, and it recovers. Lighthouse LCP ≤ 2.5 s on mid-tier mobile (NFR-PERF-04).

---

### W6 — Preview, export ladder, download
*Closes G4. The existing player is the foundation — extend, do not rewrite.*

- [ ] Wire the existing `Player.tsx` + dual engine into the real project route
- [ ] Speaker switching ≤ 120 ms p95 with the 80 ms equal-power crossfade (12) — measured, not assumed
- [ ] Per-speaker captions, transcript, confidence display, leakage-audit disclosure
- [ ] **New pipeline stage S10 — render**: mux selected speaker audio into video, `-movflags +faststart`
- [ ] Quality ladder **1080p / 720p, never upscaling** past source. Record source resolution and offer only what the source supports
- [ ] `exports` table + `POST /projects/{id}/exports` (rate-limited 20/h per 11 §10), async, its own progress stream
- [ ] Export variants: single speaker, all speakers as separate tracks, audio-only stems, captions (`.vtt`/`.srt`)
- [ ] **Download delivery: presigned GET straight from R2**, `Content-Disposition`, HTTP Range so a 5 GB download is resumable. **The API never proxies bytes** (ADR-0012)
- [ ] Export lifecycle-expiry + a visible "expires in N days" so storage does not grow without bound

**Exit:** a 1080p export of a real multi-speaker video downloads, resumes after a killed connection,
and plays with the correct isolated audio in VLC and QuickTime.

---

### W7 — Sharing
*Closes G5. Implements D3.*

- [ ] `share_links` table: token, project, scope (which speaker/export), expiry, optional password, revoked_at, access count
- [ ] `GET /s/{token}` public page — player, no auth, rate-limited 60/min per token+IP (11 §10)
- [ ] Open Graph + Twitter Card with generated thumbnail so unfurls look right
- [ ] `navigator.share({ files })` with capability detection; graceful degradation
- [ ] WhatsApp `wa.me` deep link; copy-link fallback everywhere
- [ ] Revoke UI, access log, "N views" — revocation must take effect immediately
- [ ] `X-Robots-Tag: noindex` on shared pages by default; **a share link must never be guessable** (≥128 bits of entropy)

**Exit:** a share link opens for a logged-out user on mobile, the native sheet offers Instagram and
WhatsApp with the actual file, revoking kills it instantly, and the token survives a brute-force
attempt within rate limits.

---

### W8 — Real worker, real inference
*Closes G9. This is where the product stops being a mock.*

- [ ] Implement `services/worker-gpu/` properly — currently 3 lines
- [ ] Wire the real S0–S10 pipeline behind `PIPELINE_MODE=real`; keep mock working for CI and demos
- [ ] Load the trained extractor checkpoint (`~/runs/c2-v2/best.pt` or its successor) with an explicit version pin recorded on the job
- [ ] Per-stage progress → Redis → the existing SSE stream (contract already exists)
- [ ] Idempotent resume from `job_stages` (the unique `(job_id, stage, version)` index already supports it)
- [ ] **Benchmark real RTF per stage** on the A5000 across a few durations and speaker counts, and commit it as the ETA baseline (D2). Without this, every ETA in W5 is a guess
- [ ] GPU queue: single-A5000 admission control, concurrency 1, fair queueing across users, and a visible queue position
- [ ] **GPU-seconds metering** written back to `usage_counters` — this, not rate limiting, is what bounds financial risk (15 §9)
- [ ] Timeouts, OOM handling, graceful degradation to audio-only when the visual path fails
- [ ] Ephemeral-biometrics guarantee enforced in code: voiceprints and face crops deleted at job end (ADR-0008, 16)

**Exit:** a real multi-speaker video processes end-to-end on the workstation, the numbers match the
offline eval, and killing the worker mid-job resumes at the right stage rather than restarting.

---

### W9 — Deploy for free, and harden
*Closes G10. Implements D5. **Note:** 17-infrastructure-deployment.md describes the managed/K8s
path and is now aspirational — this phase supersedes it for the project as delivered.*

- [ ] Production Dockerfiles: web, api, worker-cpu, worker-gpu; multi-stage, non-root, pinned bases
- [ ] `infra/docker/compose.prod.yaml` — the whole stack, one command, on the workstation
- [ ] **Cloudflare Tunnel** (`cloudflared`) as a container in the stack; named tunnel bound to a domain (free `.me` via GitHub Student Developer Pack), or **Tailscale Funnel** if no domain
- [ ] Systemd units (or Compose restart policies) so the stack survives a reboot without a human
- [ ] **Static landing page on Cloudflare Pages** so the public URL is up even when the workstation is not, with an honest "processing offline" state
- [ ] Health banner in the app driven by real tunnel/worker reachability
- [ ] Secrets via `.env` outside the image and outside the repo; rotate `auth_secret` off the dev default — `config.py` still ships `"dev-only-insecure-secret-…"` as the fallback
- [ ] TLS + HSTS terminated at Cloudflare; MinIO **never** exposed through the tunnel — presigned URLs only, and a test asserting no public bucket access (ADR-0012)
- [ ] Sentry free tier + structured logs; a small Grafana/Prometheus or a `/admin/status` page showing queue depth, stage latency, GPU utilisation, **disk headroom**
- [ ] **Disk-headroom and queue-depth cut-outs** replacing the cloud budget cut-out (D5): pause admission automatically, surface it in the UI
- [ ] `pg_dump` on a cron to `E:`, plus a **restore drill**. An untested backup is not a backup
- [ ] Load test (k6) against the limiter and the queue — on the real tunnel, not localhost
- [ ] Runbook update (23) covering: workstation reboot, tunnel down, disk full, GPU OOM
- [ ] Demo-day checklist: pre-warm the model, confirm headroom, confirm tunnel, have a fallback recording

**Exit:** an examiner opens the public HTTPS URL on their own phone, registers, uploads a real
video, watches it process, previews it and downloads a 1080p result — with nothing rented and no
card on file. Reboot the workstation; it comes back by itself.

---

## 5. Additional recommendations

Things worth adding that the brief did not ask for, ranked by value per unit of effort.

### High value, low effort
1. **Speaker naming.** Let users rename "Speaker 2" → "Priya". Transforms the transcript from a lab output into a usable document. One column, big perceived gain.
2. **"Try it" demo without signup.** A pre-processed fixture on the landing page — the `demo/` route and fixtures already exist. Removes the auth wall from the moment of highest curiosity.
3. **Waveform-per-speaker timeline.** Visually proves separation worked, before the user listens. The data is already in the manifest.
4. **Cost/duration estimate before upload.** "This 12-minute video, 3 speakers → about 4 minutes." Sets expectations and reduces abandonment.
5. **Keyboard shortcuts** in the player (space, J/K/L, 1–4 for speakers). Cheap, and makes it feel like a tool.
6. **Auto-delete countdown** on each project, honouring `retention_days`. Privacy posture made visible.

### High value, real effort
7. **Confidence-driven UI.** 14 §1 principle 3 says uncertainty must be visible. Mark low-confidence regions in the transcript and dim the waveform there. This is a genuine differentiator — the README names it as contribution #5.
8. **Side-by-side A/B**: original mix vs isolated, one click. The single most convincing thing the product can show.
9. **Partial results streaming.** Speaker 1 is finished while speaker 3 is still extracting — let the user listen immediately. Turns a 20-minute wait into a 6-minute one, perceptually.
10. **Export presets** for podcast/YouTube/Shorts (aspect, loudness target). LUFS normalisation already exists in `s9_package.py`.

### Security and abuse, beyond the basics
11. **Upload-slot reservation.** Rate limiting `/upload/init` is not enough — each init creates a real multipart upload that costs storage even if never completed. Require a slot, release it on complete/abort/timeout.
12. **Duration-weighted quota,** not request-count quota. See D2 — this is the control that actually maps to cost.
13. **Per-user storage ceiling** with a clear "you are at 80%" state, not a surprise failure.
14. **CAPTCHA/Turnstile on register only.** Signup is the spam vector; do not tax normal use.
15. **Disposable-email blocking** on register, plus verification before first upload.
16. **Content-hash dedup.** The same file uploaded twice reuses results. Saves GPU, and incidentally blunts a re-upload spam loop.
17. **Abuse signals table** — failed logins, rejected uploads, quota hits per user; a simple score that gates admission before it needs a human.
18. **Signed URLs scoped to method AND key**, short TTL, already the documented posture — add a test that asserts a leaked URL cannot be used for a different key.
19. **Kill switch** — a config flag that stops job admission instantly without a deploy. You will want this at 2am.

### Accessibility and reach
20. **Captions are the product**, so caption rendering must be exemplary: adjustable size, high-contrast mode, screen-reader-navigable transcript with timecode jumps.
21. **Full keyboard operability** of the player and speaker switching (NFR-A11Y).
22. **The 3D must be genuinely optional.** Not "degraded" — the poster path is a first-class experience per ADR-0011.

---

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ⚠️ **`D:` has 5.6 GB free and hosts the WSL vhdx** | Cannot store one large upload; presents as a confusing `ENOSPC` inside ffmpeg, not a clear disk error | **W0, before anything else.** Media volume on `E:`; disk checks read `/mnt/e`, never `df /` |
| Large uploads on poor connections fail repeatedly | Users cannot use the product at all | Resumable parts + per-part retry (W2). Test on throttled, lossy connections, not just localhost |
| Workstation off or offline at demo time | The "deployment" appears broken to an examiner | Static landing on Pages stays up; health banner; systemd auto-start; demo-day checklist (W9) |
| Home/college uplink is slow for a 30 GB upload | User waits hours, or the demo stalls | Show measured upload ETA *before* committing (D2); recommend a trimmed clip for live demos |
| Disk fills mid-job | Worker crashes, job lost, possibly corrupt output | Headroom reservation at init + admission cut-out (W0, W2) |
| Single A5000 becomes the bottleneck | Queue grows unboundedly; ETAs become lies | Admission control + honest queue position (W8). Decide D5.1 early |
| Egress cost from large downloads | Silent budget kill | R2 zero-egress (ADR-0012) + export expiry + budget cut-out (W9) |
| 3D landing tanks mobile performance | Fails NFR-PERF-04, bad first impression | ADR-0011 rules are non-negotiable; FPS watchdog; test on a real mid-tier phone |
| Expressive redesign bleeds into the player | Undermines the precision the product sells | D1 boundary is explicit; review every player change against 14 §1 |
| Scope: W1–W9 is a lot | Stalls like the app surface already did | W1, W2, W5, W6 are the minimum viable path. W7 and parts of W9 can follow launch |
| Stale Windows clone at `C:\Users\Admin\Desktop\visiovox-claude\VisioVox-New` | Work lands in the wrong tree and is lost | The live repo is `~/visiovox/VisioVox-New` in the **VisioVox** WSL distro. Confirm `git log` before editing |

---

## 7. Suggested order

**W0** → W1 → W2 → W3 → W4 → W5 → W6 → W8 → W7 → W9

**W0 is a hard blocker** — there is no point building an upload path onto a volume with 5.6 GB
free. It is also a couple of hours of work, not days.

W1 next because everything after it widens the attack surface. W3 before W4/W5 because building UI
before the token system means rewriting it. W8 (real worker) before W7 (sharing) because sharing
mock output is not worth the surface area.

**Minimum path to a demoable product:** W0, W1, W2, W3, W5, W6 — with the mock pipeline still
behind it. That is a complete, safe, good-looking upload→wait→preview→download loop. Add W8 to make
it real, W9 to make it public.

**Suggested order if the deadline is close:** W0 → W2 → W3 → W5 → W6 → W8 → W9, with a reduced W1
(rate limits and security headers only, no quota tables) and W7 dropped. That still demonstrates a
deployed, working, defensible product; sharing is the most cuttable feature here.
