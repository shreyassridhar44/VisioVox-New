# 01 — Requirements

Requirement IDs are stable and referenced from tests, ADRs and the implementation plan.
Priority uses MoSCoW: **M**ust / **S**hould / **C**ould / **W**on't (v1).

---

## 1. Functional requirements

### 1.1 Account & access — `FR-ACC`

| ID | Requirement | Pri |
|---|---|---|
| FR-ACC-01 | User can register with email + password (Argon2id) or an OIDC provider (Google, GitHub) | M |
| FR-ACC-02 | Email verification required before first upload | M |
| FR-ACC-03 | Session via httpOnly, Secure, SameSite=Lax cookie; short-lived access JWT for the API | M |
| FR-ACC-04 | Refresh token rotation with reuse detection → revoke entire family on reuse | M |
| FR-ACC-05 | Password reset via single-use, 15-minute, constant-time-compared token | M |
| FR-ACC-06 | User can view active sessions and revoke any or all | S |
| FR-ACC-07 | TOTP two-factor authentication | S |
| FR-ACC-08 | Account deletion cascades to all projects, artifacts and biometric derivatives within 24 h | M |
| FR-ACC-09 | Per-plan quotas: uploads/day, max duration, max file size, concurrent jobs | M |

### 1.2 Upload & ingest — `FR-UPL`

| ID | Requirement | Pri |
|---|---|---|
| FR-UPL-01 | Drag-and-drop or file-picker upload of a single video | M |
| FR-UPL-02 | Direct-to-object-storage upload via presigned multipart URLs (never proxied through the API) | M |
| FR-UPL-03 | Accept `mp4`, `mov`, `mkv`, `webm`, `avi`; and audio-only `wav`, `mp3`, `m4a`, `flac` | M |
| FR-UPL-04 | Type verified by magic bytes **and** `ffprobe`, never by extension or client `Content-Type` | M |
| FR-UPL-05 | Reject: > 2 GB, > 60 min, > 4K, > 120 fps, > 8 audio channels, or 0 audio streams | M |
| FR-UPL-06 | Resumable upload; browser refresh mid-upload does not lose progress | S |
| FR-UPL-07 | Client-side pre-check (duration, size, has-audio) before bytes are sent | S |
| FR-UPL-08 | Uploader must affirm they have the right to process the recording (logged with timestamp + IP) | M |
| FR-UPL-09 | Content-addressed dedupe by SHA-256 — re-uploading a processed file reuses results | C |

### 1.3 Processing pipeline — `FR-PIPE`

| ID | Requirement | Pri |
|---|---|---|
| FR-PIPE-01 | Automatically detect the number of distinct speakers (2–4) | M |
| FR-PIPE-02 | Produce a speaker registry: id, label, face thumbnail (if any), speaking-time %, colour token | M |
| FR-PIPE-03 | Detect and track faces; map face tracks to voices via active speaker detection | M |
| FR-PIPE-04 | Produce one **full-length** isolated audio track per speaker, sample-aligned to the video, silence-padded where that speaker is not speaking | M |
| FR-PIPE-05 | Apply dereverberation + denoising before extraction | M |
| FR-PIPE-06 | Route single-talker regions around the extractor (passthrough) with crossfaded boundaries | M |
| FR-PIPE-07 | Normalise every track to −16 LUFS integrated, −1 dBTP true peak | M |
| FR-PIPE-08 | Produce per-speaker captions with word-level timestamps | M |
| FR-PIPE-09 | Emit a per-segment confidence/trust score per speaker track | M |
| FR-PIPE-10 | Emit both a "faithful" and a "natural" (restored) variant of each track | S |
| FR-PIPE-11 | Cross-stream leakage audit and repair before packaging | S |
| FR-PIPE-12 | Degrade gracefully: no faces → audio-only; contaminated enrolment → visual-only; never fail outright | M |
| FR-PIPE-13 | Job progresses through named, observable stages with % completion | M |
| FR-PIPE-14 | Stages are idempotent and individually retryable; a stage crash does not restart the whole job | M |
| FR-PIPE-15 | Handle audio-only uploads (no video) through the audio-only path | S |

### 1.4 Job lifecycle — `FR-JOB`

| ID | Requirement | Pri |
|---|---|---|
| FR-JOB-01 | Job states: `queued → validating → analyzing → extracting → transcribing → packaging → ready` plus `failed`, `cancelled`, `expired` | M |
| FR-JOB-02 | Live progress via SSE, with polling fallback | M |
| FR-JOB-03 | User can cancel a running job; GPU work stops within 10 s | M |
| FR-JOB-04 | Failures surface a human-readable reason and a correlation ID | M |
| FR-JOB-05 | Transient failures auto-retry with exponential backoff (max 3), permanent failures do not | M |
| FR-JOB-06 | Optional email notification on completion | C |
| FR-JOB-07 | Queue position and ETA shown while queued | S |

### 1.5 Playback — `FR-PLAY`

| ID | Requirement | Pri |
|---|---|---|
| FR-PLAY-01 | Video plays with the original mixed audio by default | M |
| FR-PLAY-02 | Speaker selector: face thumbnails when available, otherwise labelled chips | M |
| FR-PLAY-03 | Selecting a speaker swaps to their isolated track with **no perceptible seam** (≤ 120 ms, equal-power crossfade) | M |
| FR-PLAY-04 | Captions swap with the selected speaker | M |
| FR-PLAY-05 | A/V sync maintained ≤ 40 ms drift over 10 minutes of continuous playback | M |
| FR-PLAY-06 | Correct behaviour across seek, scrub, pause/resume, playback-rate change, tab backgrounding, and device output change | M |
| FR-PLAY-07 | "Mixed" option always available to return to the original audio | M |
| FR-PLAY-08 | Speaking speaker is highlighted on the video overlay in real time (when ASD data exists) | S |
| FR-PLAY-09 | Interactive transcript: click a word to seek | S |
| FR-PLAY-10 | Low-confidence caption segments visually marked | S |
| FR-PLAY-11 | "Faithful / Natural" audio-mode toggle | S |
| FR-PLAY-12 | Per-speaker waveform showing where each speaker talks, with overlap regions marked | S |
| FR-PLAY-13 | Solo/mute mixer — hear any subset of speakers together | C |
| FR-PLAY-14 | Keyboard shortcuts: `space`, `←/→`, `1–4` (select speaker), `m`, `f`, `c` | M |
| FR-PLAY-15 | A/B compare: instant toggle between mixed and isolated at the same timestamp | C |

### 1.6 Outputs & sharing — `FR-OUT`

| ID | Requirement | Pri |
|---|---|---|
| FR-OUT-01 | Download a speaker's isolated audio (WAV or MP3) | M |
| FR-OUT-02 | Download captions as SRT, VTT or JSON | M |
| FR-OUT-03 | Download the full transcript with all speakers, timestamped | M |
| FR-OUT-04 | Export an MP4 muxed with one speaker's isolated audio and burned-in captions | S |
| FR-OUT-05 | Read-only share link, expiring, revocable, optionally password-protected | S |
| FR-OUT-06 | Downloads served via short-lived (≤ 15 min) signed URLs | M |

### 1.7 Project management — `FR-PRJ`

| ID | Requirement | Pri |
|---|---|---|
| FR-PRJ-01 | List all projects with status, thumbnail, duration, speaker count, created date | M |
| FR-PRJ-02 | Rename a project; rename a speaker (label persists into captions and exports) | M |
| FR-PRJ-03 | Delete a project — hard-deletes all artifacts and derivatives within 24 h | M |
| FR-PRJ-04 | Search and filter projects | C |
| FR-PRJ-05 | Automatic expiry after the retention window with prior warning | M |

### 1.8 Marketing site — `FR-SITE`

| ID | Requirement | Pri |
|---|---|---|
| FR-SITE-01 | Landing page with animated 3D hero communicating "one mixed waveform → separate voices" | M |
| FR-SITE-02 | Interactive live demo on the landing page — a preloaded sample the visitor can switch speakers on without signing up | M |
| FR-SITE-03 | How-it-works section explaining the pipeline visually | M |
| FR-SITE-04 | Pricing/plans page | S |
| FR-SITE-05 | Legal pages: privacy policy, terms, DPA, acceptable use | M |
| FR-SITE-06 | Full functionality with JavaScript animations disabled (`prefers-reduced-motion`) | M |

---

## 2. Non-functional requirements

### 2.1 Quality of the core model — `NFR-ML`

| ID | Requirement | Target | Floor |
|---|---|---|---|
| NFR-ML-01 | SI-SDRi, 2 speakers, AMI-Eval | ≥ 14 dB | 11 dB |
| NFR-ML-02 | SI-SDRi, 3 speakers, AMI-Eval | ≥ 11 dB | 8 dB |
| NFR-ML-03 | SIR (interferer suppression), 2 spk | ≥ 20 dB | 16 dB |
| NFR-ML-04 | DNSMOS-P.835 OVRL | ≥ 3.2 | 3.0 |
| NFR-ML-05 | Target-speaker WER, AMI-Eval | ≤ 15% | 22% |
| NFR-ML-06 | Cross-stream leakage word rate | ≤ 3% | 6% |
| NFR-ML-07 | Diarization error rate (DER) | ≤ 12% | 18% |
| NFR-ML-08 | Speaker-count accuracy (2–4) | ≥ 92% | 85% |
| NFR-ML-09 | ASD face↔voice mapping accuracy | ≥ 90% | 82% |
| NFR-ML-10 | No regression > 0.5 dB SI-SDRi between releases | hard gate | — |

### 2.2 Performance — `NFR-PERF`

| ID | Requirement | Target |
|---|---|---|
| NFR-PERF-01 | Pipeline real-time factor (RTF) on the reference GPU | ≤ 2.0× wall-clock per speaker |
| NFR-PERF-02 | 10-min, 2-speaker video end-to-end | ≤ 8 min |
| NFR-PERF-03 | Speaker-switch latency (click → audible) | ≤ 120 ms p95 |
| NFR-PERF-04 | Landing page LCP, mid-tier mobile, 4G | ≤ 2.5 s |
| NFR-PERF-05 | API p95 latency (non-upload, non-job) | ≤ 200 ms |
| NFR-PERF-06 | Player first-frame-to-interactive | ≤ 1.5 s |
| NFR-PERF-07 | Landing 3D hero sustained frame rate | ≥ 50 fps desktop, ≥ 30 fps mobile |
| NFR-PERF-08 | Client JS budget (landing, initial) | ≤ 250 kB gzipped excl. 3D chunk |

### 2.3 Reliability — `NFR-REL`

| ID | Requirement | Target |
|---|---|---|
| NFR-REL-01 | API availability | 99.5% monthly |
| NFR-REL-02 | Job success rate on valid input | ≥ 98% |
| NFR-REL-03 | Zero data loss on worker crash — job resumes from last completed stage | hard |
| NFR-REL-04 | Graceful degradation when GPU capacity is exhausted (queue, don't drop) | hard |
| NFR-REL-05 | RPO ≤ 24 h, RTO ≤ 4 h | hard |

### 2.4 Security — `NFR-SEC`

Full detail in [`15-security.md`](./15-security.md).

| ID | Requirement |
|---|---|
| NFR-SEC-01 | All media decoding runs in a sandbox: non-root, seccomp-filtered, no network, read-only rootfs, wall-clock + memory caps |
| NFR-SEC-02 | Every artifact access is ownership-checked server-side; signed URLs ≤ 15 min, single-purpose |
| NFR-SEC-03 | Encryption in transit (TLS 1.3) and at rest (AES-256, SSE-KMS) |
| NFR-SEC-04 | Strict CSP with per-request nonces; no `unsafe-inline`, no `unsafe-eval` |
| NFR-SEC-05 | Rate limiting on all mutating and auth endpoints; per-user GPU-minute quota |
| NFR-SEC-06 | No secret in source control; CI enforces secret scanning |
| NFR-SEC-07 | Dependency scanning + SBOM per build; block on critical CVE |
| NFR-SEC-08 | Structured audit log for authn, authz denial, upload, download, share, delete |
| NFR-SEC-09 | OWASP ASVS L2 conformance |
| NFR-SEC-10 | Container images signed; deployments verify signatures |

### 2.5 Privacy — `NFR-PRIV`

| ID | Requirement |
|---|---|
| NFR-PRIV-01 | Speaker embeddings and face crops are job-scoped; deleted with the job |
| NFR-PRIV-02 | Cross-video speaker identification disabled by default; opt-in only |
| NFR-PRIV-03 | Default retention 30 days; user-configurable down to 24 h |
| NFR-PRIV-04 | Deletion is verifiable — a delete receipt enumerates removed object keys |
| NFR-PRIV-05 | No customer media used for training without explicit, separate, revocable opt-in |
| NFR-PRIV-06 | Data residency selectable (EU / US / IN) at workspace creation |
| NFR-PRIV-07 | Full data export (GDPR Art. 20) within 30 days of request |

### 2.6 Accessibility — `NFR-A11Y`

| ID | Requirement |
|---|---|
| NFR-A11Y-01 | WCAG 2.2 Level AA |
| NFR-A11Y-02 | Player fully keyboard-operable; visible focus ring throughout |
| NFR-A11Y-03 | Speaker selection announced to screen readers via a live region |
| NFR-A11Y-04 | `prefers-reduced-motion` disables 3D and all non-essential motion |
| NFR-A11Y-05 | Contrast ≥ 4.5:1 body, ≥ 3:1 large text and UI boundaries |
| NFR-A11Y-06 | Speakers distinguishable without relying on colour alone (shape + label + colour) |
| NFR-A11Y-07 | Captions restyleable: size, font, background opacity |

### 2.7 Observability — `NFR-OBS`

| ID | Requirement |
|---|---|
| NFR-OBS-01 | Distributed tracing (OpenTelemetry) from browser through API to GPU worker stages |
| NFR-OBS-02 | Structured JSON logs with correlation ID on every request and job |
| NFR-OBS-03 | RED metrics for services; USE metrics for GPU workers |
| NFR-OBS-04 | Per-stage duration, VRAM peak and failure-rate metrics |
| NFR-OBS-05 | SLO-based alerting with burn-rate windows; no alert without a runbook link |

### 2.8 Maintainability — `NFR-MAINT`

| ID | Requirement |
|---|---|
| NFR-MAINT-01 | Backend ≥ 80% line coverage, ML pipeline ≥ 70%, frontend ≥ 70% |
| NFR-MAINT-02 | Types enforced end-to-end: mypy strict, TypeScript strict, generated API client |
| NFR-MAINT-03 | Every architectural decision recorded as an ADR |
| NFR-MAINT-04 | One-command local bring-up (`make dev`) including seeded data |
| NFR-MAINT-05 | Database migrations versioned and reversible (Alembic) |
| NFR-MAINT-06 | Model artifacts versioned with a model card and reproducible training config |

### 2.9 Compatibility — `NFR-COMPAT`

| ID | Requirement |
|---|---|
| NFR-COMPAT-01 | Chrome/Edge ≥ 111, Firefox ≥ 115, Safari ≥ 16.4 (last 2 major versions) |
| NFR-COMPAT-02 | iOS Safari supported, including its autoplay/`AudioContext` unlock constraints |
| NFR-COMPAT-03 | Responsive 360 px → 2560 px |
| NFR-COMPAT-04 | Graceful fallback where WebGL2 is unavailable (static hero) |

---

## 3. Acceptance scenarios (Gherkin)

```gherkin
Feature: Speaker isolation and selection

  Scenario: Seamless switch during overlapping speech
    Given a processed 10-minute video with 3 detected speakers
    And playback is at 04:12, inside a region where speakers A and B overlap
    When I select speaker A
    Then audio for speaker A is audible within 120 ms
    And speaker B's voice is suppressed by at least 18 dB relative to the mixed track
    And captions displayed are speaker A's only
    And no audio gap, click or level jump occurs during the switch
    And A/V sync error remains under 40 ms

  Scenario: Sync survives seeking
    Given I am listening to speaker B's isolated track
    When I scrub to 08:30 and release
    Then video and speaker B's audio resume together within 40 ms
    And the caption shown corresponds to 08:30 in speaker B's transcript

  Scenario: Graceful degradation with no visible faces
    Given an uploaded video where no face is detectable
    When processing completes
    Then the job status is "ready"
    And speakers are presented as labelled chips instead of thumbnails
    And a notice explains that visual conditioning was unavailable
    And isolated tracks are still produced via the audio-only path

  Scenario: Low-confidence segment is disclosed, not hidden
    Given a segment where extraction confidence is below 0.4
    When that segment plays
    Then the caption is visually marked as low-confidence
    And a tooltip explains that isolation may be unreliable here

  Scenario: Hostile upload is rejected safely
    Given a file whose extension is .mp4 but whose magic bytes are a ZIP archive
    When I attempt to upload it
    Then the upload is rejected before any decoding occurs
    And no ffmpeg process is spawned
    And the event is written to the audit log

  Scenario: Deletion is complete and verifiable
    Given a completed project with isolated tracks and speaker embeddings
    When I delete the project
    Then all media objects and biometric derivatives are removed within 24 hours
    And a delete receipt enumerating removed object keys is available
    And previously issued signed URLs no longer resolve
```

---

## 4. Traceability

| Goal | Requirements | Verified by |
|---|---|---|
| G1 Isolation | FR-PIPE-04/05/06, NFR-ML-01/02 | Eval harness on AMI-Eval |
| G2 Suppression | NFR-ML-03/06 | Eval harness + listening test |
| G3 Listenability | FR-PIPE-07/10, NFR-ML-04 | DNSMOS + MOS panel |
| G4 Switching | FR-PLAY-03/05/06, NFR-PERF-03 | Playwright timing harness |
| G5 Captions | FR-PIPE-08, NFR-ML-05 | WER harness |
| G6 Production | All NFR-SEC/REL/OBS | Release gate checklist |
| G7 Novelty | FR-PIPE-06/09/10/11 | Ablation suite |
| G8 Looks | FR-SITE-01/02, NFR-PERF-04/07 | Lighthouse CI |
