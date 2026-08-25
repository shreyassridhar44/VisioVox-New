# 11 — API Specification

Base: `https://api.visiovox.app/v1` · Media: signed URLs on a separate CDN origin.
The machine-readable OpenAPI document lives at `packages/contracts/openapi.yaml` and is the source
of truth; this file explains intent and the non-obvious parts.

---

## 1. Conventions

| Aspect | Rule |
|---|---|
| Versioning | URL path (`/v1`). Breaking changes → `/v2`; 6-month overlap. |
| Auth | `Authorization: Bearer <access_jwt>` (RS256, 10 min TTL). Browser uses the BFF; tokens never reach client JS. |
| IDs | ULID strings, prefixed: `usr_`, `prj_`, `job_`, `spk_`, `shr_` |
| Time | ISO-8601 UTC for API fields; **integer milliseconds** for media timing |
| Pagination | Cursor-based: `?cursor=&limit=` (max 100) |
| Idempotency | `Idempotency-Key` header required on all POSTs that create resources |
| Errors | RFC 9457 Problem Details |
| Rate limits | `RateLimit-Limit` / `-Remaining` / `-Reset` headers on every response |

### Error shape
```jsonc
{
  "type": "https://visiovox.app/errors/quota-exceeded",
  "title": "Monthly processing quota exceeded",
  "status": 429,
  "detail": "You have used 120 of 120 minutes on the free plan.",
  "instance": "/v1/uploads",
  "code": "QUOTA_EXCEEDED",
  "correlation_id": "01HX8…",
  "retry_after": 86400
}
```

`correlation_id` appears in the response, the logs and the trace. It is what a user quotes in a
support request and what turns a vague report into a single query.

**Never** leak internals in `detail` — no stack traces, no SQL, no storage keys, no model names.

---

## 2. Auth

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register` | Email + password; sends verification |
| POST | `/auth/verify-email` | Consume verification token |
| POST | `/auth/login` | Returns access + refresh (refresh set as httpOnly cookie by the BFF) |
| POST | `/auth/refresh` | Rotate. Reuse → revoke family (FR-ACC-04) |
| POST | `/auth/logout` | Revoke current session |
| POST | `/auth/logout-all` | Revoke every session |
| GET | `/auth/sessions` | List active sessions |
| DELETE | `/auth/sessions/{id}` | Revoke one |
| POST | `/auth/password-reset/request` | Always 202, regardless of whether the email exists |
| POST | `/auth/password-reset/confirm` | Single-use, 15-min token |
| POST | `/auth/totp/enrol` · `/verify` · `DELETE /totp` | Two-factor |
| GET | `/auth/oidc/{provider}/start` · `/callback` | Google, GitHub |

**Anti-enumeration:** `/register`, `/login` and `/password-reset/request` return the same shape and
take the same time whether or not the account exists. Password verification runs a dummy Argon2id
hash on unknown accounts so the timing is indistinguishable.

---

## 3. Upload

### `POST /uploads/init`
```jsonc
// request
{
  "filename": "interview.mp4",
  "bytes": 524288000,
  "content_type": "video/mp4",
  "sha256": "a3f1…",           // client-computed; server re-verifies
  "duration_ms_hint": 612000,   // client-probed, advisory only
  "title": "Council interview",
  "rights_attested": true       // FR-UPL-08 — required true
}
// 201
{
  "project_id": "prj_01HX…",
  "upload_id": "upl_01HX…",
  "part_size": 16777216,
  "parts": [{ "n": 1, "url": "https://…", "expires_at": "…" }],
  "expires_at": "2026-08-25T12:30:00Z"
}
```

Server checks **before** issuing any URL: quota, plan limits, size cap, duplicate `sha256`
(FR-UPL-09), `rights_attested`. A refused upload consumes no storage and no bandwidth.

Part URLs are `PUT`-only presigned URLs scoped to a single key and a single part number, expiring in
30 minutes. They are not general write access to the bucket.

### `POST /uploads/{upload_id}/complete`
```jsonc
{ "parts": [{ "n": 1, "etag": "\"abc…\"" }] }
// 202
{ "project_id": "prj_…", "job_id": "job_…", "state": "validating" }
```

### `POST /uploads/{upload_id}/abort` → `204`

---

## 4. Projects

| Method | Path | Notes |
|---|---|---|
| GET | `/projects` | `?status=&cursor=&limit=&q=` |
| GET | `/projects/{id}` | Summary + job state |
| PATCH | `/projects/{id}` | `title`, `retention_days` |
| DELETE | `/projects/{id}` | 202; returns `deletion_receipt_id` |
| GET | `/projects/{id}/manifest` | ⭐ Playback payload |
| GET | `/projects/{id}/deletion-receipt` | NFR-PRIV-04 |

### `GET /projects/{id}/manifest`

The endpoint the player depends on. Returns the artifact manifest with **freshly signed URLs**.

```jsonc
{
  "project_id": "prj_01HX…",
  "manifest_version": "1.0",
  "duration_ms": 612480,
  "has_video": true,
  "difficulty": "moderate",
  "overlap_ratio": 0.17,
  "video": { "url": "https://cdn…/video.mp4?X-Amz…", "width": 1920, "height": 1080 },
  "speakers": [{
    "id": "spk_01HX…",
    "ordinal": 1,
    "label": "Speaker 1",
    "color_token": "spk-1",
    "modality": "audiovisual",
    "thumbnail_url": "https://cdn…/spk_1.webp?X-Amz…",
    "speaking_ratio": 0.41,
    "mean_confidence": 0.86,
    "extraction_ok": true,
    "audio": {
      "faithful": { "url": "…/spk_1_f.m4a?X-Amz…", "bytes": 4820104 },
      "natural":  { "url": "…/spk_1_n.m4a?X-Amz…", "bytes": 4831992 },
      "hls":      "…/spk_1.m3u8?X-Amz…"
    },
    "peaks_url": "…/spk_1.peaks.json?X-Amz…",
    "captions": { "vtt": "…/spk_1.vtt?X-Amz…", "json": "…/spk_1.json?X-Amz…" }
  }],
  "mixed": { "audio_url": "…/mixed.m4a?X-Amz…", "hls": "…/mixed.m3u8?X-Amz…" },
  "master_playlist": "…/master.m3u8?X-Amz…",
  "playback_hint": "webaudio",       // webaudio | hls — see docs/12 §2
  "warnings": ["speaker_2_no_face_track"],
  "signed_until": "2026-08-25T12:45:00Z"
}
```

`playback_hint` lets the server choose the engine based on duration, track count and total bytes,
rather than duplicating that policy in the client.

`signed_until` lets the client re-fetch the manifest before URLs expire mid-playback — a real
failure mode for a 60-minute video with 15-minute URLs.

---

## 5. Jobs

| Method | Path | Notes |
|---|---|---|
| GET | `/jobs/{id}` | State, stage, progress, ETA |
| GET | `/jobs/{id}/events` | ⭐ **SSE** stream |
| POST | `/jobs/{id}/cancel` | 202; GPU work stops ≤ 10 s |
| POST | `/jobs/{id}/retry` | Only from `failed` with a retryable code |

### `GET /jobs/{id}/events` (Server-Sent Events)

```
event: progress
data: {"state":"processing","stage":"S5_extract","stage_label":"Isolating voices",
       "progress_pct":54,"eta_seconds":180,"detail":"Speaker 2 of 3"}

event: warning
data: {"code":"NO_FACE_TRACK","speaker_id":"spk_2",
       "message":"No face detected for Speaker 2; using audio-only isolation."}

event: ready
data: {"project_id":"prj_…","speaker_count":3}

event: error
data: {"code":"MEDIA_CORRUPT","message":"The audio stream could not be decoded.",
       "retryable":false,"correlation_id":"01HX…"}
```

SSE over WebSockets: the stream is unidirectional, it works through every proxy, it reconnects
automatically, and it needs no additional infrastructure. Heartbeat comment every 15 s to defeat
idle timeouts. Client falls back to polling `GET /jobs/{id}` at 3 s if `EventSource` fails
(FR-JOB-02).

Stage labels are **user-facing English**, not internal stage IDs. "Isolating voices", not
"S5_extract".

---

## 6. Speakers & captions

| Method | Path | Notes |
|---|---|---|
| PATCH | `/projects/{id}/speakers/{sid}` | Rename; propagates to exports (FR-PRJ-02) |
| GET | `/projects/{id}/captions?speaker_id=&format=` | `json`\|`vtt`\|`srt`\|`txt` |
| GET | `/projects/{id}/transcript?format=` | All speakers, interleaved by time |
| GET | `/projects/{id}/search?q=` | Full-text across captions; returns timestamps |

---

## 7. Exports

### `POST /projects/{id}/exports`
```jsonc
{ "kind": "mp4_with_captions",       // audio_wav|audio_mp3|mp4_with_captions|captions_srt
  "speaker_id": "spk_01HX…",
  "audio_mode": "faithful",           // faithful | natural
  "burn_captions": true }
// 202 → { "export_id": "exp_…", "state": "queued" }
```
`GET /exports/{id}` → poll; when `ready`, returns a signed download URL valid 15 minutes.

Exported filenames encode the mode — `interview_speaker-1_faithful.wav` — so a downloaded file
remains self-describing (Novelty 4's labelling requirement follows the file out of the product).

---

## 8. Sharing

| Method | Path | Notes |
|---|---|---|
| POST | `/projects/{id}/shares` | `{expires_in_hours, password?, allow_download, max_views?}` |
| GET | `/projects/{id}/shares` | List |
| DELETE | `/shares/{id}` | Revoke immediately |
| GET | `/shared/{token}` | **Public**, unauthenticated; password → `401` with a challenge |
| GET | `/shared/{token}/manifest` | Same manifest, download URLs omitted unless allowed |

Share endpoints are rate-limited by token **and** by IP; a wrong password is exponentially backed
off per token to prevent brute force.

---

## 9. Account

| Method | Path | Notes |
|---|---|---|
| GET | `/me` | Profile, plan, usage counters |
| PATCH | `/me` | `display_name`, `retention_days`, `data_region`, privacy toggles |
| GET | `/me/usage` | Current period consumption |
| POST | `/me/export` | GDPR Art. 20 data export (async) |
| DELETE | `/me` | Account deletion; requires password/TOTP re-auth |

---

## 10. Rate limits

| Endpoint class | Limit | Window | Key |
|---|---|---|---|
| `/auth/login`, `/password-reset` | 5 | 15 min | IP + email |
| `/auth/refresh` | 30 | 1 min | user |
| `/uploads/init` | 10 | 1 h | user |
| `/exports` | 20 | 1 h | user |
| `/shared/{token}` | 60 | 1 min | token + IP |
| Reads | 300 | 1 min | user |
| Global per IP | 1000 | 1 min | IP |

Sliding-window counters in Redis. On exceed: `429` + `Retry-After`. Auth failures additionally feed
an exponential backoff keyed on the account.

**Quotas are separate from rate limits** and are the real budget protection:

| Plan | Uploads/day | Max duration | Max size | Media min/month | Concurrent jobs |
|---|---|---|---|---|---|
| free | 3 | 10 min | 500 MB | 30 | 1 |
| pro | 50 | 60 min | 2 GB | 600 | 3 |
| team | 200 | 60 min | 2 GB | 3000 | 10 |

---

## 11. Security headers

Every API response:
```
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
Cache-Control: no-store            # on every authenticated response
Permissions-Policy: camera=(), microphone=(), geolocation=()
Cross-Origin-Resource-Policy: same-site
```

CORS: exact-origin allowlist, `credentials: true`, no wildcard, ever.

---

## 12. Contract testing

- **Server:** FastAPI generates OpenAPI from Pydantic models. CI fails if the generated spec differs
  from the committed `packages/contracts/openapi.yaml` — the spec cannot silently drift.
- **Clients:** TS and Python clients are generated from that spec. Hand-editing them is a CI failure.
- **Schemathesis** runs property-based tests against the live spec in CI, including malformed input,
  boundary values and auth bypass attempts.
- The **artifact manifest** has its own JSON Schema, validated at the S9 write and at the API read.
  A pipeline change that breaks the manifest fails before it reaches a browser.

This is the mechanism that lets the ML track and the app track move independently
([`09-system-design.md`](./09-system-design.md) §11).
