# W5 — Upload UI and the engaged wait

**State:** ✅ Done
**Plan of record:** [`../28-product-delivery-plan.md`](../28-product-delivery-plan.md) §W5
**Depends on:** W2 ✅, W3 ✅

---

## Tasks

- [x] Landing page with the separating-ribbon hero (ADR-0011)
- [x] Every ADR-0011 fallback verified: reduced motion, no canvas, content complete without it
- [x] `InstancedMesh`, `dpr` cap, `frameloop` on demand, IntersectionObserver pause, FPS watchdog
- [x] Drag-and-drop upload with depth response
- [x] Live limits shown before a file is chosen
- [x] Pre-flight metadata read in the browser before a byte moves
- [x] Rights attestation wired to the existing `rights_attested` check
- [x] Resumable uploader: slicing, 4-way concurrency, per-part retry, pause/resume/cancel
- [x] Throughput and ETA from an exponentially-weighted rate
- [x] Processing view driven by the existing SSE stream, narrating real stages
- [x] Honest ETA — a widening range, never a countdown
- [x] Tab title carries progress for a backgrounded tab
- [x] SSE reconnect with capped backoff; slow poll as a safety net
- [ ] Favicon badge and Web Push — tab title only for now
- [ ] Ambient 3D that reflects job progress — the hero is scroll-linked, not job-linked

---

## Decisions made while building

- **2026-09-16 — `XMLHttpRequest` for part uploads, not `fetch`.** `fetch` cannot report progress
  *within* a request, so a 16 MiB part is either 0% or 100% and the bar sits still for a minute on
  a slow link. XHR's `upload.progress` is the only way to show movement inside a part.
- **2026-09-16 — no whole-file hash in the browser.** Hashing several gigabytes on the main thread
  freezes the tab for minutes before anything moves. Per-part ETags already give the object store
  its integrity check.
- **2026-09-16 — throughput is exponentially weighted.** A plain average stops responding once
  enough samples accumulate, so a connection that has died reads as healthy for minutes.
- **2026-09-16 — a failed part drops its presigned URL before retrying.** Expiry mid-upload is the
  common failure, and retrying a dead link just burns the attempt budget.
- **2026-09-16 — stage copy is keyed on the worker's stage identifiers.** A renamed stage shows its
  raw name rather than a confidently wrong description; the narration cannot drift into fiction
  while the backend does something else.
- **2026-09-16 — the remaining-time estimate *widens*.** Narrowing a guess as it becomes less
  reliable is how progress bars come to be distrusted.
- **2026-09-16 — the SSE token travels on the query string, and the trade is written down.**
  `EventSource` cannot set an Authorization header. URLs reach access logs and `Referer` in a way
  headers do not, so this is acceptable only for a 10-minute revocable access token. A refresh
  token must never go this way. The access log must not record query strings for that path.
- **2026-09-16 — the hero is a client island, not a client page.** The App Router forbids
  `ssr: false` in a Server Component, and the scene genuinely needs a WebGL context, so the dynamic
  import lives in the smallest possible wrapper and the landing page stays server-rendered.

---

## Measurements

**2026-09-16 — full loop in a real browser** (API + Celery worker + MinIO + Postgres):

```
registered and signed in
limits shown live: Up to 50.0 GB per file · up to 60 minutes of video
read locally before upload: 286 KB · 20s · 640×480
processing estimate: about 10s–28s
upload completed -> /projects/prj_01M2N7KSAWWRH1EHAK3X0MTY9W
processing view: Getting started
final status: ready
result: 2 speakers · overlap 7% · easy
player rendered: yes, 3 speaker cards
worker: run_mock_pipeline succeeded in 11.05s
```

**ADR-0011 fallback matrix, each verified by putting the browser in that state:**

| Condition | Expected | Result |
|---|---|---|
| default | canvas mounts | `canvas=1` ✅ |
| `prefers-reduced-motion` | WebGL never initialises | `canvas=0` ✅ |
| no canvas at all | page complete and actionable | CTA visible, all sections present ✅ |

**Bundle:** landing 5.97 kB, 108 kB first load — three.js is code-split, not in the shared chunk.

---

## Gotchas

- **`NEXT_PUBLIC_*` is inlined at BUILD time.** Setting `NEXT_PUBLIC_API_URL` when starting an
  already-built app does nothing; it talks to whatever it was compiled against. This cost a
  confusing `ERR_CONNECTION_REFUSED` during e2e and it will bite W9 — the deploy must build with
  the real API URL.
- **CORS is locked to a single origin** (`next_public_app_url`), so any test or preview served from
  a different port must tell the API its origin or every request fails preflight.
- **Without a Celery worker the job sits queued forever** and the page never leaves "Getting
  started". It looks exactly like a front-end bug and is not one.
- **A small upload finishes faster than Playwright can observe the "Uploading" panel.** Asserting
  on it is a race that passes on slow machines and fails on fast ones; wait for the navigation
  instead.
- **The mock manifest points at `https://cdn.local/mock/`**, so the browser logs
  `ERR_NAME_NOT_RESOLVED` for every media file on a mock project. Expected until W8, and worth
  recognising rather than chasing.
