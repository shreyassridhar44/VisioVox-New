# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **W6 — Preview, export ladder, download** (next)
- **State:** 🟢 A complete upload→wait→preview loop now works in a real browser, on the mock
  pipeline. W0–W3 and W5 are done. W4 is partly done. W6–W9 are not started.
- **Branch:** `feat/phase6-playback-engine`

---

## Where things are

| Phase | State |
|---|---|
| **W0 — Storage headroom** | ✅ **Done.** 200 GB ext4 volume at `/srv/media`, MinIO on it, disk guard live |
| **W1 — Limits, quotas, security** | ✅ **Done.** Limiter, problem details, headers, quotas, audit, idempotency, auth backoff |
| **W2 — Large upload** | ✅ **Server done.** Computed limits, reservations, resume, sandboxed probe. Browser uploader landed in W5 |
| **W3 — Design system** | ✅ **Done.** OKLCH tokens, Tailwind v4, primitives, `/styleguide`, contrast gate in `make check` |
| **W4 — Auth UI** | 🟡 **Partly done.** Register/login/refresh work end to end. **Tokens still live in `localStorage`** — see Carried forward |
| **W5 — Upload UI + engaged wait** | ✅ **Done.** Landing with the 3D hero, resumable uploader, SSE stage narration |
| **W6 — Preview, export, download** | ⬜ **Next.** The player mounts; there is no export ladder or download yet |
| W7 — Sharing | ⬜ Not started |
| W8 — Real worker | ⬜ Not started — the pipeline still runs in mock mode |
| W9 — Free deploy | ⬜ Not started |

---

## What a person can actually do right now

Verified in a real browser against the real stack (API + Celery worker + MinIO + Postgres):

```
registered and signed in
limits shown live: Up to 50.0 GB per file · up to 60 minutes of video
read locally before upload: 286 KB · 20s · 640×480
processing estimate: about 10s–28s
upload completed, now at /projects/prj_01M2N7KSAWWRH1EHAK3X0MTY9W
processing view: Getting started
final status: ready
result: 2 speakers · overlap 7% · easy
player rendered: yes
speaker cards: 3
```

**The media in that result is mock output.** The job ran the mock pipeline, and the manifest points
at `https://cdn.local/mock/`, which is why the browser logs `ERR_NAME_NOT_RESOLVED` for the media
files. Real media needs W8.

---

## Carried forward — do not lose these

- **🔴 Tokens are still in `localStorage`.** W4 says access tokens belong in memory and refresh
  tokens in an httpOnly cookie. The current store persists both. Fixing it properly means the API
  setting cookies and the client sending `credentials: 'include'`; it was deferred rather than
  half-done. **This should not ship to real users as it stands.**
- **`upload_peak_multiplier` is a guess at 2.5.** Every computed upload limit inherits its honesty.
  Measure it in W8 when the working copy is first derived.
- **Processing ETAs are not measured.** The upload screen shows a wide range and says so; the
  waiting screen derives a widening range from observed pace. Neither is grounded in real per-stage
  timings until W8.
- **`trusted_client_ip_header` must be set when the tunnel goes up in W9**, and not before.
- **The sandbox is hardened Docker, not gVisor** (ADR-0009 deviation, recorded in W2).
- **`D:` is still at 5.6 GB free.** ~80 GB of dead slack in the distro vhdx; needs the distro
  stopped. The datasets are not to be deleted.
- **Storage lifecycle rule for incomplete multipart uploads** is still unwritten.
- **`NEXT_PUBLIC_API_URL` is inlined at BUILD time.** The deployed bundle talks to whatever it was
  compiled against, so W9 must build with the real API URL — setting it at start-up does nothing.

---

## Next actions, in order

1. **W8 before W6.** Exports need real media to mux; building the ladder against mock output cannot
   be verified end to end. Wire the real pipeline first, then export.
2. **W6** — S10 render stage, 1080p/720p ladder, `exports` table, presigned download with Range.
3. **W4 remainder** — move tokens out of `localStorage`.
4. **W9** — production Compose, Cloudflare Tunnel, static landing on Pages, build-time API URL.
5. **W7** — sharing, last, because sharing mock output is not worth the surface area.

---

## Verified

```
uv run pytest -m "not gpu"       556 passed
uv run ruff check ml tests apps   clean
uv run mypy apps tests            clean
node scripts/check-contrast.mjs   46 pairings pass in both themes
pnpm --filter @visiovox/web build 9 routes, 102 kB shared
```

---

## Standing notes

- **Do not start long GPU work without checking** `ps -eo args | grep -E "[t]rain_|[e]val_"`.
- **Never trust `df` inside the distro.** Measure the media volume.
- **Another session may be working in this repo.** Check Docker and GPU state before assuming idle.
- **Never inline `$(...)` or `$VAR` in `wsl.exe ... bash -lc`.** The outer shell evaluates them on
  the Windows side, so `$(mktemp -d)` arrives empty and the script silently operates on `/`. Write
  a script file and run it. This cost several cycles across the session.
- **Commit hooks need both toolchains on PATH:** `~/.local/bin` for `uv`, and
  `~/.nvm/versions/node/v22.23.2/bin` for `pnpm`, or prettier fails the commit.
- **The e2e harness needs a Celery worker running**, or the job sits queued and the page never
  leaves "Getting started" — which looks like a UI bug and is not one.
