# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **All W0–W9 complete.** Remaining work is operational, not phased.
- **State:** 🟢 The product works end to end on real media and the stack is deployable.
- **Branch:** `feat/phase6-playback-engine`

---

## Where things are

| Phase | State |
|---|---|
| W0 — Storage headroom | ✅ Done |
| W1 — Limits, quotas, security | ✅ Done |
| W2 — Large upload | ✅ Done |
| W3 — Design system | ✅ Done |
| W4 — Auth UI | ✅ Done — refresh token now in an httpOnly cookie |
| W5 — Upload UI + engaged wait | ✅ Done |
| W6 — Preview, export, download | ✅ Done |
| W7 — Sharing | ✅ Done |
| W8 — Real worker | ✅ Done — the trained checkpoint runs the pipeline |
| W9 — Free deploy | ✅ Done — stack builds, starts and serves |

---

## Verified

```
uv run pytest -m "not gpu"        567 passed
uv run ruff check                  clean
uv run mypy apps services tests    clean
node scripts/check-contrast.mjs    46 pairings, both themes
pnpm --filter @visiovox/web build  9 routes
```

**The real pipeline**, on 60 s of genuine AMI meeting audio with the C2-v2 checkpoint:
125.5 GPU-seconds, ~2.1x realtime, all eight stages, producing a manifest, an isolated track and
captions. Transcription was 94 s of that — 75% of the cost.

**The browser path**, driven with Playwright: register → live limits → local metadata → estimate →
resumable upload → stage narration → ready → player with speaker cards.

**Auth**, checked the way an XSS payload would: localStorage and sessionStorage hold nothing,
`document.cookie` is empty, the refresh cookie is httpOnly and path-scoped, and the session survives
a reload.

**Shares:** 11 security tests — scoped manifests, immediate revocation, no existence oracle.

**Deployment**, brought up for real on a scratch volume: all services healthy, migrations applied,
`/readyz` ready, HSTS present in production mode, web serving every route, build-time API URL baked
into both bundles, and nothing published to the host.

---

## Before this is used by anyone real

1. **Run the deploy.** `cp .env.production.example .env`, fill it in, `./infra/local/deploy.sh
   --tunnel`. It refuses on a missing secret, a dev default, an unmounted volume or a failed
   migration, so a clean run means a working stack.
2. **Install the GPU worker unit** — `infra/local/visiovox-worker-gpu.service`. Without it, jobs
   queue and the page never leaves "Getting started", which looks like a UI bug and is not one.
3. **Run the restore drill** — `./infra/local/backup.sh --verify`.
4. **Reclaim `D:`** — still 5.6 GB free, ~80 GB of dead slack in the distro vhdx. Needs the distro
   stopped. The datasets are not to be deleted.

---

## Known gaps, honestly

- **`upload_peak_multiplier` is still 2.5 and unmeasured.** The pipeline runs now, so it can be
  measured from a real job's peak scratch usage. Every computed upload limit inherits its honesty.
- **Only one speaker of two was recovered** in the real run; the other had no usable enrolment cue.
  That is invariant 8 behaving correctly, but it may mean enrolment is too strict on short meeting
  turns. Worth investigating before calling extraction quality good.
- **Transcription is 75% of pipeline cost.** The only thing worth attacking if this needs to be
  faster.
- **The sandbox is hardened Docker, not gVisor** (ADR-0009 deviation, recorded in W2 §Decisions).
  Revisit before a genuinely public deployment.
- **No storage lifecycle rule** for incomplete multipart uploads. Aborts are handled in-app; an
  orphan from a crashed API is not.
- **`docs/17-infrastructure-deployment.md` is superseded** by W9 for this deployment. Its threat
  model still applies; its topology does not.

---

## Standing notes

- **Do not start long GPU work without checking** `ps -eo args | grep -E "[t]rain_|[e]val_"`.
- **Never trust `df` inside the distro.** Measure the media volume; the root reading is the vhdx's
  virtual maximum.
- **Never inline `$(...)` or `$VAR` in `wsl.exe ... bash -lc`.** The outer shell evaluates them on
  the Windows side, so `$(mktemp -d)` arrives empty and the script silently operates on `/`. Write
  a script file and run it. This cost several cycles across the build.
- **`uv sync` alone prunes the environment.** Use `uv sync --all-packages --extra ml`, which is
  what `make install` runs. A bare `uv sync` removed fastapi and broke every import.
- **Commit hooks need both toolchains on PATH:** `~/.local/bin` for `uv`,
  `~/.nvm/versions/node/v22.23.2/bin` for `pnpm`.
- **`/srv/media` must be owned by the app user**, or every job dies with EACCES before S0.
- **A virtualenv is not relocatable.** Build and run at the same path, or every entrypoint fails
  with "no such file or directory" about the interpreter in its shebang.
- **`NEXT_PUBLIC_*` is inlined at BUILD time.** The deploy passes it as a build arg; setting it at
  start-up does nothing.
