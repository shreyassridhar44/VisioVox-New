# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **W9 — Free deploy** (next), plus the W4 remainder
- **State:** 🟢 W0–W3 and W5–W8 are done. The product works end to end on real media: upload,
  separate with the trained model, preview, download, share.
- **Branch:** `feat/phase6-playback-engine`

---

## Where things are

| Phase | State |
|---|---|
| **W0 — Storage headroom** | ✅ Done |
| **W1 — Limits, quotas, security** | ✅ Done |
| **W2 — Large upload** | ✅ Done |
| **W3 — Design system** | ✅ Done |
| **W4 — Auth UI** | 🟡 **Works, one defect left** — tokens still in `localStorage` |
| **W5 — Upload UI + engaged wait** | ✅ Done |
| **W6 — Preview, export, download** | ✅ Done |
| **W7 — Sharing** | ✅ Done |
| **W8 — Real worker** | ✅ Done — the trained checkpoint runs the pipeline |
| **W9 — Free deploy** | ⬜ **Next.** Nothing is deployed yet |

---

## What works, measured

**The real pipeline, on a real 60 s AMI meeting** (`worker_gpu`, C2-v2 checkpoint):

```
S0_ingest        1.74s     sandboxed probe + normalise
S2a_audio        8.65s     pyannote diarization
S2b_video       20.23s     insightface face tracking
S3_fuse          0.00s
S4_enrol         0.83s     ECAPA
S5_extract       1.75s     the trained SEAVE checkpoint
S7_transcribe   94.08s     <- 75% of total cost
S9_package       0.75s
                125.5 GPU-seconds, ~2.1x realtime
```

Produced a manifest, an isolated track and captions. One speaker of two was
recovered; the other had no usable enrolment cue and was dropped with a named warning — invariant 8
working as designed.

**The browser path**, driven end to end with Playwright: register → live limits → local metadata →
estimate → resumable upload → stage narration → ready → player with speaker cards.

**Exports:** the ladder reports honestly from 2160p down to 288p, produces playable faststart MP4,
and never upscales.

**Shares:** 11 security tests, including scoped manifests, immediate revocation, and no
existence oracle.

---

## Verified

```
uv run pytest -m "not gpu"       567 passed
uv run ruff check                 clean
uv run mypy apps services tests   clean, 72 source files
node scripts/check-contrast.mjs   46 pairings pass in both themes
pnpm --filter @visiovox/web build 9 routes
```

---

## Carried forward — do not lose these

- **🔴 Tokens are still in `localStorage`** (W4). Access tokens belong in memory and refresh tokens
  in an httpOnly cookie. Needs the API to set cookies and the client to send
  `credentials: 'include'`. **Should not ship to real users as it stands.**
- **`upload_peak_multiplier` is still 2.5 and unmeasured.** The pipeline now runs, so this can
  finally be measured from a real job's peak scratch usage.
- **Transcription is 75% of pipeline cost.** If processing needs to get faster, that is the only
  thing worth attacking — a smaller Whisper or a faster backend.
- **`trusted_client_ip_header` must be set when the tunnel goes up**, and not before.
- **`NEXT_PUBLIC_API_URL` is inlined at BUILD time.** W9 must build with the real API URL; setting
  it at start-up does nothing. This already cost a confusing debugging round.
- **The sandbox is hardened Docker, not gVisor** (ADR-0009 deviation, recorded in W2).
- **`D:` is still at 5.6 GB free.** ~80 GB of reclaimable slack; needs the distro stopped.
- **Only one speaker was recovered** in the real run. Worth investigating whether enrolment is too
  strict on short meeting turns before calling extraction quality good.

---

## Next actions

1. **W9** — production Compose, Cloudflare Tunnel, static landing on Pages, systemd units,
   build-time API URL, backups and a restore drill.
2. **W4 remainder** — move tokens out of `localStorage`.
3. Measure `upload_peak_multiplier` from a real job.
4. Investigate the single-speaker recovery rate on meeting audio.

---

## Standing notes

- **Do not start long GPU work without checking** `ps -eo args | grep -E "[t]rain_|[e]val_"`.
- **Never trust `df` inside the distro.** Measure the media volume.
- **Never inline `$(...)` or `$VAR` in `wsl.exe ... bash -lc`.** The outer shell evaluates them on
  the Windows side. Write a script file and run it. This cost several cycles.
- **`uv sync` alone prunes the environment.** The workspace needs
  `uv sync --all-packages --extra ml`, which is what `make install` runs. A bare `uv sync` removed
  fastapi and broke every import.
- **Commit hooks need both toolchains on PATH:** `~/.local/bin` for `uv`, and
  `~/.nvm/versions/node/v22.23.2/bin` for `pnpm`.
- **The e2e harness needs a Celery worker**, or the job sits queued and the page never leaves
  "Getting started".
- **`/srv/media` must be owned by the app user.** Root-owned, every job dies with EACCES before S0.
