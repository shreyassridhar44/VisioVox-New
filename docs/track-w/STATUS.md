# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **W3 — Design system** (next), then W5
- **State:** 🟢 W0, W1 and the W2 server side are done and verified. The backend can now accept,
  validate and admit a large upload safely. Nothing of it is visible in a browser yet.
- **Branch:** `feat/phase6-playback-engine`

---

## Where things are

| Phase | State |
|---|---|
| **W0 — Storage headroom** | ✅ **Done.** 200 GB ext4 volume at `/srv/media`, MinIO on it, disk guard live |
| **W1 — Limits, quotas, headers** | ✅ **Done.** Limiter, problem details, security headers, quotas, audit, idempotency, auth backoff |
| **W2 — Large upload** | 🟢 **Server side done.** Computed limits, reservations, resumable sessions, sandboxed probe. Browser uploader is W5 |
| **W3 — Design system** | 🟡 **Next up** |
| W4 — Auth UI | ⬜ Not started — depends on W3 |
| W5 — Upload UI + engaged wait | ⬜ Not started — depends on W2, W3 |
| W6 — Preview, export, download | ⬜ Not started |
| W7 — Sharing | ⬜ Not started |
| W8 — Real worker | ⬜ Not started |
| W9 — Free deploy | ⬜ Not started |

---

## What exists now

**The whole backend path for an upload works and is tested end to end:**
register → sign in → create project → ask for limits → init upload → upload parts →
interrupt → resume from server state → complete → job queued, with quotas charged, disk
reserved and released, and a sandboxed probe ready to validate the media.

| Area | Module |
|---|---|
| Disk headroom + admission cut-out | `apps/api/src/visiovox_api/diskspace.py` |
| Rate limiting (docs/11 §10 table) | `ratelimit.py` |
| RFC 9457 errors + correlation ids | `problems.py` |
| Security headers, correlation id, global limit | `middleware.py` |
| Quotas and usage counters | `quotas.py` |
| Audit log, salted IP hashing | `audit.py` |
| Idempotency keys | `idempotency.py` |
| Exponential auth backoff | `authguard.py` |
| Upload planning, reservations, resume | `uploads.py` |
| Confined media processing | `ml/pipeline/sandbox.py` |

---

## Next actions, in order

1. **W3 — design system.** Tailwind v4 wired to the OKLCH tokens in `docs/14` §2, the
   expressive/restrained split from D1, component primitives, and the CI contrast gate.
2. **W4 — auth UI** against the existing service layer.
3. **W5 — upload UI and the engaged wait.** This is the one that makes the product visible:
   the R3F landing, the drag-and-drop uploader against the W2 endpoints, and the SSE-driven
   processing narration.
4. **W6 — preview, export ladder, download.**
5. **W8 — real worker**, and with it the two measurements that are still placeholders.

---

## Carried forward — do not lose these

- **`upload_peak_multiplier` is a guess at 2.5.** Every computed upload limit inherits its
  honesty. Measure it in W8 when the working copy is first derived.
- **ETAs are not measured yet.** Until the pipeline is benchmarked (W8), W5 must show a range or
  nothing — never a fake countdown.
- **`trusted_client_ip_header` must be set when the tunnel goes up in W9**, and not before.
  Behind a proxy every caller shares one bucket; with no proxy the header is a spoofable identity.
- **The sandbox is hardened Docker, not gVisor** (ADR-0009 deviation, recorded in W2). Revisit
  before any genuinely public deployment.
- **`D:` is still at 5.6 GB free.** ~80 GB of dead slack in the distro vhdx; needs the distro
  stopped. The datasets are not to be deleted.
- **Storage lifecycle rule for incomplete multipart uploads** is still unwritten. Aborted uploads
  are handled in-app; an orphan from a crashed API is not.

---

## Verified

```
uv run pytest -m "not gpu"     556 passed
uv run ruff check ml tests apps  clean
uv run mypy apps tests           clean, 60 source files
make contracts                   18 paths
```

Commits on `feat/phase6-playback-engine`:
`704cf1f` `9f4e0ac` `43ebdcd` `c656357` `73a7a43` `be16700` `25d2fe8` `fd211cf` `587fc2a` `8a9ed82`

---

## Standing notes

- **Do not start long GPU work without checking** `ps -eo args | grep -E "[t]rain_|[e]val_"`.
- **Never trust `df` inside the distro.** Measure the media volume; the root reading is the vhdx's
  virtual maximum.
- **Another session may be working in this repo.** Check Docker and GPU state before assuming idle.
- **Never inline `$(...)` or `$VAR` in `wsl.exe ... bash -lc`.** The outer shell evaluates them on
  the Windows side, so `$(mktemp -d)` arrives empty and the script silently operates on `/`.
  Write a script file and run it.
- **Commit hooks need both toolchains on PATH:** `~/.local/bin` for `uv`, and
  `~/.nvm/versions/node/v22.23.2/bin` for `pnpm`, or prettier fails the commit.
