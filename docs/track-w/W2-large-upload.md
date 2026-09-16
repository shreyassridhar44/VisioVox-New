# W2 — Large upload, resumable, validated, computed limit

**State:** 🟡 Server side done and verified. The browser uploader is W5.
**Plan of record:** [`../28-product-delivery-plan.md`](../28-product-delivery-plan.md) §W2
**Depends on:** W0 ✅, W1 ✅

---

## Tasks

- [x] `GET /v1/limits` — live computed cap, duration and speaker ceilings, quota position
- [x] `max_upload_bytes` demoted to a backstop; the advertised number is computed
- [x] Admit on live headroom less in-flight reservations
- [x] **Reserve** disk at init; release on complete, abort or expiry
- [x] Stale-reservation sweep before every admission decision
- [x] Batched part-URL issuance (50 at a time) via `POST /upload/{id}/parts`
- [x] Part sizing that scales with file size, staying under the 10,000-part ceiling
- [x] `upload_sessions` + per-part state, so a refresh resumes
- [x] `GET /upload/{id}` — what a resuming client needs
- [x] `POST /upload/{id}/abort` — tears down the multipart upload at the store too
- [x] Refusals that name the number and the remedy
- [x] `/upload/init` rate limit wired (the W1 leftover)
- [x] Quota charged at init, not completion
- [x] **Sandboxed probe** — `ml/pipeline/sandbox.py`, hardened container, verified
- [x] Magic-byte sniffing; filename and `Content-Type` ignored entirely
- [x] Reject the malformed-media class: no audio, zero duration, over-length, stream and pixel bombs
- [ ] Pre-flight estimate in the browser (client metadata → upload/processing ETA) — **W5**
- [ ] Client uploader: slicing, 4-way concurrency, retry, pause/resume — **W5**
- [ ] Derive the compact working copy, and **measure `upload_peak_multiplier`** — **W8**
- [ ] Lifecycle rule for incomplete multipart uploads on the storage side

---

## Decisions made while building

- **2026-09-16 — the limit is computed per request, not configured.** `max_upload_bytes` is now a
  50 GB backstop against one absurd upload; the number shown comes from live free space less
  in-flight reservations, divided by the peak multiplier. A cap that ignores the disk is a promise
  the machine cannot keep.
- **2026-09-16 — space is reserved, not counted.** Rate-limiting `/upload/init` does not help:
  every init creates a real multipart upload that occupies storage whether or not it completes.
  Ten concurrent 30 GB uploads must not all be admitted because each individually fits.
- **2026-09-16 — the stale sweep runs before each admission decision**, so an upload abandoned
  yesterday cannot refuse a live one today.
- **2026-09-16 — quota is charged at init, not completion.** An upload that is started and walked
  away from still costs storage and a slot.
- **2026-09-16 — part size scales, part count does not.** At a fixed 16 MiB the 10,000-part ceiling
  arrives at 160 GB. Scaling the part size keeps very large files legal.
- **2026-09-16 — one endpoint both records finished parts and issues the next batch.** One round
  trip per batch instead of two, and progress becomes durable at batch granularity.
- **2026-09-16 — the server takes the greater of the client's `after` and its own record**, so a
  confused client cannot skip parts by asking for a later batch.
- **2026-09-16 — ⚠️ deviation from ADR-0009: hardened Docker, not gVisor.** Neither `runsc` nor
  Kata is available on the target (WSL2, runc only), and the zero-cost constraint rules out a
  separate hardened host. Implemented the ADR's Option B with **every** control the platform does
  offer, and `build_argv(runtime=...)` makes the gVisor upgrade a one-line change.
  **What is lost:** a container escape through a kernel bug no longer meets a second syscall
  barrier. **What is kept** is the control the ADR itself calls decisive — the sandbox holds *no
  credentials* and has *no network*, so a fully compromised ffmpeg gets a scratch directory and
  nothing else. This deviation should be revisited before any genuinely public deployment.
- **2026-09-16 — the sandbox image is built locally, not pulled.** A security-critical container
  should not be a third-party image that can change underneath us, and the stack keeps working with
  no registry access.
- **2026-09-16 — magic bytes decide the format; the extension and `Content-Type` never do.** Both
  are attacker-chosen. Refusing unrecognised files also keeps ffmpeg's rarest, least-audited
  demuxers out of reach of strangers.
- **2026-09-16 — `ffprobe` stderr is never surfaced verbatim.** It can echo container metadata,
  which is attacker-controlled.

---

## Measurements

### Sandbox confinement — verified 2026-09-16, not merely configured

| Property | Result |
|---|---|
| Probe of a real clip | `duration=3.00s`, `h264 + aac`, `320x240` ✅ |
| Network | `Failed to resolve hostname example.com` ✅ |
| Root filesystem | `can't create /etc/passwd: Read-only file system` ✅ |
| Bind mount | `can't create /work/evil: Permission denied` ✅ |
| User | `uid=65534(nobody) gid=65534(nobody)` ✅ |
| Environment | `HOME`, `HOSTNAME`, `PATH` only — no credentials ✅ |
| Fork bomb | `can't fork: Resource temporarily unavailable` ✅ |
| Image size | 189 MB |

### Upload path end to end — 2026-09-16, real MinIO

```
limits: max_upload=50.0 GB  available=175.8 GB  reserved=0.0 GB
init -> 200   plan: 2 parts of 16 MiB, 2 URLs issued
reserved after init: 0.04 GB          (18 MiB x 2.5 multiplier)
part 1 PUT -> 200
resume: status=active completed=[1] of 2
next batch -> parts [2]
part 2 PUT -> 200
complete -> 202  job queued
reserved after complete: 0.00 GB
uploads quota now: {'used': 1, 'limit': 20}
900 GB init -> 413: "This file is 900.0 GB and the current limit is 50.0 GB..."
```

**Verification:** `uv run pytest -m "not gpu"` → **556 passed**; `ruff` and `mypy --strict` clean.

---

## Gotchas

- **🔥 The sandbox runs as `nobody` but scratch files belong to the worker's user**, so without an
  explicit `chmod` the container cannot read its own input. It fails as a bare "Permission denied"
  from inside a container, which is a genuinely confusing thing to debug. `prepare_workdir()` now
  handles it.
- **The 50 GB ceiling binds before the disk does on this machine.** 175.8 GB usable ÷ 2.5 = 70 GB
  affordable, so the backstop is what a user sees. A test asserting "reservations lower the limit"
  must raise the ceiling first or it silently compares two identical numbers.
- **`--memory-swap` must equal `--memory`.** Without it the container swaps indefinitely and the
  memory ceiling stops bounding a decompression bomb.
- **`ml/` is a source root**: imports are `from pipeline.x`, not `from ml.pipeline.x`.
- **`ruff`'s `noqa` must sit on the line it reports**, which for `S607` is the argv list, not the
  `subprocess.run(` line above it.
- **Starlette's `HTTP_413_REQUEST_ENTITY_TOO_LARGE` and `HTTP_422_UNPROCESSABLE_ENTITY` are
  deprecated** and renamed. Literals avoid coupling to the installed version's spelling.
- **Never inline `$(...)` or `$VAR` in a `wsl.exe ... bash -lc` command.** The outer shell evaluates
  them on the Windows side, so `$(mktemp -d)` arrives empty and the script silently operates on `/`.
  Write a script file and run it.
