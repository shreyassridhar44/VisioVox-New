# Track W — decisions and standing rules

> **Read this before writing any Track W code.** It is the short answer to "why is it like this"
> and "what am I not allowed to change without asking".
>
> Part 1 is the **decision register** — what was decided, and what it means in practice.
> Part 2 is the **standing rules** that follow from those decisions.
> Part 3 is the **build log** — decisions taken *during* implementation, appended as they happen.
>
> Rationale lives in [`../28-product-delivery-plan.md`](../28-product-delivery-plan.md) §3.
> This file is the operative summary; that file is the argument.

---

## Part 1 — Decision register

| # | Decision | Status | What it means when you are writing code |
|---|---|---|---|
| **D1** | **Split the design direction.** Expressive 3D on landing/auth/upload/**waiting**; restrained studio instrument in the player/transcript/export | ✅ Settled | Before adding a visual flourish, ask which surface you are on. Glow in the player is a bug, not a style choice |
| **D2** | **No fixed upload cap.** The limit is computed from live disk, queue and measured throughput, and displayed to the user up front | ✅ Settled | Never write a size constant into the UI or a config as *the* limit. The 50 GB ceiling is a backstop, not the number users see |
| **D3** | **Sharing is Web Share API + `wa.me` + share links.** Direct posting to Instagram is impossible from a web app | ✅ Settled | Do not build a button that implies direct posting. Label what each control actually does |
| **D4** | **Postgres is the ledger, Redis is disposable.** Quotas, shares, exports, audit and upload sessions are tables | ✅ Settled | If losing Redis would lose correctness rather than speed, it is in the wrong store |
| **D5** | **Zero-cost self-hosted deployment.** The workstation is the server, exposed via a free tunnel. Nothing rented, no card | ✅ Settled | Never introduce a paid or card-required service. `docs/17` (managed + Kubernetes) is **aspirational and must not be followed literally** |
| **D6.1** | Tiering — plans may be unnecessary without billing | ⏳ Open | Do not wire plan-based branching until decided |
| **D6.2** | Retention — 30 days default; 7 recommended on a fixed disk | ⏳ Open | Keep retention a config value, not a literal |
| **D6.3** | Public signup vs invite-only | ⏳ Open | Keep the registration path swappable |

---

## Part 2 — Standing rules

### Inherited — these already bind, do not restate or weaken them
[`../../CLAUDE.md`](../../CLAUDE.md) holds the **8 project invariants** (identical sample counts,
integer-millisecond timing, ephemeral embeddings, server-side ownership checks, sandboxed ffmpeg,
Faithful-track transcription, idempotent stages, partial results) and the Python/TypeScript/ML
conventions. Track W does not get exemptions from any of them.

Two are load-bearing for this track specifically:
- **Invariant 4 — every artifact access is ownership-checked server-side.** IDOR is the highest-impact
  vulnerability in this product. Share links (W7) are the obvious place to get this wrong.
- **Invariant 5 — ffmpeg never runs outside the sandbox, never with credentials.** The upload probe
  (W2) and the export render (W6) are both ffmpeg on user-supplied bytes.

### Track W additions

**Money**
- No paid services, no managed databases, no rented GPU, no "free tier that later bills". If a
  hosted thing is genuinely needed, find the no-card option or self-host it. (D5)

**Disk**
- **Never call `df /` or `statvfs("/")` to decide anything.** Inside this distro it reports the
  vhdx's virtual ceiling, not real free space. Measure the configured media volume. (W0)
- Any code path that writes media must reserve headroom first and release it on
  completion/abort/timeout.

**Limits**
- Admit jobs on **probed duration × speaker count**, never on client-declared metadata, never on
  bytes alone. (D2)
- Every user-facing limit is computed and explained. "Too large" without a reason or a remedy is
  not an acceptable error message.

**Estimates**
- **An ETA that was not measured is not shipped.** Until the pipeline is benchmarked (W8), show a
  range or show nothing. Never a fake countdown, never a bar that parks at 99%. (D2)

**Truthfulness in the UI**
- The interface does not claim capabilities the product lacks — no fake Instagram posting (D3), no
  invented confidence, no progress that is not real progress. This is the same principle as
  `docs/14` §1.3 ("uncertainty is visible") applied to the product surface.

**Process**
- A task is ticked in a phase file only when its **verification command has been run and passed**.
- Every measurement is recorded with its date and the command that produced it.
- If implementation contradicts the plan, **fix `28-product-delivery-plan.md` in the same commit**.
  Two documents disagreeing is worse than one being wrong.

---

## Part 3 — Build log

Decisions taken during implementation that Part 1 did not anticipate. Append; do not rewrite.

### 2026-09-16 — W0 · storage route

- **Media lives on a dedicated ext4 vhdx on `E:`, mounted at `/srv/media`.** Native ext4 on NVMe,
  full speed. Costs a one-time elevated setup and a logon task, because `wsl --mount` needs
  Administrator and does not survive a reboot.
- **🚫 The datasets stay. Do not delete `~/data/Libri2Mix`, `~/data/Libri3Mix` or
  `~/data/voxceleb2`.** More training is planned (the `c4` attempt of 2026-09-13 may be retried),
  and VoxCeleb2 may be unrecoverable because its credentials are still outstanding. This removes
  160 GB of easy reclaim from the table — deliberately. **Find space elsewhere; do not revisit
  this.**
- **`D:` is therefore rescued without deleting anything**, via `fstrim` plus
  `wsl --manage VisioVox --set-sparse true`. The vhdx file is 360.6 GB while the filesystem holds
  281 GB, so roughly 80 GB is dead slack that a dynamic vhdx never returns on its own. This needs
  the distro stopped, so it is scheduled rather than done opportunistically.

### 2026-09-16 — W0 · implementation
- **The headroom check measures the configured media path, not `/`.** `statvfs("/")` reports the
  vhdx's virtual ceiling here and is useless. The check resolves the configured media directory and
  **refuses to run if that directory shares a filesystem with `/`**, because that silently
  reintroduces the original bug rather than failing loudly.
- **`docs/17-infrastructure-deployment.md` is now aspirational.** It predates the zero-cost
  constraint and specifies managed services and Kubernetes. W9 supersedes it. Flagged in the
  document itself rather than deleted, because its threat model and topology reasoning are still
  sound.
