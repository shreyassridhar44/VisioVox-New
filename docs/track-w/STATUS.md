# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **W0 — Storage headroom**
- **State:** 🟡 Code done and verified; volume work blocked on a decision from the project owner
- **Branch:** `feat/phase6-playback-engine` (Track W work should get its own branch)

---

## Where things are

| Phase | State |
|---|---|
| **W0 — Storage headroom** | 🟡 **Partly done.** Disk check, admission cut-out, config and 13 regression tests landed and verified. Creating the volume is blocked (see below) |
| W1 — Limits, quotas, headers | ⬜ Not started |
| W2 — Large upload | ⬜ Not started — depends on W0 |
| W3 — Design system | ⬜ Not started |
| W4 — Auth UI | ⬜ Not started |
| W5 — Upload UI + engaged wait | ⬜ Not started |
| W6 — Preview, export, download | ⬜ Not started |
| W7 — Sharing | ⬜ Not started |
| W8 — Real worker | ⬜ Not started |
| W9 — Free deploy | ⬜ Not started |

---

## 🔴 Blocked on — one elevated command

Everything that can be done without Administrator is done. To finish W0, run this **in an elevated
PowerShell** (Start → Windows Terminal → *Run as administrator*):

```powershell
cd \\wsl.localhost\VisioVox\home\dmin\visiovox\VisioVox-New\infra\local
.\create-media-volume.ps1 -RegisterLogonTask
```

It creates a 250 GB expandable ext4 vhdx at `E:\wsl\media.vhdx`, attaches it to the WSL2 VM, and
registers a logon task so the attach survives a reboot. It refuses to overwrite an existing vhdx
and is safe to re-run.

Then the rest can be finished from a normal shell.

**Why it needs elevation:** `wsl --mount` and `diskpart` both require Administrator, and the
Hyper-V PowerShell module is absent on this machine so `New-VHD` is not an option.

**Route decided:** dedicated vhdx on `E:`. **The datasets are kept** — no deleting
`Libri2Mix`, `Libri3Mix` or `voxceleb2`, so `D:` gets rescued by sparse reclaim instead.

---

## Done so far

**W0 code half — landed and verified 2026-09-16.** Independent of the route decision, because it
measures whatever `media_root` points at.

| What | Where |
|---|---|
| Headroom measurement + trustworthiness flag | `apps/api/src/visiovox_api/diskspace.py` |
| Admission cut-out (`can_admit`) and derived limit (`max_ingestible_bytes`) | same module |
| Media-volume settings | `apps/api/src/visiovox_api/config.py` |
| 13 regression tests, incl. the real 2026-09-16 failure case | `tests/test_diskspace.py` |

`pytest tests/test_diskspace.py` 13 passed · `ruff` clean · `mypy --strict` clean ·
full suite `402 passed`, no regressions.

---

## Next actions, in order

1. **Decide the W0 route** (project owner). See `W0-storage-headroom.md` §Options.
2. Create the volume, mount at `/srv/media`, repoint the MinIO Compose volume, persist across reboot.
3. Set `require_dedicated_media_volume=true` on the deployed workstation and confirm the guard fires.
4. Create a `feat/track-w` branch off `feat/phase6-playback-engine`.
5. Start W1 — rate limiting first, since everything after it widens the attack surface.

---

## Standing notes

- **Do not start long GPU work without checking** `ps -eo args | grep -E "[t]rain_|[e]val_"`.
  A second job halves throughput and pushes the A5000 to its 24 GB ceiling.
- **Never trust `df` inside the distro.** It reports the vhdx's virtual maximum (~1 TB), not real
  host free space. Check `/mnt/d` or `/mnt/e` instead. This has already caused one bug.
- **Another session may be working in this repo.** Check the Docker and GPU state before assuming
  the machine is idle — starting a second job halves throughput and can OOM the card.
