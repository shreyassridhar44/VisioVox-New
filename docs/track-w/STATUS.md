# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **W1 — Foundation: limits, quotas, headers, idempotency**
- **State:** 🟢 W0 done — media volume live at `/srv/media`, 175.8 GB usable
- **Branch:** `feat/phase6-playback-engine` (Track W work should get its own branch)

---

## Where things are

| Phase | State |
|---|---|
| **W0 — Storage headroom** | ✅ **Done.** 200 GB ext4 volume at `/srv/media`, MinIO migrated onto it, disk guard live. `D:` slack reclaim deferred |
| **W1 — Limits, quotas, headers** | 🟡 **Next up** |
| W2 — Large upload | ⬜ Not started — W0 dependency now cleared |
| W3 — Design system | ⬜ Not started |
| W4 — Auth UI | ⬜ Not started |
| W5 — Upload UI + engaged wait | ⬜ Not started |
| W6 — Preview, export, download | ⬜ Not started |
| W7 — Sharing | ⬜ Not started |
| W8 — Real worker | ⬜ Not started |
| W9 — Free deploy | ⬜ Not started |

---

## 🟢 Not blocked

W0 needed no Administrator in the end. The elevated vhdx route was prepared but proved unnecessary
once the loop-image route measured at 1.4 GB/s write / 5.2 GB/s read — the 9p boundary costs far
less than expected for large sequential objects. `infra/local/create-media-volume.ps1` is kept for
whenever the native-vhdx upgrade is wanted; nothing depends on it.

**One deferred item:** `D:` is still at 5.6 GB free. Roughly 80 GB of dead slack sits in the distro
vhdx (360 GB file, 283 GB of data). Reclaiming it needs the distro stopped, so it is scheduled
rather than done opportunistically. **The datasets are not to be deleted** — see DECISIONS.md.

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
