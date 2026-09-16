# W0 — Storage headroom

**State:** 🟡 In progress — audited, blocked on a route decision
**Plan of record:** [`../28-product-delivery-plan.md`](../28-product-delivery-plan.md) §W0

---

## Why this phase exists

The upload feature cannot be built onto a volume with 5.6 GB free, and the failure mode is nasty:
not a clear "disk full" error, but a confusing `ENOSPC` from inside ffmpeg partway through a job,
after the user has already waited through an upload.

---

## Measurements

### Host drives — 2026-09-16
`Get-PSDrive -PSProvider FileSystem`

| Drive | Size | Free | Notes |
|---|---|---|---|
| `C:` | 477 G | **20.6 G** | Chronically tight. Nothing project-related goes here |
| `D:` | 500 G | **5.6 G** | ⚠️ Hosts the distro vhdx |
| `E:` | 454 G | **334.4 G** | NVMe, mostly empty. The only place with room |

### What is on `D:` — 2026-09-16

| Item | GB |
|---|---|
| `D:\wsl\VisioVox\ext4.vhdx` | **360.6** |
| `final-mistral-merged` | 13.5 |
| `ollama` | 13.5 |
| `GAN` | 10.4 |
| `New folder` | 8.6 |
| `vbooks` | 5.2 |
| others (`Prarthana`, `textbook_pipeline`, …) | ~4.5 |
| unaccounted (hidden / system / recycle bin) | ~78 |

The vhdx is **not** a VisioVox-only cost — but it is 72% of the drive.

### Inside the distro — 2026-09-16
`du -x -h -d1 ~` and `du -x -h -d2 ~/data`

| Path | Size |
|---|---|
| `/home/dmin` | **277 G** |
| ├─ `~/data` | **247 G** ← the whole problem |
| │  ├─ `data/Libri2Mix` | 91 G |
| │  ├─ `data/voxceleb2` | 76 G (55 packed + 11 extracted) |
| │  ├─ `data/Libri3Mix` | 69 G |
| │  ├─ `data/corpora/wham_noise` | 11 G |
| │  ├─ `data/ami` | 909 M |
| │  └─ `data/testvideos` | 937 M |
| ├─ `~/visiovox` | 12 G |
| ├─ `~/.cache` | 9.5 G |
| ├─ `~/mistral-venv` | 6.7 G |
| └─ `~/runs` | 1.5 G |

**vhdx slack:** the file is 360.6 G but the filesystem holds 281 G — roughly **80 G already
reclaimable** without deleting anything, because a dynamic vhdx never shrinks on its own.

### Environment constraints — 2026-09-16

| Fact | Consequence |
|---|---|
| The shell these commands run in is **not elevated** | `wsl --mount --vhd` and `diskpart` unavailable — both need Administrator |
| **Hyper-V module absent** (`New-VHD` missing) | Cannot create or compact a vhdx from PowerShell |
| WSL 2.7.14.0, kernel 6.18 | Modern enough for `--set-sparse` and `--manage --move` |
| `sudo` works inside the distro | Loop-device mounts *are* available without Windows elevation |

---

## Options

Every route either deletes data, moves the distro, or needs one elevated shell. None is a
unilateral call.

### Option A — Reclaim inside the distro *(no admin; do this regardless)*
Delete datasets that training no longer needs, then let the vhdx give space back:
```bash
rm -rf ~/data/Libri2Mix ~/data/Libri3Mix          # 160 G, only if not needed again
sudo fstrim -av
wsl.exe --manage VisioVox --set-sparse true        # returns freed blocks to D:
```
- **Frees:** up to 160 G, plus ~80 G of existing slack
- **Risk:** LibriMix is *regenerable* (`ml/training/librimix_data.py`) but costs hours and needs
  the source corpora. **Do not delete `voxceleb2`** — per project notes the credentials are still
  outstanding, so it may be unrecoverable.
- **Verdict:** `D:` at 99% is a risk to Windows itself, not just to this project. Worth doing on
  its own merits — but it does not by itself put media on the fast, roomy drive.

### Option B — Move the whole distro to `E:` *(no admin)*
```powershell
wsl --manage VisioVox --move E:\wsl\VisioVox
```
- **Blocked as things stand:** the vhdx is 360.6 G and `E:` has 334.4 G free. It does not fit.
- Would only work after Option A shrinks it, and then `E:` is nearly full too — so it solves the
  `D:` problem and leaves no room for media. **Not recommended alone.**

### Option C — Loop-mounted ext4 image on `E:` *(no admin, reversible)* ⭐ fallback
Create a large sparse file on `E:` through drvfs, format it ext4, mount it at `/srv/media`:
```bash
sudo mkdir -p /srv/media
truncate -s 250G /mnt/e/wsl/media.img
sudo mkfs.ext4 -m 0 /mnt/e/wsl/media.img
sudo mount -o loop /mnt/e/wsl/media.img /srv/media
```
- **Pros:** no elevation; fully reversible (delete the file); real ext4 semantics, which MinIO
  needs and which raw drvfs cannot give it
- **Cons:** I/O crosses the 9p boundary. Fine for MinIO's large sequential objects, poor for many
  small files. Needs a boot-time remount hook
- **Gives:** ~250 G for media

### Option D — Proper vhdx on `E:` *(needs two elevated commands)* ⭐ recommended
```powershell
# elevated PowerShell, once
diskpart /s create-media-vhd.txt      # create vdisk file=E:\wsl\media.vhdx maximum=256000 type=expandable
wsl --mount --vhd E:\wsl\media.vhdx --bare
```
then inside the distro, once:
```bash
sudo mkfs.ext4 /dev/sdX && sudo mkdir -p /srv/media && sudo mount /dev/sdX /srv/media
```
- **Pros:** native ext4 on NVMe at full speed. The correct long-term answer
- **Cons:** needs Administrator; `wsl --mount` does not persist across reboots, so it needs a
  scheduled task at logon (also a one-time elevated setup)
- **Gives:** ~250 G for media at full disk speed

### Recommendation
**Option A + Option D.** A because `D:` at 5.6 G free endangers the whole machine; D because media
belongs on fast native ext4 and the elevation is a one-time cost. **Option C if elevation is not
available at all** — it is genuinely workable, just slower.

---

## Tasks

- [x] Audit real free space on host drives, not `df` inside the distro
- [x] Identify what is consuming `D:` and the distro vhdx
- [x] Establish which operations are available without elevation
- [x] **DECIDE the route** — dedicated ext4 vhdx on `E:`, mounted at `/srv/media`.
      **Datasets are kept**, so the 160 GB reclaim is off the table; `D:` gets rescued by sparse
      reclaim instead (see DECISIONS.md)
- [x] Write the setup scripts — `infra/local/create-media-volume.ps1` (elevated),
      `infra/local/setup-media-volume.sh` (idempotent, verifies the mount is a distinct filesystem)
- [x] Make MinIO's storage location configurable — `MEDIA_MINIO_PATH` in `infra/docker/compose.yaml`,
      defaulting to the existing named volume so nothing breaks until the volume exists
- [x] Document the new settings in `.env.example`
- [ ] ⏳ **Run `create-media-volume.ps1` elevated** ← needs the project owner; cannot be done from here
- [ ] Run `setup-media-volume.sh` and confirm `/srv/media` is a distinct filesystem
- [ ] Set `MEDIA_MINIO_PATH` and restart the stack; migrate any existing bucket contents
- [ ] Register the logon task so the mount survives a reboot
- [ ] Reclaim `D:` slack: `fstrim -av`, then `wsl --manage VisioVox --set-sparse true` (needs the distro stopped)
- [x] Disk-headroom check that reads the **media volume**, never `df /` — `apps/api/src/visiovox_api/diskspace.py`
- [x] Admission cut-out below a configured headroom floor — `can_admit()` in the same module
- [x] Regression test so the wrong disk check cannot come back — `tests/test_diskspace.py`, 13 tests
- [x] Config settings for the media volume — `media_root`, `disk_reserved_bytes`, `upload_peak_multiplier`, `require_dedicated_media_volume`

**Verified 2026-09-16:**
```
uv run pytest tests/test_diskspace.py -q     13 passed
uv run ruff check / format --check           clean
uv run mypy --strict                         no issues
uv run pytest -q -m "not gpu"                402 passed (no regressions)
```

The code half of W0 is done and does not depend on the route decision — it measures whatever
`media_root` points at. The remaining tasks all need the volume to exist first.

---

## Decisions made while building

- **2026-09-16 — the headroom check measures the configured media path, not `/`.** Inside this
  distro `statvfs("/")` reports the vhdx's virtual ceiling (~1 TB) and is therefore useless. The
  check resolves the configured media directory and refuses to run if that directory is on the same
  filesystem as `/`, because that silently reintroduces the bug.
- **2026-09-16 — the untrustworthy-reading guard is configuration, not an absolute.** Local
  development and CI legitimately have no dedicated volume, so
  `require_dedicated_media_volume` defaults to **off** and the deployed workstation turns it **on**.
  Making it unconditional would have meant either a failing test suite or a check weak enough to
  pass anywhere, and the second is how the original bug survived.
- **2026-09-16 — a missing `media_root` raises instead of reporting zero space.** An unmounted
  volume is a deployment fault; reporting it as "disk full" sends the operator to the wrong problem
  entirely. This matters more than usual here because the mount does **not** survive a reboot.
- **2026-09-16 — `max_upload_bytes` was repurposed, not replaced.** It is now a 50 GB backstop
  against a single absurd upload (was 2 GB, used as *the* limit). The user-facing number is computed
  from live headroom by `max_ingestible_bytes()`. Wiring it into `upload_init` belongs to W2.
- **2026-09-16 — `upload_peak_multiplier` is a placeholder at 2.5.** It encodes "a job holds source
  + working copy + outputs at once". The real figure must be **measured** in W2; the derived limit
  is only as honest as this constant.

---

## Gotchas

- **`df` inside the distro lies.** It reports the vhdx's virtual maximum, not real host free space.
  It will happily report 675 G available on a drive with 5.6 G left. Already caused one bug before
  this phase; the guard in `diskspace.py` exists specifically to stop it recurring.
- **A dynamic vhdx never shrinks by itself.** Deleting files inside the distro frees nothing on
  `D:` until `fstrim` + `--set-sparse true`.
- **`wsl --mount` needs Administrator and does not survive a reboot.** Plan for a logon task, or
  the media volume silently vanishes and MinIO starts writing to an empty directory.
- **Docker named volumes live on the distro vhdx.** Moving MinIO's storage means changing the
  Compose volume to a bind mount; simply mounting `/srv/media` is not enough.
- **⚠️ Setting `MEDIA_MINIO_PATH` before the volume is mounted is silently wrong.** Docker creates
  a missing bind-mount source as an ordinary directory — so MinIO starts, works, and writes to the
  **root vhdx** while every sign says it is on the media volume. Mount first, then set the variable.
  The API-side guard (`require_dedicated_media_volume`) catches this; Docker will not.
- **Device letters move.** The attached vhdx can appear as `/dev/sdc` or `/dev/sdd` depending on
  attach order, which is why `setup-media-volume.sh` writes fstab by **UUID**. `nofail` is equally
  load-bearing: without it, a boot where the vhdx was not attached drops the distro into an
  emergency shell rather than starting without the volume.
- **`mkfs.ext4 -m 0` is deliberate.** The default 5% root reserve would be ~12 GB of unusable space
  on a 250 GB media volume. Headroom is managed explicitly by `disk_reserved_bytes`, where it is
  visible and tunable, rather than hidden in the filesystem.
