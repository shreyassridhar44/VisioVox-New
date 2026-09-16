# W0 — Storage headroom

**State:** ✅ Volume built and verified. One deferred item: reclaiming `D:` slack.
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

### The volume as built — 2026-09-16

| | |
|---|---|
| Backing file | `/mnt/e/wsl/visiovox-media.img`, 200 G, ext4, **not sparse** — NTFS commits the space immediately |
| Mount | `/srv/media`, via `srv-media.mount` (systemd-fstab-generator) |
| Capacity | 195.8 G total, **175.8 G usable** after the 20 G reserved floor |
| `st_dev` | media `1793`, root `2096` — distinct, so the guard reads real numbers |
| Max single upload | **50 G** — capped by the ceiling, not the disk (175.8 / 2.5 = 70 G available) |
| `E:` after | 133 G free |

**Throughput, measured with `dd conv=fdatasync` and caches dropped:**

| | |
|---|---|
| Write | **1.4 GB/s** |
| Read | **5.2 GB/s** |

This is the number that settled the route. The 9p boundary was expected to be the objection to the
loop-image approach; for the large sequential objects MinIO deals in it costs far less than feared,
and the elevated vhdx setup was not worth requiring.

### Environment constraints — 2026-09-16

| Fact | Consequence |
|---|---|
| The shell these commands run in is **not elevated** | `wsl --mount --vhd` and `diskpart` unavailable — both need Administrator |
| **Hyper-V module absent** (`New-VHD` missing) | Cannot create or compact a vhdx from PowerShell |
| WSL 2.7.14.0, kernel 6.18 | Modern enough for `--set-sparse` and `--manage --move` |
| `sudo` works inside the distro | Loop-device mounts *are* available without Windows elevation |

---

## Options considered

Kept as the record of why the built volume looks the way it does. **Option C was taken** — see
§"The volume as built". Options A, B and D were not.

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

### Option C — Loop-mounted ext4 image on `E:` *(no admin, reversible)* ✅ **CHOSEN**
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
- **Measured after building it:** 1.4 GB/s write, 5.2 GB/s read — the 9p objection largely did not
  materialise, which is why this beat Option D in practice

### Option D — Proper vhdx on `E:` *(needs two elevated commands)* — prepared, not used
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

### Outcome
**Option C**, on measurement rather than preference. The case for D was that native ext4 would be
meaningfully faster; at 1.4 GB/s write the loop image is not the bottleneck for anything this
product does, and C needs no Administrator, no logon task, and is reversible by deleting one file.
`infra/local/create-media-volume.ps1` is kept so the D upgrade stays available; nothing depends on it.

**Option A was rejected by the project owner** — the datasets stay (see DECISIONS.md). `D:` is
therefore still at 5.6 G free and needs the sparse reclaim, which is deferred.

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
- [x] **Create the volume without Administrator** — loop-mounted ext4 image at
      `/mnt/e/wsl/visiovox-media.img`, 200 G, mounted at `/srv/media`. The elevated vhdx route was
      not needed after the loop route measured well (see Measurements)
- [x] Mount **via systemd**, so it lands in PID 1's namespace and Docker can see it
- [x] Confirm `/srv/media` is a distinct filesystem — `st_dev` 1793 vs root 2096
- [x] Boot persistence — `srv-media.mount` generated, `WantedBy=local-fs.target`, `Requires=mnt-e.mount`
- [x] Migrate the existing MinIO bucket (5.3 MB) off the named volume, no data loss
- [x] Repoint MinIO — verified `bind /srv/media/minio -> /data`, container healthy
- [x] Fix Compose env loading — `--project-directory .` in the Makefile, so the repo-root `.env` is read
- [x] Silence WSL's competing `mount -a` — `mountFsTab = false` in `/etc/wsl.conf`
- [ ] Reclaim `D:` slack: `fstrim -av`, then `wsl --manage VisioVox --set-sparse true` (needs the
      distro stopped, so scheduled rather than done opportunistically — `D:` is still at 5.6 GB)
- [ ] Verify the mount returns after a real reboot (the `wsl.conf` change only takes effect then)
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
- **`mkfs.ext4 -m 0` is deliberate.** The default 5% root reserve would be ~10 GB of unusable space
  on a 200 GB media volume. Headroom is managed explicitly by `disk_reserved_bytes`, where it is
  visible and tunable, rather than hidden in the filesystem.
- **🔥 Every `wsl.exe … bash -c` invocation gets its OWN mount namespace.** A `sudo mount` made that
  way succeeds, prints nothing wrong, and is **invisible to the next command, to Docker and to the
  API**. It cost a confusing round of debugging where `df` showed the volume mounted and `stat`
  insisted it was on `/`. Mount through **systemd** (`systemctl start srv-media.mount`), which runs
  in PID 1's namespace, and verify with `nsenter -t 1 -m -- …`.
- **Compose ignores the repo-root `.env` when invoked as `-f infra/docker/compose.yaml`.** The
  project directory becomes `infra/docker/`, so `MEDIA_MINIO_PATH` was silently unset and MinIO kept
  writing to the named volume — while every other sign said the cutover had worked. Fixed with
  `--project-directory .` in the Makefile. Safe because `name: visiovox` is pinned in the compose
  file, so the project is not renamed and its volumes are not orphaned.
- **`truncate` on drvfs is not sparse.** A 2 G test file consumed a real 2 G. So the image reserves
  its full size up front — which is arguably right here, since capacity is committed rather than
  discovered to be missing halfway through a job.
- **WSL runs its own `mount -a` before `/mnt/e` exists**, which failed noisily on every single
  invocation. `mountFsTab = false` in `/etc/wsl.conf` hands `/etc/fstab` to systemd, which orders it
  correctly behind `mnt-e.mount`. Takes effect at the next distro restart.
