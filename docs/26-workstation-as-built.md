# 26 — Workstation as built

The result of running the [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §1b
pre-flight on the actual college workstation, and the setup that followed from it.

§1b assumed a Linux host. The machine is Windows, so several checks needed translating and three
of them came back differently from what the doc anticipated. Recorded here because the differences
change operational instructions, not just setup steps.

---

## 1. What the machine actually is

| | |
|---|---|
| OS | Windows 11 Pro 22631 |
| Working environment | **WSL2, Ubuntu 24.04.3**, systemd enabled |
| GPU | NVIDIA RTX A5000, 24 GB — driver 596.51, CUDA 13.2 capable |
| CPU/RAM | 128 GB system RAM |
| Admin | The account is in `Administrators`; shells are not elevated by default |

### Storage

| Volume | Device | Media | Role |
|---|---|---|---|
| `C:` | Micron 512 GB | NVMe | Windows. **Effectively full** — keep nothing here |
| `D:` | Samsung 1 TB (disk 1) | NVMe | Free space for datasets |
| `E:` | Samsung 1 TB (disk 1) | NVMe | **WSL2 root filesystem** (`E:\wsl\Ubuntu\ext4.vhdx`) |
| `F:` | WD My Passport | **HDD over USB** | Cold storage and backups only |

---

## 2. The three findings that differed from §1b

### `~` is not network-mounted — but it was full anyway
§1b's ⭐ check is for NFS/CIFS/SMB. All four volumes are `DriveType=3` (local fixed), so that check
passes. The failure arrived by another route: `C:` had **1.3 GB free**, and every default cache
path — pip, npm, HuggingFace, Docker — lands there.

**The rule §1b states still holds, for the reason it states.** It just needs a second trigger:
*not enough local space* is as disqualifying as *wrong kind of storage*.

### `F:` looks like the answer and is not
`F:` had the most free space by far. It is a spinning USB disk. It bottlenecks the dataloader for
exactly the reason §1b rejects network homes, and it is the more dangerous case because `df` alone
does not reveal it — the media type has to be checked explicitly:

```powershell
Get-PhysicalDisk | Select-Object FriendlyName, MediaType, BusType
```

Datasets go on `D:`/`E:` (NVMe). `F:` holds the VVX backup required by R-26.

### HuggingFace is not blocked — the DNS resolver is unreliable
§1b's reachability check failed for `huggingface.co` and the §1b table maps that to "download
weights elsewhere, set `HF_HUB_OFFLINE=1`". That would have been the wrong response here.

The host resolver (`4.2.2.2`, single, no secondary) times out on A queries for that domain while
resolving `github.com`, `pypi.org` and `openslr.org` normally. Connecting by IP with SNI returns
`200`, so the host is reachable and only the name lookup fails. IPv6 is advertised by public
resolvers but is not routable here, which turns some clients' happy-eyeballs attempt into a stall.

Fixed inside WSL, which needs no Windows admin rights:

```ini
# /etc/wsl.conf
[network]
generateResolvConf=false
```
```conf
# /etc/resolv.conf  (chattr +i to survive restarts)
nameserver 1.1.1.1
nameserver 8.8.8.8
nameserver 9.9.9.9
options timeout:2 attempts:3
```

`huggingface.co` went from **0/10 to 10/10** requests. This matters most for long unattended
downloads: intermittent resolution is how a multi-hour fetch dies partway with a misleading error.

**Diagnose before believing a firewall.** `curl --resolve host:443:<ip>` separates
"name does not resolve" from "host is blocked", and the two have opposite remedies.

---

## 3. Setup decisions that followed

| Decision | Reason |
|---|---|
| **WSL2 root filesystem moved to `E:`** | `C:` could not hold it. Copied the vhdx and re-registered with `wsl --import-in-place`; the distro is named `VisioVox` |
| **Docker Engine inside WSL, not Docker Desktop** | Docker Desktop stores its disk image on `C:`. Engine in the distro puts `/var/lib/docker` inside the vhdx on `E:` and needs no elevation |
| **Datasets inside the ext4 vhdx, never `/mnt/c`** | 9p is several times slower than ext4 and bottlenecks the dataloader — the constraint already recorded in `MEMORY.md` |
| **Python 3.12 from the Ubuntu base image** | Ubuntu 24.04 ships 3.12.3, matching the repository convention exactly; no separate install needed |
| **tmux for long jobs** | Windows has no equivalent. Available in WSL, and the dataset fetch and smoke test both run under it |

### Reproducing the environment

```bash
wsl -d VisioVox                    # Ubuntu 24.04, user dmin, systemd on
cd ~/visiovox/VisioVox-New         # ext4, not /mnt/c
make install                       # uv sync --all-packages, pnpm install, pre-commit
make dev                           # Postgres, Redis, MinIO, mail
make check                         # lint, typecheck, test
make smoke                         # pretrained models on one real clip (GPU)
```

---

## 4. Verified GPU baseline

```
device       NVIDIA RTX A5000, sm_86, 24.0 GiB
bf16         supported            <- docs/25 §5 training config depends on this
bf16 matmul  8192³ x20 -> 48.4 TFLOP/s
```

Roughly 89% of the card's dense bf16 peak, measured with the desktop session running. That is the
number to compare against if throughput ever looks wrong: a large shortfall means contention or
thermal limiting, not a model problem.

Check for other users before any long run — `nvidia-smi` — as §2 of docs/25 already says. The
compute-apps list on Windows also shows desktop compositing processes; those are `C+G` entries and
are not competition for VRAM.

---

## 5. What the smoke test found

`make smoke` runs every pretrained component once on one real two-speaker mixture.
Result: **6 ok, 1 skipped, 0 failed** — the Phase 0 exit condition for the ML side.

One thing surfaced that is worth acting on before Phase 1:

> **`speechbrain/sepformer-wsj02mix` is an 8 kHz checkpoint.** Given 16 kHz input it prints
> `Resampling the audio from 16000 Hz to 8000 Hz` and separates at 8 kHz. It is a log line, not a
> warning, and the call still returns two plausible sources.

Everything else in the project is 16 kHz — `docs/06-datasets.md` §2 pins Libri2Mix to 16k. A Tier 0
baseline measured at 8 kHz would understate SI-SDR and, more importantly, would not be comparable
with the Tier 1 numbers it exists to be compared against.

**Fixed.** The smoke test now uses `speechbrain/sepformer-whamr16k` and asserts that the output
sample count matches the input to within 1%. A silent downsample halves it, so the check fails hard
instead of relying on anyone reading a log line. Output is now 74,800 samples at 16 kHz for a
74,800-sample input.

This is the smoke test doing its job: the failure it caught was silent, and it would otherwise have
shown up later as an unexplained gap between the Tier 0 and Tier 1 numbers.

---

## 6. Still outstanding

| Item | Blocks | Owner |
|---|---|---|
| **HuggingFace token + accepted pyannote licences** | S2A diarization; the smoke test skips it | Manual — accept terms, put token in `.env.local` |
| Reclaim `C:` space | Nothing yet, but leaves no headroom for Windows | `wsl --unregister Ubuntu` (the superseded distro) |
| Lab-admin questions from §1b item 8 | Whether long runs survive; reimaging policy | Ask before the first multi-day run |
