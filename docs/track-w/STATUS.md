# Track W — STATUS

> **Update this at the end of every working session.** It is the resume pointer.

- **Last updated:** 2026-09-16
- **Current phase:** **W1 → W2**
- **State:** 🟢 W0 done. W1 substantially done — limiter, problem details, security headers,
  quotas, usage counters and audit log all landed and verified.
- **Branch:** `feat/phase6-playback-engine` (Track W work should get its own branch)

---

## Where things are

| Phase | State |
|---|---|
| **W0 — Storage headroom** | ✅ **Done.** 200 GB ext4 volume at `/srv/media`, MinIO migrated onto it, disk guard live. `D:` slack reclaim deferred |
| **W1 — Limits, quotas, headers** | 🟡 **Mostly done.** Limiter, problem details, security headers, quotas, audit log ✅. Idempotency and auth backoff remain |
| **W2 — Large upload** | 🟡 **Next up.** The `upload_sessions` table is already in place |
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
once the loop-image route measured at 1.4 GB/s write and 5.2 GB/s read — the 9p boundary costs far
less than expected for large sequential objects. `infra/local/create-media-volume.ps1` is kept for
whenever the native-vhdx upgrade is wanted; nothing depends on it.

**One deferred item:** `D:` is still at 5.6 GB free. Roughly 80 GB of dead slack sits in the distro
vhdx (360 GB file, 283 GB of data). Reclaiming it needs the distro stopped, so it is scheduled
rather than done opportunistically. **The datasets are not to be deleted** — see DECISIONS.md.

---

## Done so far

### W0 — storage headroom ✅

| What | Where |
|---|---|
| Headroom measurement + trustworthiness flag | `apps/api/src/visiovox_api/diskspace.py` |
| Admission cut-out (`can_admit`), derived limit (`max_ingestible_bytes`) | same module |
| Media-volume settings | `apps/api/src/visiovox_api/config.py` |
| Volume provisioning scripts | `infra/local/` |
| 13 regression tests, incl. the real 2026-09-16 failure case | `tests/test_diskspace.py` |

Volume: 195.8 GB total, **175.8 GB usable**, `st_dev` distinct from `/`, remounted at boot by
`srv-media.mount`. MinIO verified as `bind /srv/media/minio -> /data`.

### W1 — foundation 🟡

| What | Where |
|---|---|
| Sliding-window Redis limiter, the docs/11 §10 table | `apps/api/src/visiovox_api/ratelimit.py` |
| RFC 9457 problem details + correlation ids | `apps/api/src/visiovox_api/problems.py` |
| Security headers, correlation id, global per-IP limit | `apps/api/src/visiovox_api/middleware.py` |
| Shared async Redis pool | `apps/api/src/visiovox_api/redis_client.py` |
| Quotas: uploads/day, media-seconds and GPU-seconds/month, concurrency | `apps/api/src/visiovox_api/quotas.py` |
| Audit log with salted IP hashing | `apps/api/src/visiovox_api/audit.py` |
| `usage_counters`, `audit_events`, `upload_sessions` | migration `612347771b6e`, applied |
| 65 tests | `tests/test_ratelimit.py`, `test_api_hardening.py`, `test_quotas.py` |

---

## Next actions, in order

1. **Finish W1** — `Idempotency-Key` on creating POSTs, and exponential backoff on repeated auth
   failure keyed on the account.
2. **W2** — `GET /v1/limits`, batched part URLs, resumable upload sessions, the sandboxed probe.
3. **Measure `upload_peak_multiplier` during W2** and replace the 2.5 placeholder. Every computed
   upload limit is only as honest as that constant.
4. **Set `trusted_client_ip_header`** when the tunnel goes up in W9 — and not before, because a
   header nothing overwrites is a spoofable identity.
5. **Reclaim `D:` slack** (deferred from W0) — needs the distro stopped.

---

## Verified so far

```
uv run pytest -m "not gpu"     467 passed
uv run ruff check apps tests   clean
uv run mypy apps tests         clean, 54 source files
```

Commits on `feat/phase6-playback-engine`:
`704cf1f` `9f4e0ac` `43ebdcd` `c656357` `73a7a43` `be16700`

---

## Standing notes

- **Do not start long GPU work without checking** `ps -eo args | grep -E "[t]rain_|[e]val_"`.
  A second job halves throughput and pushes the A5000 to its 24 GB ceiling.
- **Never trust `df` inside the distro.** It reports the vhdx's virtual maximum (~1 TB), not real
  host free space. Measure the media volume instead. This has already caused one bug.
- **Another session may be working in this repo.** Check the Docker and GPU state before assuming
  the machine is idle.
- **Do not pipe heredocs through `wsl.exe ... bash -lc`.** The quoting layers strip backticks and
  `$`, and it will silently mangle a file it appears to have written. Write a script file and run it.
