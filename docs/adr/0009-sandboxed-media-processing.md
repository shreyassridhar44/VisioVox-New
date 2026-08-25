# ADR-0009 — Sandboxed, credential-free media processing

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`15-security.md`](../15-security.md) §4, NFR-SEC-01, R-17

## Context

The application's core function is **decoding attacker-supplied binary media**. FFmpeg and its
demuxers/decoders are a large C/C++ surface with a long history of memory-safety CVEs. A malicious
upload is a plausible remote-code-execution vector, not a theoretical one.

Secondary risks: decompression bombs (a 2 MB file declaring 32000×32000 frames), complexity bombs,
SSRF via ffmpeg's network protocol handlers, and path traversal via container-embedded references.

## Options considered

### A — Run ffmpeg in the worker process
**Pros:** Simplest.
**Cons:** RCE gives the attacker the worker's full identity — cloud credentials, network access, other
users' media. Unacceptable.

### B — Separate container, standard Docker isolation
**Pros:** Namespace isolation; better than A.
**Cons:** Shared kernel; container escape from a compromised process is a known class of attack.
Credentials often still present.

### C — Hardened, credential-free sandbox with syscall isolation ✅
gVisor or Kata runtime, non-root, all capabilities dropped, seccomp profile, read-only rootfs,
**no network**, **no credentials**, memory/CPU/wall-clock limits, scratch volumes only.

### D — Rewrite media handling in a memory-safe language
**Pros:** Eliminates the vulnerability class.
**Cons:** Not feasible — no memory-safe demuxer covers the format range, and we would be
reimplementing ffmpeg.

## Decision

**Option C**, plus defence in depth around it:

1. Magic-byte verification (extension and `Content-Type` are ignored entirely)
2. `ffprobe` under the sandbox with hard caps — reject before any full decode
3. `ffmpeg` under the sandbox with `-protocol_whitelist file` and an explicit demuxer
4. Output validation before anything is trusted

Sandbox specification:
```yaml
runtime: gvisor
user: 65534:65534
capabilities: { drop: [ALL] }
readOnlyRootFilesystem: true
seccompProfile: { type: Localhost, localhostProfile: ffmpeg-restricted.json }
network: none
resources: { limits: { memory: 2Gi, cpu: 2, ephemeral-storage: 10Gi } }
timeout: 900s
env: {}                     # no credentials, ever
```

Media processing runs on **CPU workers**, never on GPU workers.

## Rationale

Two design choices carry most of the weight:

**No credentials in the sandbox.** The orchestrating worker downloads input to a scratch volume, runs
the sandboxed process against that volume, and uploads the output itself. A fully compromised ffmpeg
process obtains a scratch directory and nothing else — no bucket access, no network, no identity.
This bounds the blast radius of an RCE to "can corrupt one job's temporary files."

**No network.** ffmpeg's protocol handlers are an SSRF vector reachable through crafted container
metadata. `network: none` plus `-protocol_whitelist file` closes it at two independent layers.

Keeping this off GPU workers matters too: GPU workers are expensive, hold model weights, and are slow
to restart. The highest-risk code belongs in the cheapest, most disposable container.

`ffprobe`-first validation is what stops resource-exhaustion attacks cheaply — a decode bomb is
rejected in 5 seconds by reading metadata, without ever allocating a frame buffer.

## Consequences

**Positive** — RCE blast radius reduced to a scratch directory. SSRF closed. Resource exhaustion
bounded. Highest-risk code isolated from the most valuable compute.

**Negative** — gVisor imposes ~10–20% syscall overhead on I/O-heavy work. Additional operational
complexity (runtime class, seccomp profile maintenance). Debugging inside a sandbox is harder. Data
must be copied to and from scratch volumes.

**Neutral** — Applies identically in development via Compose, so behaviour is consistent.

## Revisit when

- gVisor overhead becomes a measured bottleneck → evaluate Kata, or a microVM per job.
- A memory-safe demuxer covering our format range becomes viable.

**Never revisit by removing the sandbox.** Any sandbox escape found in testing blocks launch
([`15-security.md`](../15-security.md) §12).
