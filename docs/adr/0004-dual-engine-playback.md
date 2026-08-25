# ADR-0004 — Dual-engine playback: Web Audio primary, HLS for long media

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`02-approach-review.md`](../02-approach-review.md) §F2, [`12-media-pipeline-and-sync.md`](../12-media-pipeline-and-sync.md), FR-PLAY-03/05/06, NFR-PERF-03

## Context

The player must switch between N per-speaker audio tracks with ≤ 120 ms latency, no audible seam,
and ≤ 40 ms A/V drift over 10 minutes — including after seeking, rate changes and tab backgrounding.

The archived roadmap proposed N `<audio>` elements synchronised by assigning
`audio.currentTime = video.currentTime`.

Three facts constrain the design:
- Assigning `currentTime` issues an **asynchronous seek** (10–150 ms), landing on the nearest
  decodable point rather than the requested one.
- Independent media elements run **independent decoder clocks** and drift relative to one another.
- Lip-sync detection threshold is roughly **+45 ms / −125 ms** (ITU-R BT.1359), so drift is
  perceptible well before it looks broken.

## Options considered

### A — N `<audio>` elements, `currentTime` sync (original proposal)
**Pros:** Trivial to implement; streams naturally; low memory.
**Cons:** Cannot meet either target. Drift is structural, not a tuning problem. Correcting drift
requires seeking, which is audible — so the fix is worse than the fault.

### B — Web Audio, single `AudioContext` ✅ (default)
All tracks decoded to `AudioBuffer`s, scheduled on one clock, switched by gain crossfade.
**Pros:** One clock ⇒ **zero inter-track drift by construction**. Switch is a gain ramp — no seek,
no decode, no network. Sample-accurate scheduling. Drift vs video corrected by inaudible
`playbackRate` nudges rather than seeks.
**Cons:** Requires full decode into memory (~10 MB per track per 10 minutes). Not viable for long
media. Seeking rebuilds the graph (cheap, but code to write). Platform quirks: suspended contexts,
iOS silent switch.

### C — HLS with `EXT-X-MEDIA` audio renditions ✅ (long media)
One playlist, per-speaker audio renditions, `hls.audioTrack = n`.
**Pros:** Streams — no duration limit. Browser handles A/V sync natively. Standard, well-supported.
**Cons:** Rendition switching flushes and refills the audio buffer → **200–500 ms gap**, missing
NFR-PERF-03. Safari's native implementation differs from hls.js.

### D — Server-side muxing on switch
**Pros:** Simple client.
**Cons:** Network round trip per switch; seconds of latency. Disqualified immediately.

## Decision

Implement **both B and C behind one `PlaybackEngine` interface**. Select per project:

| Condition | Engine |
|---|---|
| duration ≤ 10 min **and** total audio ≤ 40 MB | **WebAudioSyncEngine** (B) |
| otherwise | **HlsSyncEngine** (C) |

The server decides and communicates it via `playback_hint` in the manifest, so the policy lives in
one place.

## Rationale

Neither engine covers the whole range, and the product's typical input — interviews and meetings of
a few minutes — sits squarely in B's range. So B gets the engineering attention and delivers the
headline claim; C exists so long media works at all rather than failing.

Option B's central property is worth restating: putting every track on one `AudioContext` clock does
not *reduce* inter-track drift, it **eliminates the possibility of it**. Combined with the S9
length assertion (invariant I3), the tracks are sample-aligned for their entire duration. That turns
a continuous engineering problem into a one-time packaging check.

Correcting video drift by rate rather than by seeking is the other key move: a 0.4% rate change is
inaudible (~7 cents of pitch) and resolves a 40 ms error over ten seconds, whereas any seek is
instantly audible. This is precisely what the original design got backwards.

## Consequences

**Positive** — ~80 ms switch, entirely crossfade, no network. Zero inter-track drift. Meets all
playback requirements for typical content. Instant A/B comparison becomes possible (FR-PLAY-15), as
does a multi-speaker mixer (FR-PLAY-13) — both fall out of the architecture for free.

**Negative** — Two engines to build, test and maintain. Memory cost for B. Long media has a
documented, worse switching experience. Significant platform-quirk handling (iOS, Bluetooth, device
changes). An `AudioContext` leak breaks playback after several navigations, so teardown discipline
matters.

**Neutral** — The 40 MB threshold is a tunable policy, not an architectural boundary.

## Revisit when

- Browsers ship gapless HLS audio-rendition switching → collapse to C alone.
- A standard multi-track media API arrives with real cross-browser support.
- Telemetry shows most projects exceed the WebAudio threshold → raise it or invest in C.
- Measured p95 switch latency in production exceeds 120 ms.
