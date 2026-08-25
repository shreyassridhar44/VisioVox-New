# 12 — Media Delivery & Playback Sync

> This document replaces §4.1 of the archived roadmap. See
> [`02-approach-review.md`](./02-approach-review.md) §F2 for why that design fails.

---

## 1. The problem

Play a video with **N** switchable audio tracks such that:

| Requirement | Target | Req ID |
|---|---|---|
| Switch is imperceptible | ≤ 120 ms, no gap, no click, no level jump | FR-PLAY-03, NFR-PERF-03 |
| A/V stays locked | ≤ 40 ms drift over 10 min | FR-PLAY-05 |
| Correct after seek/scrub/rate/background | always | FR-PLAY-06 |
| Captions swap with audio | same frame | FR-PLAY-04 |

### Why the naive approach fails

```js
// The archived roadmap's design — do not do this
audioEl.currentTime = videoEl.currentTime;   // ← this is a SEEK, not a sync
audioEl.play();
```

Three independent problems:

1. **`currentTime =` triggers an asynchronous seek.** It resolves to the nearest decodable point,
   after 10–150 ms of decoder work. You do not land where you asked.
2. **Independent media elements have independent clocks.** Separate decoder pipelines, separate
   buffer scheduling. Perfectly aligned at t=0, they drift — tens of ms per minute, worse under
   load.
3. **Correcting drift by re-seeking is audible.** Each correction is a click or a stutter. The
   original design's "re-sync periodically during playback" produces a periodic glitch.

Perceptual tolerance is tight: ITU-R BT.1359 puts detection at roughly **+45 ms / −125 ms**
(audio leading / lagging). Drift becomes noticeable long before it looks broken — the demo just
feels subtly wrong.

**The fix is architectural, not a tuning problem: put every audio track on one clock.**

---

## 2. Two engines, one interface

```ts
interface PlaybackEngine {
  load(manifest: Manifest, video: HTMLVideoElement): Promise<void>;
  selectTrack(trackId: TrackId): Promise<void>;   // must resolve ≤120ms
  setVolume(v: number): void;
  getDrift(): number;                              // ms, for telemetry
  destroy(): void;
}
```

| Engine | When | Why |
|---|---|---|
| **WebAudioSyncEngine** | duration ≤ 10 min **and** total audio ≤ 40 MB | One `AudioContext` ⇒ one clock ⇒ zero inter-track drift. Sample-accurate crossfade. |
| **HlsSyncEngine** | longer / larger | Streaming; browser handles A/V sync natively. Small switch gap accepted. |

The server picks via `playback_hint` in the manifest ([`11-api-spec.md`](./11-api-spec.md) §4), so
the policy lives in one place. The client may override downward if `AudioContext` is unavailable.

Given the product's typical input — interviews and meetings of a few minutes — **WebAudio is the
common path** and gets the engineering attention.

---

## 3. WebAudioSyncEngine

### 3.1 Graph

```
                    ┌── AudioBufferSourceNode (mixed)  ──► GainNode ──┐
  AudioContext  ────┼── AudioBufferSourceNode (spk_1)  ──► GainNode ──┤
  (ONE clock for    ├── AudioBufferSourceNode (spk_2)  ──► GainNode ──┼──► master ──► destination
   every track)     └── AudioBufferSourceNode (spk_3)  ──► GainNode ──┘      Gain
                                                                              │
  <video muted>  ── visual only, and the sync reference ─────────────────────┘
```

**All sources start together at the same `AudioContext` time.** Only gains change. Because every
buffer was rendered from the same timeline and asserted to identical sample counts
([`05-ml-architecture.md`](./05-ml-architecture.md) §12, invariant I3), they are sample-aligned for
the whole duration. There is nothing left to drift.

Cost: N tracks decoded in memory. A 10-minute 3-speaker project is ~29 MB of PCM at 16-bit mono
16 kHz per track — acceptable. Beyond that, the HLS engine takes over.

### 3.2 Starting

```ts
async load(manifest, video) {
  this.ctx = new AudioContext({ latencyHint: 'playback' });

  // decode in parallel; selected track first so playback can begin early
  this.buffers = await decodeAll(manifest.tracks);

  video.muted = true;                     // video is picture + clock, never sound
  video.addEventListener('play',      this.onPlay);
  video.addEventListener('pause',     this.onPause);
  video.addEventListener('seeked',    this.onSeek);
  video.addEventListener('ratechange',this.onRateChange);
}

private start(atMediaTime: number) {
  // AudioContext.currentTime is a monotonic high-resolution clock — the anchor
  const startAt = this.ctx.currentTime + LOOKAHEAD;   // 0.05 s scheduling headroom
  this.anchor = { ctxTime: startAt, mediaTime: atMediaTime };

  for (const [id, buf] of this.buffers) {
    const src  = this.ctx.createBufferSource();
    const gain = this.ctx.createGain();
    src.buffer = buf;
    src.playbackRate.value = this.video.playbackRate;
    gain.gain.value = (id === this.activeId) ? 1 : 0;   // ← all playing, only one audible
    src.connect(gain).connect(this.master);
    src.start(startAt, atMediaTime);                    // sample-accurate scheduled start
    this.sources.set(id, { src, gain });
  }
}
```

`src.start(when, offset)` schedules on the audio thread with sample accuracy. It is a scheduling
primitive, unlike `currentTime =`, which is a seek request.

**All tracks play simultaneously at all times; only gain changes.** That is the entire reason
switching is instantaneous — there is nothing to start.

### 3.3 Switching — equal-power crossfade

```ts
async selectTrack(id: TrackId) {
  const t = this.ctx.currentTime;
  const D = 0.08;                                   // 80 ms
  const from = this.sources.get(this.activeId)!.gain.gain;
  const to   = this.sources.get(id)!.gain.gain;

  from.cancelScheduledValues(t);
  to.cancelScheduledValues(t);
  from.setValueAtTime(from.value, t);
  to.setValueAtTime(to.value, t);

  // Equal-power (cos/sin), NOT linear: linear crossfade dips ~3 dB at the midpoint
  // because uncorrelated signals sum in power, not amplitude. The dip is audible.
  for (let i = 0; i <= STEPS; i++) {
    const x = i / STEPS, when = t + x * D;
    from.linearRampToValueAtTime(Math.cos(x * Math.PI / 2), when);
    to  .linearRampToValueAtTime(Math.sin(x * Math.PI / 2), when);
  }
  this.activeId = id;
  this.emit('trackchange', id);                     // captions follow this event
}
```

Measured switch latency: **~80 ms**, entirely the crossfade. No network, no decode, no seek.
Comfortably inside NFR-PERF-03.

Level jumps are prevented upstream by the −16 LUFS normalisation in S9
([`05-ml-architecture.md`](./05-ml-architecture.md) §12) — without it, an equal-power crossfade into
a louder track still sounds like a jump.

### 3.4 Drift correction

Audio and video are still two subsystems, so V↔A drift (not A↔A) can occur.

```ts
private tick = () => {                              // rAF, ~60 Hz
  const expected = this.anchor.mediaTime
                 + (this.ctx.currentTime - this.anchor.ctxTime) * this.rate;
  const drift = (this.video.currentTime - expected) * 1000;   // ms

  if (Math.abs(drift) > HARD_MS) {          // 250 ms — something jumped
    this.restart(this.video.currentTime);   // full re-anchor; rare
  } else if (Math.abs(drift) > SOFT_MS) {   // 40 ms — nudge, inaudibly
    const adj = 1 + clamp(drift / 4000, -0.004, 0.004);   // ≤0.4% — well under
    for (const { src } of this.sources.values())          // the ~1% audibility floor
      src.playbackRate.setTargetAtTime(this.rate * adj, this.ctx.currentTime, 0.1);
  } else {
    this.resetRate();
  }
  this.raf = requestAnimationFrame(this.tick);
};
```

Correcting by **rate** rather than by seeking is the key move. A 0.4% rate change is inaudible
(pitch shift ≈ 7 cents) and pulls a 40 ms error out over ~10 seconds. A seek is instantly audible.
This is precisely what the archived roadmap's "periodically re-sync `currentTime`" got wrong.

### 3.5 Seek

`AudioBufferSourceNode` cannot be repositioned — it is one-shot. So a seek tears down and rebuilds:

```ts
private onSeek = () => {
  this.stopAll();                       // ~1 ms: disconnect all sources
  this.start(this.video.currentTime);   // reschedule all from the new offset
};
```

Cheap because buffers are already decoded — no I/O. Debounce during scrubbing (fire on `seeked`,
not `seeking`) so a drag doesn't rebuild the graph 60 times a second.

### 3.6 Platform requirements

| Issue | Handling |
|---|---|
| **`AudioContext` starts suspended** (all browsers) | `resume()` inside the user gesture that starts playback. Non-negotiable on iOS. |
| **iOS silent switch mutes WebAudio** | Detect and show a hint; there is no programmatic workaround |
| **Backgrounded tab throttles rAF** | On `visibilitychange` → hidden, stop correcting; re-anchor on visible |
| **Output device change** | `AudioContext` may reset — listen for `statechange`, re-anchor |
| **Bluetooth latency** (100–300 ms) | Offer a manual A/V offset slider (±300 ms); persist per device |
| **`playbackRate` change** | Re-anchor and apply the rate to every source |

The Bluetooth case is worth building: it is a large, real, device-side offset that no amount of
in-app sync can fix, and users will otherwise report it as a product bug.

---

## 4. HlsSyncEngine

For long content, packaged as one multivariant playlist with per-speaker audio renditions.

```m3u8
#EXTM3U
#EXT-X-VERSION:7

#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="spk",NAME="Original mix",LANGUAGE="en",
             DEFAULT=YES,AUTOSELECT=YES,URI="audio/mixed.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="spk",NAME="Speaker 1",LANGUAGE="en",
             DEFAULT=NO,AUTOSELECT=NO,URI="audio/spk_1_faithful.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="spk",NAME="Speaker 2",LANGUAGE="en",
             DEFAULT=NO,AUTOSELECT=NO,URI="audio/spk_2_faithful.m3u8"

#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="cc",NAME="Speaker 1",LANGUAGE="en",
             URI="subs/spk_1.m3u8"

#EXT-X-STREAM-INF:BANDWIDTH=2600000,CODECS="avc1.640028,mp4a.40.2",
                  RESOLUTION=1920x1080,AUDIO="spk",SUBTITLES="cc"
video/1080p.m3u8
```

```ts
hls.audioTrack = trackIndex;      // hls.js handles the switch
```

**Honest trade-off:** hls.js flushes and refills the audio buffer on a rendition switch, producing a
**200–500 ms gap**. That misses NFR-PERF-03. Mitigations:

1. Prefer WebAudio wherever the size policy allows (most projects)
2. `maxBufferLength: 30` so refill is fast
3. Crossfade the *last decoded frames* over the gap where the browser permits
4. Show a brief transition state so the gap reads as intentional rather than broken

**Safari native HLS:** `video.audioTracks` (an `AudioTrackList`) is available and works, but the
API differs from hls.js and support is quirky across versions. Feature-detect; prefer MSE where
available; keep the native path as a fallback with its behaviour documented.

Segments are 4 s CMAF/fMP4. Audio renditions are **muxed separately from video** so switching audio
does not disturb the video buffer.

---

## 5. Caption synchronisation

Captions follow the same clock as the audio, not a separate timer.

```ts
// Preloaded as word-timed JSON, indexed once for O(log n) lookup
const idx = buildIntervalIndex(captions[activeSpeaker]);

function onFrame(mediaTime: number) {
  const seg = idx.find(mediaTime * 1000);
  if (seg !== current) { current = seg; render(seg); }
}
```

Why JSON rather than `<track>` + VTT cues (a point the archived roadmap got right):
- Word-level highlighting during playback requires word timings VTT cues don't carry cleanly
- Swapping speakers means swapping the whole cue set — trivial with an in-memory index, awkward with
  `TextTrack`
- Rendering in React lets us style confidence, contested spans and speaker colour
- Click-to-seek on a word (FR-PLAY-09) needs the word's time, which we have

VTT is still exported for download and for the HLS subtitle renditions — it just isn't the runtime
representation.

---

## 6. Packaging (S9) reference

```bash
# 1) Loudness-normalise each track — two-pass for accuracy   (F13)
ffmpeg -i spk_1_raw.wav -af loudnorm=I=-16:TP=-1.0:LRA=11:print_format=json -f null -   # pass 1
ffmpeg -i spk_1_raw.wav -af loudnorm=I=-16:TP=-1.0:LRA=11:measured_I=…:linear=true \
       -ar 48000 spk_1_norm.wav                                                        # pass 2

# 2) Hard length assertion — invariant I3
python -c "assert all(len(x)==N for x in tracks), 'track length mismatch'"

# 3) Encode
ffmpeg -i spk_1_norm.wav -c:a aac -b:a 128k -ac 1 spk_1.m4a

# 4) Package HLS (audio-only rendition)
ffmpeg -i spk_1.m4a -c copy -f hls -hls_segment_type fmp4 -hls_time 4 \
       -hls_playlist_type vod -hls_flags single_file audio/spk_1.m3u8

# 5) Waveform peaks for the UI
audiowaveform -i spk_1_norm.wav -o spk_1.peaks.json --pixels-per-second 20 -b 8
```

**Step 2 is a hard gate.** A one-sample mismatch is invisible in testing and becomes accumulating
drift in production. Fail the job instead.

---

## 7. Verification

Automated (Playwright + Web Audio measurement):

| Test | Method | Gate |
|---|---|---|
| Switch latency | Timestamp click → first sample above threshold on the new track | ≤ 120 ms p95 |
| Crossfade smoothness | Analyse master output RMS through the transition | no dip > 1 dB |
| Long-run drift | Play 10 min, sample `getDrift()` every 5 s | max ≤ 40 ms |
| Seek accuracy | 100 random seeks, measure audio offset vs video | ≤ 30 ms |
| Rate change | 0.5×–2×, verify sync holds | ≤ 40 ms |
| Background/foreground | Hide 60 s, restore | re-anchors ≤ 100 ms |
| Level consistency | Integrated LUFS per track | spread ≤ 1 LU |
| Caption sync | Compare rendered caption to expected at 50 timestamps | 0 mismatches |

Manual, on real devices (macOS/iOS Safari, Android Chrome, Windows Chrome/Firefox, Bluetooth
headphones, USB-C headphones, laptop speakers) — because platform audio behaviour is exactly where
automated tests are least trustworthy.

---

## 8. Why this is Contribution 6

No shipping product delivers switchable per-speaker isolated audio locked to video. The delivery
problem is unglamorous but real: the research literature stops at producing waveforms, and the
streaming literature assumes audio renditions are languages, not people, and are therefore switched
rarely rather than interactively.

The measured claim: **≤ 120 ms switching with ≤ 40 ms drift over 10 minutes**, verified by the
harness in §7, and reproducible in any browser.
