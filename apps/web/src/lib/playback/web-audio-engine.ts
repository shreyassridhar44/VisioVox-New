/**
 * WebAudioSyncEngine — one AudioContext, one clock, zero A-to-A drift.
 *
 * The design is docs/12 §3, and the reason it works: every track is scheduled
 * on the same AudioContext at the same moment, and switching speakers only
 * changes gain. Nothing starts, nothing seeks, nothing re-decodes — so there
 * is no mechanism by which two tracks could come to disagree about where they
 * are. The archived approach (N audio elements re-synced by assigning
 * `currentTime`) fails because `currentTime =` is a seek request rather than a
 * clock write, and independent elements have independent clocks to begin with.
 *
 * What can still drift is video against audio, because those genuinely are two
 * subsystems. That is what the rAF tick corrects, by rate rather than by seek.
 */

import type { AudioMode, Manifest, Track, TrackId } from './manifest';
import { MIXED_TRACK, tracksFor } from './manifest';
import type { EngineEvents, EngineState, PlaybackEngine, Unsubscribe } from './engine';
import { Emitter } from './engine';
import type { Anchor } from './sync';
import {
  LOOKAHEAD_S,
  MAX_AV_OFFSET_MS,
  clamp,
  classifyDrift,
  crossfadeCurve,
  driftMs,
  rateTrim,
} from './sync';

interface Voice {
  readonly src: AudioBufferSourceNode;
  readonly gain: GainNode;
}

export interface WebAudioEngineOptions {
  readonly mode?: AudioMode;
  /** Injected so tests and the measurement harness can supply fixtures. */
  readonly fetchImpl?: typeof fetch;
}

const OFFSET_STORAGE_KEY = 'visiovox.avOffsetMs';

export class WebAudioSyncEngine implements PlaybackEngine {
  private readonly emitter = new Emitter<EngineEvents>();
  private readonly buffers = new Map<TrackId, AudioBuffer>();
  private readonly voices = new Map<TrackId, Voice>();
  private readonly fetchImpl: typeof fetch;
  private readonly mode: AudioMode;

  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private video: HTMLVideoElement | null = null;
  private tracks: Track[] = [];

  private anchor: Anchor | null = null;
  private activeId: TrackId | null = null;
  private rate = 1;
  private raf = 0;
  private lastDrift = 0;
  private offsetMs = 0;
  private volume = 1;
  private state: EngineState = 'idle';
  private destroyed = false;

  constructor(options: WebAudioEngineOptions = {}) {
    this.mode = options.mode ?? 'faithful';
    this.fetchImpl = options.fetchImpl ?? ((...args) => fetch(...args));
    this.offsetMs = readStoredOffset();
  }

  // ---------------------------------------------------------------- loading

  async load(manifest: Manifest, video: HTMLVideoElement): Promise<void> {
    this.setState('loading');
    this.video = video;
    this.tracks = tracksFor(manifest, this.mode);
    this.activeId = this.tracks[1]?.id ?? MIXED_TRACK;

    const ctx = new AudioContext({ latencyHint: 'playback' });
    this.ctx = ctx;
    const master = ctx.createGain();
    master.gain.value = this.volume;
    master.connect(ctx.destination);
    this.master = master;

    // The video element is picture and clock only. Were it ever audible we
    // would be mixing the un-normalised original under the isolated track.
    video.muted = true;
    video.addEventListener('play', this.onPlay);
    video.addEventListener('pause', this.onPause);
    video.addEventListener('seeked', this.onSeek);
    video.addEventListener('ratechange', this.onRateChange);
    document.addEventListener('visibilitychange', this.onVisibility);
    ctx.addEventListener('statechange', this.onCtxStateChange);

    // Decode the selected track first and resolve on it, so playback can begin
    // while the rest are still arriving. A late buffer is scheduled into the
    // running graph at the right offset by `attach`, which is what makes the
    // early resolve safe rather than a race.
    const first = this.tracks.find((t) => t.id === this.activeId) ?? this.tracks[0];
    if (first === undefined) throw new Error('manifest contains no playable audio');

    const rest = this.tracks.filter((t) => t.id !== first.id);
    await this.decode(first);
    void Promise.allSettled(rest.map((t) => this.decode(t)));

    this.setState('ready');
  }

  /**
   * Read through a call rather than the field directly: TypeScript narrows
   * `this.destroyed` to `false` after the first check and keeps that narrowing
   * across every `await`, then reports the later checks as dead code. A call
   * expression is never narrowed — the same convention as `apps/web/src/app/
   * projects/[id]/page.tsx`.
   */
  private isDestroyed(): boolean {
    return this.destroyed;
  }

  private async decode(track: Track): Promise<void> {
    const ctx = this.ctx;
    if (ctx === null || this.isDestroyed()) return;
    try {
      const response = await this.fetchImpl(track.url);
      if (!response.ok) throw new Error(track.label + ': HTTP ' + String(response.status));
      const bytes = await response.arrayBuffer();
      const buffer = await ctx.decodeAudioData(bytes);
      if (this.isDestroyed()) return;
      this.buffers.set(track.id, buffer);
      // If the graph is already running this track has missed the scheduled
      // start, so it has to be placed at the current position instead.
      if (this.anchor !== null) this.attach(track.id, buffer, this.currentMediaTime());
      this.emitter.emit('trackready', track.id);
    } catch (error) {
      this.emitter.emit('error', error instanceof Error ? error : new Error(String(error)));
    }
  }

  // -------------------------------------------------------------- transport

  private readonly onPlay = (): void => {
    const ctx = this.ctx;
    const video = this.video;
    if (ctx === null || video === null) return;
    // Every browser starts the context suspended, and resuming is only allowed
    // inside a user gesture. `play` is that gesture — non-negotiable on iOS.
    void ctx.resume();
    this.rate = video.playbackRate;
    this.start(video.currentTime);
    this.startTicking();
  };

  private readonly onPause = (): void => {
    this.stopAll();
    this.stopTicking();
  };

  private readonly onSeek = (): void => {
    const video = this.video;
    if (video === null || this.anchor === null) return;
    // AudioBufferSourceNode is one-shot and cannot be repositioned, so a seek
    // tears the graph down and rebuilds it. That is cheap because the buffers
    // are already decoded — no I/O. Bound to `seeked` rather than `seeking` so
    // a scrub does not rebuild the graph sixty times a second.
    this.stopAll();
    this.start(video.currentTime);
  };

  private readonly onRateChange = (): void => {
    const video = this.video;
    if (video === null) return;
    this.rate = video.playbackRate;
    if (this.anchor === null) return;
    this.stopAll();
    this.start(video.currentTime);
  };

  private readonly onVisibility = (): void => {
    // A hidden tab throttles rAF to about 1 Hz, so drift readings go stale and
    // corrections would chase noise. Stop correcting, then re-anchor once on
    // return rather than trusting the error accumulated while throttled.
    if (document.visibilityState === 'hidden') {
      this.stopTicking();
      return;
    }
    const video = this.video;
    if (video !== null && this.anchor !== null && !video.paused) {
      this.stopAll();
      this.start(video.currentTime);
      this.startTicking();
    }
  };

  private readonly onCtxStateChange = (): void => {
    // An output device change can reset the context. Re-anchoring is the only
    // honest response: the old anchor referred to a clock that no longer runs.
    const ctx = this.ctx;
    const video = this.video;
    if (ctx === null || video === null) return;
    if (ctx.state === 'running' && this.anchor !== null && !video.paused) {
      this.stopAll();
      this.start(video.currentTime);
    }
  };

  private start(atMediaTime: number): void {
    const ctx = this.ctx;
    if (ctx === null) return;
    const startAt = ctx.currentTime + LOOKAHEAD_S;
    this.anchor = { ctxTime: startAt, mediaTime: atMediaTime };
    for (const [id, buffer] of this.buffers) this.attach(id, buffer, atMediaTime, startAt);
  }

  /** Schedule one track into the graph. Silent unless it is the active one. */
  private attach(id: TrackId, buffer: AudioBuffer, atMediaTime: number, startAt?: number): void {
    const ctx = this.ctx;
    const master = this.master;
    if (ctx === null || master === null) return;

    const src = ctx.createBufferSource();
    const gain = ctx.createGain();
    src.buffer = buffer;
    src.playbackRate.value = this.rate;
    gain.gain.value = id === this.activeId ? 1 : 0;
    src.connect(gain).connect(master);
    src.start(startAt ?? ctx.currentTime, Math.max(0, atMediaTime));
    this.voices.set(id, { src, gain });
  }

  private stopAll(): void {
    for (const { src, gain } of this.voices.values()) {
      try {
        src.stop();
      } catch {
        // Already stopped. Web Audio throws rather than no-opping here, and
        // there is nothing to recover from.
      }
      src.disconnect();
      gain.disconnect();
    }
    this.voices.clear();
    this.anchor = null;
  }

  // -------------------------------------------------------------- switching

  selectTrack(trackId: TrackId): Promise<void> {
    const ctx = this.ctx;
    if (ctx === null || trackId === this.activeId) return Promise.resolve();

    const from = this.activeId === null ? undefined : this.voices.get(this.activeId);
    const to = this.voices.get(trackId);
    this.activeId = trackId;

    // Selecting a track that has not finished decoding is legitimate: it
    // becomes audible when `attach` places it, because `activeId` is already
    // set by then. No wait, no error.
    if (to === undefined) {
      this.emitter.emit('trackchange', trackId);
      return Promise.resolve();
    }

    const now = ctx.currentTime;
    const params: AudioParam[] = [to.gain.gain];
    if (from !== undefined) params.push(from.gain.gain);
    for (const param of params) {
      param.cancelScheduledValues(now);
      param.setValueAtTime(param.value, now);
    }

    for (const point of crossfadeCurve()) {
      const when = now + point.at;
      if (from !== undefined) from.gain.gain.linearRampToValueAtTime(point.from, when);
      to.gain.gain.linearRampToValueAtTime(point.to, when);
    }

    this.emitter.emit('trackchange', trackId);
    // Resolves immediately: the switch is scheduled on the audio thread and
    // completes in CROSSFADE_S regardless of what the main thread does next.
    return Promise.resolve();
  }

  // ------------------------------------------------------------------ drift

  private startTicking(): void {
    if (this.raf !== 0) return;
    this.raf = requestAnimationFrame(this.tick);
  }

  private stopTicking(): void {
    if (this.raf === 0) return;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  private readonly tick = (): void => {
    this.raf = 0;
    const ctx = this.ctx;
    const video = this.video;
    const anchor = this.anchor;
    if (ctx === null || video === null || anchor === null) return;

    const drift = driftMs(anchor, ctx.currentTime, video.currentTime, this.rate, this.offsetMs);
    this.lastDrift = drift;

    switch (classifyDrift(drift)) {
      case 'reanchor':
        this.stopAll();
        this.start(video.currentTime);
        this.emitter.emit('drift', drift);
        break;
      case 'trim': {
        const target = this.rate * rateTrim(drift);
        for (const { src } of this.voices.values()) {
          src.playbackRate.setTargetAtTime(target, ctx.currentTime, 0.1);
        }
        this.emitter.emit('drift', drift);
        break;
      }
      case 'hold':
        for (const { src } of this.voices.values()) {
          src.playbackRate.setTargetAtTime(this.rate, ctx.currentTime, 0.1);
        }
        break;
    }

    this.startTicking();
  };

  private currentMediaTime(): number {
    const ctx = this.ctx;
    const anchor = this.anchor;
    if (ctx === null || anchor === null) return this.video?.currentTime ?? 0;
    return anchor.mediaTime + (ctx.currentTime - anchor.ctxTime) * this.rate;
  }

  // ----------------------------------------------------------------- public

  setVolume(volume: number): void {
    this.volume = clamp(volume, 0, 1);
    if (this.master !== null) this.master.gain.value = this.volume;
  }

  setOffsetMs(offsetMs: number): void {
    this.offsetMs = clamp(offsetMs, -MAX_AV_OFFSET_MS, MAX_AV_OFFSET_MS);
    writeStoredOffset(this.offsetMs);
  }

  getOffsetMs(): number {
    return this.offsetMs;
  }

  getDrift(): number {
    return this.lastDrift;
  }

  getActiveTrack(): TrackId | null {
    return this.activeId;
  }

  getState(): EngineState {
    return this.state;
  }

  on<K extends keyof EngineEvents>(
    event: K,
    handler: (value: EngineEvents[K]) => void,
  ): Unsubscribe {
    return this.emitter.on(event, handler);
  }

  destroy(): void {
    this.destroyed = true;
    this.stopTicking();
    this.stopAll();
    const video = this.video;
    if (video !== null) {
      video.removeEventListener('play', this.onPlay);
      video.removeEventListener('pause', this.onPause);
      video.removeEventListener('seeked', this.onSeek);
      video.removeEventListener('ratechange', this.onRateChange);
    }
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.ctx?.removeEventListener('statechange', this.onCtxStateChange);
    void this.ctx?.close();
    this.ctx = null;
    this.master = null;
    this.video = null;
    this.buffers.clear();
    this.emitter.clear();
    this.setState('idle');
  }

  private setState(state: EngineState): void {
    this.state = state;
    this.emitter.emit('state', state);
  }
}

function readStoredOffset(): number {
  if (typeof window === 'undefined') return 0;
  try {
    const raw = window.localStorage.getItem(OFFSET_STORAGE_KEY);
    if (raw === null) return 0;
    const value = Number(raw);
    return Number.isFinite(value) ? clamp(value, -MAX_AV_OFFSET_MS, MAX_AV_OFFSET_MS) : 0;
  } catch {
    return 0;
  }
}

function writeStoredOffset(offsetMs: number): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(OFFSET_STORAGE_KEY, String(offsetMs));
  } catch {
    /* storage unavailable; the offset simply will not survive a reload */
  }
}
