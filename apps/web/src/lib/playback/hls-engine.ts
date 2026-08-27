/**
 * HlsSyncEngine — streaming playback for recordings too long to decode (docs/12 §4).
 *
 * WebAudio decodes every track up front, which is what buys it sample-accurate
 * switching and what puts a ceiling on duration. Past that ceiling the
 * trade-off inverts: one media element streams, the browser owns A/V sync, and
 * switching costs a buffer flush instead of a gain ramp.
 *
 * That flush is a real regression and is not hidden here. hls.js discards and
 * refills the audio buffer on a rendition switch, so a switch takes roughly
 * 200-500 ms against WebAudio's 80 — outside NFR-PERF-03. The engine reports a
 * `switching` state for exactly as long as the gap lasts so the player can say
 * so, because a silence the interface has acknowledged reads as a transition
 * and an unacknowledged one reads as a bug.
 *
 * Two implementations sit behind this class. Where Media Source Extensions
 * exist, hls.js drives the switch. Safari plays HLS natively but will not
 * expose MSE on iOS, so there the engine falls back to the built-in
 * `AudioTrackList`, whose behaviour differs enough between versions to be
 * worth isolating rather than papering over.
 */

import type { AudioMode, Manifest, TrackId } from './manifest';
import { MIXED_TRACK } from './manifest';
import type { EngineEvents, EngineState, PlaybackEngine, Unsubscribe } from './engine';
import { Emitter } from './engine';
import { MAX_AV_OFFSET_MS, clamp } from './sync';

/** How long the player shows a transition state after a rendition switch. */
export const SWITCH_SETTLE_MS = 400;

export interface HlsEngineOptions {
  readonly mode?: AudioMode;
}

/** The subset of hls.js this engine touches, so the rest can be stubbed. */
interface HlsLike {
  attachMedia(video: HTMLMediaElement): void;
  loadSource(url: string): void;
  destroy(): void;
  audioTrack: number;
  readonly audioTracks: { name: string; url?: string }[];
  on(event: string, handler: (event: string, data: unknown) => void): void;
}

interface NativeAudioTrack {
  enabled: boolean;
  label: string;
  id: string;
}

interface NativeAudioTrackList {
  readonly length: number;
  [index: number]: NativeAudioTrack;
}

/** The rendition ordering the manifest implies: mixed first, then speakers. */
export function renditionOrder(manifest: Manifest): { id: TrackId; label: string }[] {
  return [
    { id: MIXED_TRACK, label: 'Original mix' },
    ...manifest.speakers.map((s) => ({ id: s.id, label: s.label })),
  ];
}

/**
 * Find the rendition index for a track.
 *
 * Matched on the playlist URI where one is exposed, and on NAME only as a
 * fallback. Names are display strings — two speakers can legitimately share a
 * label once a user renames them, and matching on that would switch to the
 * wrong person rather than fail.
 */
export function renditionIndexFor(
  order: { id: TrackId; label: string }[],
  tracks: { name: string; url?: string }[],
  trackId: TrackId,
  uriFor: (id: TrackId) => string | undefined,
): number {
  const wanted = uriFor(trackId);
  if (wanted !== undefined) {
    const byUri = tracks.findIndex((t) => t.url !== undefined && sameResource(t.url, wanted));
    if (byUri !== -1) return byUri;
  }
  const entry = order.find((o) => o.id === trackId);
  if (entry === undefined) return -1;
  return tracks.findIndex((t) => t.name === entry.label);
}

/** Compare URLs ignoring the query, because signed URLs re-sign per request. */
export function sameResource(a: string, b: string): boolean {
  const strip = (url: string): string => {
    const q = url.indexOf('?');
    return q === -1 ? url : url.slice(0, q);
  };
  return strip(a) === strip(b);
}

export class HlsSyncEngine implements PlaybackEngine {
  private readonly emitter = new Emitter<EngineEvents>();
  private readonly mode: AudioMode;

  private hls: HlsLike | null = null;
  private video: HTMLVideoElement | null = null;
  private manifest: Manifest | null = null;
  private order: { id: TrackId; label: string }[] = [];
  private activeId: TrackId | null = null;
  private offsetMs = 0;
  private volume = 1;
  private state: EngineState = 'idle';
  private settleTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(options: HlsEngineOptions = {}) {
    this.mode = options.mode ?? 'faithful';
  }

  async load(manifest: Manifest, video: HTMLVideoElement): Promise<void> {
    this.setState('loading');
    this.manifest = manifest;
    this.video = video;
    this.order = renditionOrder(manifest);
    this.activeId = manifest.speakers[0]?.id ?? MIXED_TRACK;

    const source = manifest.master_playlist;
    if (source === undefined) {
      this.setState('error');
      throw new Error('this recording has no HLS playlist; it cannot be streamed');
    }

    // Unlike the WebAudio engine the video element carries the audio here, so
    // muting it would mute everything.
    video.muted = false;
    video.volume = this.volume;

    const Hls = (await import('hls.js')).default;
    if (Hls.isSupported()) {
      const hls = new Hls({
        // Short buffer so a rendition switch refills quickly: the flush is the
        // whole cost of switching, and it scales with what has to be refilled.
        maxBufferLength: 30,
      }) as unknown as HlsLike;
      this.hls = hls;
      hls.on('hlsError', (_event, data) => {
        const detail = data as { fatal?: boolean; details?: string };
        if (detail.fatal === true) {
          this.emitter.emit('error', new Error(detail.details ?? 'fatal HLS error'));
        }
      });
      hls.attachMedia(video);
      hls.loadSource(source);
    } else if (video.canPlayType('application/vnd.apple.mpegurl') !== '') {
      // Safari, where HLS is native and MSE is unavailable on iOS.
      video.src = source;
    } else {
      this.setState('error');
      throw new Error('this browser cannot play streamed audio');
    }

    this.setState('ready');
  }

  selectTrack(trackId: TrackId): Promise<void> {
    if (trackId === this.activeId) return Promise.resolve();
    const previous = this.activeId;
    this.activeId = trackId;

    const applied = this.applyRendition(trackId);
    if (!applied) {
      this.activeId = previous;
      this.emitter.emit('error', new Error('that speaker has no streamed rendition'));
      return Promise.resolve();
    }

    // 'switching' is the honest signal. The audio really is gone for a few
    // hundred milliseconds while the buffer refills, and the player showing
    // that is the difference between a transition and an apparent fault.
    this.setState('switching');
    if (this.settleTimer !== null) clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => {
      this.settleTimer = null;
      if (this.state === 'switching') this.setState('ready');
    }, SWITCH_SETTLE_MS);

    this.emitter.emit('trackchange', trackId);
    return Promise.resolve();
  }

  private applyRendition(trackId: TrackId): boolean {
    const manifest = this.manifest;
    if (manifest === null) return false;

    const uriFor = (id: TrackId): string | undefined =>
      id === MIXED_TRACK
        ? manifest.mixed.hls
        : manifest.speakers.find((s) => s.id === id)?.audio.hls;

    const hls = this.hls;
    if (hls !== null) {
      const index = renditionIndexFor(this.order, hls.audioTracks, trackId, uriFor);
      if (index === -1) return false;
      hls.audioTrack = index;
      return true;
    }

    const native = this.nativeTracks();
    if (native === null) return false;
    const entry = this.order.find((o) => o.id === trackId);
    if (entry === undefined) return false;
    let found = false;
    for (let i = 0; i < native.length; i++) {
      const track = native[i];
      if (track === undefined) continue;
      const wanted = track.label === entry.label;
      track.enabled = wanted;
      found = found || wanted;
    }
    return found;
  }

  private nativeTracks(): NativeAudioTrackList | null {
    // `audioTracks` is Safari's and is absent from the DOM lib types, because
    // no other engine implements it. The widening is kept to this one line
    // rather than spread across the class.
    const video = this.video;
    if (video === null) return null;
    const carrier = video as unknown as { audioTracks?: NativeAudioTrackList };
    return carrier.audioTracks ?? null;
  }

  setVolume(volume: number): void {
    this.volume = clamp(volume, 0, 1);
    if (this.video !== null) this.video.volume = this.volume;
  }

  setOffsetMs(offsetMs: number): void {
    // Accepted and remembered so the control behaves consistently across
    // engines, but it cannot be applied: there is one media element and the
    // browser owns the relationship between its audio and its video. Anything
    // that claimed to shift one against the other here would be theatre.
    this.offsetMs = clamp(offsetMs, -MAX_AV_OFFSET_MS, MAX_AV_OFFSET_MS);
  }

  getOffsetMs(): number {
    return this.offsetMs;
  }

  getDrift(): number {
    // Structurally zero: audio and video are one element with one clock, so
    // there is no pair of clocks to disagree. Reported rather than omitted so
    // telemetry has the same shape from both engines.
    return 0;
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
    if (this.settleTimer !== null) clearTimeout(this.settleTimer);
    this.settleTimer = null;
    this.hls?.destroy();
    this.hls = null;
    this.video = null;
    this.manifest = null;
    this.emitter.clear();
    this.setState('idle');
  }

  private setState(state: EngineState): void {
    this.state = state;
    this.emitter.emit('state', state);
  }
}
