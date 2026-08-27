/**
 * The one interface both playback engines implement (docs/12 §2).
 *
 * Two engines exist because one clock cannot serve both cases: WebAudio gives
 * sample-accurate switching but has to decode everything up front, and HLS
 * streams but hands sync to the browser. The player must not care which it
 * got, so everything past `load` is identical.
 */

import type { Manifest, TrackId } from './manifest';

export type EngineState = 'idle' | 'loading' | 'ready' | 'error';

export interface EngineEvents {
  /** The audible track changed. Captions follow this, not a separate timer. */
  trackchange: TrackId;
  /** Video-minus-audio drift in ms, emitted on correction. Telemetry. */
  drift: number;
  state: EngineState;
  /** A track finished decoding and joined the graph. */
  trackready: TrackId;
  error: Error;
}

export type Unsubscribe = () => void;

export interface PlaybackEngine {
  load(manifest: Manifest, video: HTMLVideoElement): Promise<void>;
  /** Must resolve within 120 ms (NFR-PERF-03). */
  selectTrack(trackId: TrackId): Promise<void>;
  setVolume(volume: number): void;
  /** Manual A/V correction for device latency, in ms. docs/12 §3.6. */
  setOffsetMs(offsetMs: number): void;
  getOffsetMs(): number;
  getDrift(): number;
  getActiveTrack(): TrackId | null;
  on<K extends keyof EngineEvents>(
    event: K,
    handler: (value: EngineEvents[K]) => void,
  ): Unsubscribe;
  destroy(): void;
}

/**
 * A minimal typed emitter. Nothing here needs the ceremony of a library, and
 * the engine must stay usable outside React — the Playwright measurement
 * harness drives it directly.
 */
export class Emitter<E> {
  private readonly handlers = new Map<keyof E, Set<(value: never) => void>>();

  on<K extends keyof E>(event: K, handler: (value: E[K]) => void): Unsubscribe {
    let set = this.handlers.get(event);
    if (set === undefined) {
      set = new Set();
      this.handlers.set(event, set);
    }
    const entry = handler as (value: never) => void;
    set.add(entry);
    return () => {
      set.delete(entry);
    };
  }

  emit<K extends keyof E>(event: K, value: E[K]): void {
    const set = this.handlers.get(event);
    if (set === undefined) return;
    for (const handler of [...set]) (handler as (v: E[K]) => void)(value);
  }

  clear(): void {
    this.handlers.clear();
  }
}
