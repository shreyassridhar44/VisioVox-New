/**
 * The synchronisation arithmetic, kept free of the Web Audio API (docs/12 §3).
 *
 * Everything here is a pure function of numbers so it can be tested without a
 * browser, an AudioContext or a real audio file. That matters more than usual:
 * the sync engine's failure mode is a slow, small drift that no unit test on
 * the class itself would catch, so the parts that *can* be pinned down exactly
 * — the crossfade shape, the drift thresholds, the rate trim — are pinned down
 * here, and what remains for the browser harness is genuinely browser
 * behaviour.
 */

import type { Manifest, PlaybackHint } from './manifest';

/** Scheduling headroom before a scheduled start. docs/12 §3.2. */
export const LOOKAHEAD_S = 0.05;

/** Equal-power crossfade duration. This is the whole switch latency. */
export const CROSSFADE_S = 0.08;

/** Ramp segments per crossfade. Web Audio interpolates linearly between
 *  scheduled points, so the count is what makes the curve read as a curve. */
export const CROSSFADE_STEPS = 16;

/** Below this, leave the clocks alone — correcting noise causes wobble. */
export const SOFT_DRIFT_MS = 40;

/** Above this, something jumped (tab throttle, device change): re-anchor. */
export const HARD_DRIFT_MS = 250;

/** Maximum playback-rate trim. 0.4% is ~7 cents — well under audibility. */
export const MAX_RATE_TRIM = 0.004;

/**
 * Proportional gain on the trim: the drift is divided by this many seconds to
 * get a rate offset.
 *
 * Worth being precise about, because the clamp does the real work. The
 * proportional term only binds below MAX_RATE_TRIM * 1000 * TRIM_RESPONSE_S =
 * 16 ms of drift, which is inside the hold band — so in practice every
 * correction that actually runs is pinned at the 0.4% ceiling, and a 40 ms
 * error takes about ten seconds to disappear. That is the behaviour docs/12
 * §3.4 describes; the divisor exists to make the approach smooth rather than
 * to set the timescale.
 */
export const TRIM_RESPONSE_S = 4;

/** Manual A/V offset range offered for Bluetooth latency. docs/12 §3.6. */
export const MAX_AV_OFFSET_MS = 300;

/** WebAudio decodes everything up front; past these the HLS engine wins. */
export const WEBAUDIO_MAX_DURATION_MS = 10 * 60 * 1000;
export const WEBAUDIO_MAX_BYTES = 40 * 1024 * 1024;

/**
 * The fixed point relating the audio clock to the media timeline. Every
 * position the engine computes derives from this rather than from any
 * element's `currentTime`, which is a seek request and not a clock.
 */
export interface Anchor {
  readonly ctxTime: number;
  readonly mediaTime: number;
}

/** Where the audio graph believes it is, given the AudioContext clock. */
export function expectedMediaTime(anchor: Anchor, ctxTime: number, rate: number): number {
  return anchor.mediaTime + (ctxTime - anchor.ctxTime) * rate;
}

/**
 * Video position minus audio position, in milliseconds. Positive means the
 * picture is ahead of the sound.
 */
export function driftMs(
  anchor: Anchor,
  ctxTime: number,
  videoTime: number,
  rate: number,
  offsetMs = 0,
): number {
  return (videoTime - expectedMediaTime(anchor, ctxTime, rate)) * 1000 - offsetMs;
}

export type DriftAction = 'hold' | 'trim' | 'reanchor';

export function classifyDrift(ms: number): DriftAction {
  const magnitude = Math.abs(ms);
  if (magnitude > HARD_DRIFT_MS) return 'reanchor';
  if (magnitude > SOFT_DRIFT_MS) return 'trim';
  return 'hold';
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * The playback-rate multiplier that pulls `ms` of drift back out.
 *
 * Correcting by rate rather than by seeking is the load-bearing decision: a
 * 0.4% rate change is inaudible, and a seek is instantly audible. Trading
 * "correct in ten seconds, silently" against "correct now, with a click" is
 * only a hard call if you have not heard the click.
 */
export function rateTrim(ms: number): number {
  return 1 + clamp(ms / (TRIM_RESPONSE_S * 1000), -MAX_RATE_TRIM, MAX_RATE_TRIM);
}

export interface CrossfadePoint {
  /** Offset from the start of the fade, in seconds. */
  readonly at: number;
  readonly from: number;
  readonly to: number;
}

/**
 * Equal-power (cos/sin) crossfade, not linear.
 *
 * Two uncorrelated signals sum in power, not amplitude, so a linear fade dips
 * about 3 dB at the midpoint. On a speaker switch that reads as a dropout
 * rather than a transition.
 */
export function crossfadeCurve(
  duration: number = CROSSFADE_S,
  steps: number = CROSSFADE_STEPS,
): CrossfadePoint[] {
  const points: CrossfadePoint[] = [];
  for (let i = 0; i <= steps; i++) {
    const x = i / steps;
    points.push({
      at: x * duration,
      from: Math.cos((x * Math.PI) / 2),
      to: Math.sin((x * Math.PI) / 2),
    });
  }
  return points;
}

/**
 * The engine to actually use.
 *
 * The server already decided via `playback_hint`, so the thresholds are not
 * re-implemented here — duplicating that policy is how the two copies drift
 * apart. The client only ever overrides *downward*, when the platform cannot
 * support what the server picked.
 */
export function resolveEngine(manifest: Manifest, audioContextAvailable: boolean): PlaybackHint {
  if (!audioContextAvailable) return 'hls';
  return manifest.playback_hint;
}

/**
 * Whether a manifest is within the WebAudio envelope. Used by the server-side
 * policy and mirrored here only so the same numbers can be asserted in tests.
 */
export function fitsWebAudio(durationMs: number, totalBytes: number): boolean {
  return durationMs <= WEBAUDIO_MAX_DURATION_MS && totalBytes <= WEBAUDIO_MAX_BYTES;
}
