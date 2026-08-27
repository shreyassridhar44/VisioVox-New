/**
 * The sync arithmetic, pinned down.
 *
 * These are the properties a browser test cannot check cheaply and a reviewer
 * cannot check by eye: that the crossfade really is equal-power, that a drift
 * correction pulls in the right direction, and that the rate trim can never
 * exceed the audibility floor. The browser harness (Playwright + Web Audio
 * measurement, docs/12 §7) covers what happens when real devices are involved;
 * everything provable without one is proved here.
 */

import { describe, expect, it } from 'vitest';
import {
  CROSSFADE_S,
  HARD_DRIFT_MS,
  MAX_RATE_TRIM,
  SOFT_DRIFT_MS,
  TRIM_RESPONSE_S,
  WEBAUDIO_MAX_BYTES,
  WEBAUDIO_MAX_DURATION_MS,
  classifyDrift,
  crossfadeCurve,
  driftMs,
  expectedMediaTime,
  fitsWebAudio,
  rateTrim,
  resolveEngine,
  type Anchor,
} from './sync';
import type { Manifest } from './manifest';

describe('crossfadeCurve', () => {
  it('holds constant power across the whole transition', () => {
    // The reason for cos/sin rather than a linear ramp. Two uncorrelated
    // signals sum in power, so if from² + to² ever dips below 1 the listener
    // hears a hole in the middle of the switch.
    for (const point of crossfadeCurve()) {
      expect(point.from ** 2 + point.to ** 2).toBeCloseTo(1, 10);
    }
  });

  it('would fail this check if it were linear', () => {
    // Guards the guard: a linear fade dips to 0.5 power (about -3 dB) at the
    // midpoint, so the assertion above is genuinely discriminating.
    const linearMidpoint = 0.5 ** 2 + 0.5 ** 2;
    expect(linearMidpoint).toBeLessThan(0.9);
  });

  it('starts fully on the old track and ends fully on the new one', () => {
    const curve = crossfadeCurve();
    const first = curve[0];
    const last = curve[curve.length - 1];
    expect(first).toBeDefined();
    expect(last).toBeDefined();
    expect(first?.from).toBeCloseTo(1, 10);
    expect(first?.to).toBeCloseTo(0, 10);
    expect(last?.from).toBeCloseTo(0, 10);
    expect(last?.to).toBeCloseTo(1, 10);
  });

  it('spans exactly the crossfade duration, which is the switch latency', () => {
    const curve = crossfadeCurve();
    expect(curve[curve.length - 1]?.at).toBeCloseTo(CROSSFADE_S, 10);
    // NFR-PERF-03 allows 120 ms; the whole switch is the fade and nothing else.
    expect(CROSSFADE_S * 1000).toBeLessThanOrEqual(120);
  });

  it('rises monotonically on the incoming track', () => {
    const curve = crossfadeCurve();
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i]?.to ?? 0).toBeGreaterThan(curve[i - 1]?.to ?? 0);
    }
  });
});

describe('classifyDrift', () => {
  it('leaves small errors alone rather than chasing noise', () => {
    expect(classifyDrift(0)).toBe('hold');
    expect(classifyDrift(SOFT_DRIFT_MS)).toBe('hold');
    expect(classifyDrift(-SOFT_DRIFT_MS)).toBe('hold');
  });

  it('trims between the soft and hard thresholds', () => {
    expect(classifyDrift(SOFT_DRIFT_MS + 1)).toBe('trim');
    expect(classifyDrift(HARD_DRIFT_MS)).toBe('trim');
    expect(classifyDrift(-(SOFT_DRIFT_MS + 1))).toBe('trim');
  });

  it('re-anchors when something has clearly jumped', () => {
    expect(classifyDrift(HARD_DRIFT_MS + 1)).toBe('reanchor');
    expect(classifyDrift(-(HARD_DRIFT_MS + 1))).toBe('reanchor');
  });
});

describe('rateTrim', () => {
  it('does nothing at zero drift', () => {
    expect(rateTrim(0)).toBe(1);
  });

  it('speeds the audio up when the picture is ahead', () => {
    // Sign matters more than magnitude here: getting it backwards doubles the
    // error instead of erasing it, and the symptom is a slow slide out of
    // sync that looks like the correction is not running at all.
    expect(rateTrim(100)).toBeGreaterThan(1);
    expect(rateTrim(-100)).toBeLessThan(1);
  });

  it('never exceeds the audibility floor, however large the drift', () => {
    for (const ms of [500, 5_000, 1_000_000, -1_000_000]) {
      expect(rateTrim(ms)).toBeLessThanOrEqual(1 + MAX_RATE_TRIM);
      expect(rateTrim(ms)).toBeGreaterThanOrEqual(1 - MAX_RATE_TRIM);
    }
  });

  it('is proportional only below the clamp knee', () => {
    // The knee sits at MAX_RATE_TRIM * TRIM_RESPONSE_S * 1000 = 16 ms.
    expect(rateTrim(8) - 1).toBeCloseTo(8 / (TRIM_RESPONSE_S * 1000), 12);
  });

  it('runs every real correction at the ceiling, clearing 40 ms in ten seconds', () => {
    // The knee is below SOFT_DRIFT_MS, so every drift large enough to act on is
    // already clamped. The timescale therefore comes from the ceiling, not from
    // the proportional term — which is what docs/12 §3.4 actually claims.
    const trim = rateTrim(SOFT_DRIFT_MS);
    expect(trim - 1).toBeCloseTo(MAX_RATE_TRIM, 12);
    expect(SOFT_DRIFT_MS / ((trim - 1) * 1000)).toBeCloseTo(10, 6);
  });
});

describe('drift measurement', () => {
  const anchor: Anchor = { ctxTime: 10, mediaTime: 4 };

  it('reads zero when both clocks agree', () => {
    expect(expectedMediaTime(anchor, 12, 1)).toBe(6);
    expect(driftMs(anchor, 12, 6, 1)).toBe(0);
  });

  it('scales with playback rate', () => {
    expect(expectedMediaTime(anchor, 12, 2)).toBe(8);
  });

  it('reports the picture running ahead as positive', () => {
    expect(driftMs(anchor, 12, 6.05, 1)).toBeCloseTo(50, 6);
  });

  it('subtracts the manual device offset', () => {
    // A Bluetooth headset lags by a fixed amount that no in-app correction can
    // fix, so the user-set offset has to leave the measurement at zero rather
    // than being fought by the trim loop.
    expect(driftMs(anchor, 12, 6.05, 1, 50)).toBeCloseTo(0, 6);
  });
});

describe('engine selection', () => {
  const manifest = { playback_hint: 'webaudio' } as Manifest;

  it('follows the server hint, because the policy lives there', () => {
    expect(resolveEngine(manifest, true)).toBe('webaudio');
  });

  it('only ever overrides downward', () => {
    expect(resolveEngine(manifest, false)).toBe('hls');
    expect(resolveEngine({ playback_hint: 'hls' } as Manifest, true)).toBe('hls');
  });
});

describe('fitsWebAudio', () => {
  it('accepts a typical few-minute interview', () => {
    expect(fitsWebAudio(4 * 60 * 1000, 12 * 1024 * 1024)).toBe(true);
  });

  it('rejects on either axis independently', () => {
    expect(fitsWebAudio(WEBAUDIO_MAX_DURATION_MS + 1, 1)).toBe(false);
    expect(fitsWebAudio(1, WEBAUDIO_MAX_BYTES + 1)).toBe(false);
  });

  it('is inclusive at the boundary', () => {
    expect(fitsWebAudio(WEBAUDIO_MAX_DURATION_MS, WEBAUDIO_MAX_BYTES)).toBe(true);
  });
});
