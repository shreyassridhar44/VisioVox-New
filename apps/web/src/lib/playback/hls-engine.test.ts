/**
 * Rendition matching for the HLS engine.
 *
 * The consequence of getting this wrong is specific and bad: the player
 * switches to the wrong person and nothing errors, because a valid rendition
 * did load. So the matching is pulled out as a pure function and pinned here,
 * separately from the streaming machinery that genuinely needs a browser.
 */

import { describe, expect, it } from 'vitest';
import { renditionIndexFor, renditionOrder, sameResource } from './hls-engine';
import type { Manifest, Speaker } from './manifest';
import { MIXED_TRACK } from './manifest';

function speaker(id: string, label: string, hls?: string): Speaker {
  return {
    id,
    ordinal: 1,
    label,
    color_token: 'spk-1',
    modality: 'audio_only',
    speaking_ratio: 0.3,
    mean_confidence: 0.9,
    extraction_ok: true,
    audio: { faithful: { url: `${id}.m4a`, bytes: 1 }, ...(hls === undefined ? {} : { hls }) },
    captions: { vtt: `${id}.vtt` },
  };
}

const manifest: Manifest = {
  project_id: 'prj_01HX8ZQ3M7N4P5R6S7T8V9W0XY',
  manifest_version: '1.0',
  duration_ms: 1_800_000,
  has_video: true,
  speakers: [
    speaker('spk_a', 'Speaker 1', 'https://cdn/audio/spk_1.m3u8'),
    speaker('spk_b', 'Speaker 2', 'https://cdn/audio/spk_2.m3u8'),
  ],
  mixed: { audio_url: 'mixed.m4a', hls: 'https://cdn/audio/mixed.m3u8' },
  master_playlist: 'https://cdn/master.m3u8',
  playback_hint: 'hls',
  warnings: [],
  signed_until: '2099-01-01T00:00:00Z',
};

const order = renditionOrder(manifest);

function uriFor(id: string): string | undefined {
  if (id === MIXED_TRACK) return manifest.mixed.hls;
  return manifest.speakers.find((s) => s.id === id)?.audio.hls;
}

describe('renditionOrder', () => {
  it('puts the original mix first, as the playlist declares it DEFAULT', () => {
    expect(order.map((o) => o.id)).toEqual([MIXED_TRACK, 'spk_a', 'spk_b']);
  });
});

describe('renditionIndexFor', () => {
  const tracks = [
    { name: 'Original mix', url: 'https://cdn/audio/mixed.m3u8' },
    { name: 'Speaker 1', url: 'https://cdn/audio/spk_1.m3u8' },
    { name: 'Speaker 2', url: 'https://cdn/audio/spk_2.m3u8' },
  ];

  it('matches on the playlist URI', () => {
    expect(renditionIndexFor(order, tracks, 'spk_b', uriFor)).toBe(2);
    expect(renditionIndexFor(order, tracks, MIXED_TRACK, uriFor)).toBe(0);
  });

  it('matches despite a re-signed query string', () => {
    const signed = tracks.map((t) => ({ ...t, url: `${t.url}?X-Amz-Signature=later` }));
    expect(renditionIndexFor(order, signed, 'spk_a', uriFor)).toBe(1);
  });

  it('ignores playlist order, which need not match the manifest', () => {
    const shuffled = [tracks[2], tracks[0], tracks[1]].filter((t) => t !== undefined);
    expect(renditionIndexFor(order, shuffled, 'spk_b', uriFor)).toBe(0);
    expect(renditionIndexFor(order, shuffled, 'spk_a', uriFor)).toBe(2);
  });

  it('falls back to the name when the player exposes no URL', () => {
    const nameless = tracks.map((t) => ({ name: t.name }));
    expect(renditionIndexFor(order, nameless, 'spk_b', uriFor)).toBe(2);
  });

  it('prefers the URI over the name when the two disagree', () => {
    // Two speakers can end up sharing a label once a user renames them. Name
    // matching would then switch to whichever came first — the wrong person,
    // with no error, which is the failure this ordering exists to prevent.
    const duplicated = [
      { name: 'Original mix', url: 'https://cdn/audio/mixed.m3u8' },
      { name: 'Alex', url: 'https://cdn/audio/spk_1.m3u8' },
      { name: 'Alex', url: 'https://cdn/audio/spk_2.m3u8' },
    ];
    expect(renditionIndexFor(order, duplicated, 'spk_b', uriFor)).toBe(2);
  });

  it('reports failure rather than guessing when nothing matches', () => {
    expect(renditionIndexFor(order, [{ name: 'Something else' }], 'spk_a', uriFor)).toBe(-1);
    expect(renditionIndexFor(order, tracks, 'spk_unknown', uriFor)).toBe(-1);
  });
});

describe('sameResource', () => {
  it('compares paths, not signatures', () => {
    expect(sameResource('https://cdn/a.m3u8?sig=1', 'https://cdn/a.m3u8?sig=2')).toBe(true);
    expect(sameResource('https://cdn/a.m3u8', 'https://cdn/b.m3u8')).toBe(false);
  });
});
