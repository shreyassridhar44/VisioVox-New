/**
 * Caption lookup and parsing.
 *
 * The lookup is a binary search, so the interesting cases are the edges — gaps
 * between segments, the instant a segment ends, and input that is not sorted.
 * A wrong answer here shows up as captions that lag or stick, which reads to a
 * user as a sync bug and gets chased in the engine instead.
 */

import { describe, expect, it } from 'vitest';
import type { CaptionSegment, Transcript } from './captions';
import {
  buildCaptionIndex,
  EMPTY_CAPTION_INDEX,
  parseTranscript,
  segmentAt,
  wordAt,
} from './captions';

function segment(start: number, end: number, text = 'hello there'): CaptionSegment {
  return {
    text,
    start_ms: start,
    end_ms: end,
    words: [
      { text: 'hello', start_ms: start, end_ms: start + 400, probability: 0.98 },
      { text: 'there', start_ms: start + 400, end_ms: end, probability: 0.42 },
    ],
  };
}

const transcript: Transcript = {
  language: 'en',
  language_probability: 0.99,
  segments: [segment(0, 1000), segment(2000, 3000), segment(5000, 6000)],
};

describe('segmentAt', () => {
  const index = buildCaptionIndex(transcript);

  it('finds the covering segment', () => {
    expect(segmentAt(index, 0)?.start_ms).toBe(0);
    expect(segmentAt(index, 999)?.start_ms).toBe(0);
    expect(segmentAt(index, 2500)?.start_ms).toBe(2000);
    expect(segmentAt(index, 5999)?.start_ms).toBe(5000);
  });

  it('shows nothing in a gap rather than holding the last caption', () => {
    // Leaving the previous line up through a pause makes the speaker look like
    // they are still talking, which in a product about who said what is
    // precisely the wrong thing to imply.
    expect(segmentAt(index, 1500)).toBeNull();
    expect(segmentAt(index, 4000)).toBeNull();
  });

  it('treats the end of a segment as exclusive', () => {
    expect(segmentAt(index, 1000)).toBeNull();
    expect(segmentAt(index, 3000)).toBeNull();
  });

  it('handles positions outside the transcript', () => {
    expect(segmentAt(index, -1)).toBeNull();
    expect(segmentAt(index, 60_000)).toBeNull();
  });

  it('returns null for an empty index instead of throwing', () => {
    expect(segmentAt(EMPTY_CAPTION_INDEX, 100)).toBeNull();
  });
});

describe('buildCaptionIndex', () => {
  it('sorts segments so the bisection is valid', () => {
    // A single out-of-order segment would make the search silently return the
    // wrong caption, so the index does not trust the file it was handed.
    const shuffled: Transcript = {
      ...transcript,
      segments: [segment(5000, 6000), segment(0, 1000), segment(2000, 3000)],
    };
    const index = buildCaptionIndex(shuffled);
    expect(index.segments.map((s) => s.start_ms)).toEqual([0, 2000, 5000]);
    expect(segmentAt(index, 2500)?.start_ms).toBe(2000);
  });
});

describe('wordAt', () => {
  const seg = segment(2000, 3000);

  it('picks the word being spoken', () => {
    expect(wordAt(seg, 2000)?.text).toBe('hello');
    expect(wordAt(seg, 2399)?.text).toBe('hello');
    expect(wordAt(seg, 2400)?.text).toBe('there');
  });

  it('returns null past the last word', () => {
    expect(wordAt(seg, 3000)).toBeNull();
  });
});

describe('parseTranscript', () => {
  it('reads a well-formed body', () => {
    const parsed = parseTranscript({
      language: 'en',
      language_probability: 0.97,
      segments: [
        {
          text: 'hi',
          start_ms: 0,
          end_ms: 500,
          words: [{ text: 'hi', start_ms: 0, end_ms: 500, probability: 0.9 }],
        },
      ],
    });
    expect(parsed?.language).toBe('en');
    expect(parsed?.segments).toHaveLength(1);
    expect(parsed?.segments[0]?.words[0]?.text).toBe('hi');
  });

  it('drops malformed segments instead of failing the whole transcript', () => {
    // A truncated or partially-written transcript should cost the segments it
    // actually lost, not every caption in the recording.
    const parsed = parseTranscript({
      segments: [
        { text: 'good', start_ms: 0, end_ms: 100, words: [] },
        { text: 'no timings' },
        { start_ms: 200, end_ms: 300 },
        null,
      ],
    });
    expect(parsed?.segments).toHaveLength(1);
    expect(parsed?.segments[0]?.text).toBe('good');
  });

  it('defaults a missing word probability to certain', () => {
    const parsed = parseTranscript({
      segments: [
        { text: 'x', start_ms: 0, end_ms: 10, words: [{ text: 'x', start_ms: 0, end_ms: 10 }] },
      ],
    });
    expect(parsed?.segments[0]?.words[0]?.probability).toBe(1);
  });

  it('returns null for a body that is not a transcript at all', () => {
    expect(parseTranscript(null)).toBeNull();
    expect(parseTranscript('<html>expired</html>')).toBeNull();
    expect(parseTranscript({ detail: 'Forbidden' })).toBeNull();
  });
});
