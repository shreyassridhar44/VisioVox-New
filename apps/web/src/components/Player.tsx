'use client';

/**
 * The player: pick a speaker, hear only them, in sync with the picture.
 *
 * The component owns the DOM and the engine owns the clock. Nothing here
 * measures time by counting frames or running an interval — the media position
 * is read from the video element once per frame and everything else derives
 * from it, so captions and highlighting cannot drift away from the audio.
 *
 * Selecting a speaker is deliberately not awaited anywhere: the engine
 * schedules the crossfade on the audio thread and it completes in 80 ms
 * whatever the main thread is doing, so blocking the UI on it would add
 * latency rather than remove it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AudioMode, Manifest, Speaker, TrackId } from '@/lib/playback/manifest';
import { MIXED_TRACK, hasNaturalMode } from '@/lib/playback/manifest';
import type { CaptionIndex, CaptionSegment } from '@/lib/playback/captions';
import {
  EMPTY_CAPTION_INDEX,
  buildCaptionIndex,
  parseTranscript,
  segmentAt,
  wordAt,
} from '@/lib/playback/captions';
import type { PlaybackEngine } from '@/lib/playback/engine';
import { WebAudioSyncEngine } from '@/lib/playback/web-audio-engine';
import { MAX_AV_OFFSET_MS, resolveEngine } from '@/lib/playback/sync';

interface PlayerProps {
  readonly manifest: Manifest;
}

export function Player({ manifest }: PlayerProps): React.JSX.Element {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const engineRef = useRef<PlaybackEngine | null>(null);

  const [mode, setMode] = useState<AudioMode>('faithful');
  const [activeId, setActiveId] = useState<TrackId>(manifest.speakers[0]?.id ?? MIXED_TRACK);
  const [ready, setReady] = useState(false);
  const [drift, setDrift] = useState(0);
  const [offsetMs, setOffsetMs] = useState(0);
  const [mediaMs, setMediaMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [captions, setCaptions] = useState<Record<string, CaptionIndex>>({});

  const engineKind = useMemo(
    () => resolveEngine(manifest, typeof AudioContext !== 'undefined'),
    [manifest],
  );
  const naturalAvailable = useMemo(() => hasNaturalMode(manifest), [manifest]);

  // ------------------------------------------------------------ engine wiring

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) return;

    if (engineKind === 'hls') {
      // HlsSyncEngine is weeks 9-11 of Phase 6. Saying so is better than a
      // player that silently plays the mixture and looks broken instead.
      setError(
        'This recording is too long for in-browser mixing. Streaming playback is not built yet.',
      );
      return;
    }

    const engine = new WebAudioSyncEngine({ mode });
    engineRef.current = engine;

    const offTrack = engine.on('trackchange', setActiveId);
    const offDrift = engine.on('drift', setDrift);
    const offError = engine.on('error', (e) => {
      setError(e.message);
    });

    let cancelled = false;
    void engine
      .load(manifest, video)
      .then(() => {
        if (cancelled) return;
        setReady(true);
        setOffsetMs(engine.getOffsetMs());
        const active = engine.getActiveTrack();
        if (active !== null) setActiveId(active);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not load audio');
      });

    return () => {
      cancelled = true;
      offTrack();
      offDrift();
      offError();
      engine.destroy();
      engineRef.current = null;
      setReady(false);
    };
  }, [manifest, mode, engineKind]);

  // --------------------------------------------------------- caption loading

  useEffect(() => {
    let cancelled = false;
    async function loadCaptions(): Promise<void> {
      const entries = await Promise.all(
        manifest.speakers.map(async (speaker): Promise<[string, CaptionIndex] | null> => {
          const url = speaker.captions.json;
          if (url === undefined) return null;
          try {
            const response = await fetch(url);
            if (!response.ok) return null;
            const transcript = parseTranscript(await response.json());
            return transcript === null ? null : [speaker.id, buildCaptionIndex(transcript)];
          } catch {
            // A missing transcript costs captions for one speaker, not the
            // session. The warning already rides in manifest.warnings.
            return null;
          }
        }),
      );
      if (cancelled) return;
      setCaptions(Object.fromEntries(entries.filter((e) => e !== null)));
    }
    void loadCaptions();
    return () => {
      cancelled = true;
    };
  }, [manifest]);

  // -------------------------------------------------------- media time ticker

  useEffect(() => {
    let raf = 0;
    const tick = (): void => {
      const video = videoRef.current;
      if (video !== null) setMediaMs(video.currentTime * 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
    };
  }, []);

  // -------------------------------------------------------------- interaction

  const select = useCallback((id: TrackId): void => {
    setActiveId(id);
    void engineRef.current?.selectTrack(id);
  }, []);

  const seekToMs = useCallback((ms: number): void => {
    const video = videoRef.current;
    if (video !== null) video.currentTime = ms / 1000;
  }, []);

  const changeOffset = useCallback((value: number): void => {
    setOffsetMs(value);
    engineRef.current?.setOffsetMs(value);
  }, []);

  const activeIndex = captions[activeId] ?? EMPTY_CAPTION_INDEX;
  const segment = segmentAt(activeIndex, mediaMs);

  return (
    <div className="player stack">
      <div className="stage">
        {manifest.has_video && manifest.video !== undefined ? (
          <video
            ref={videoRef}
            className="stage-video"
            src={manifest.video.url}
            controls
            playsInline
            muted
            preload="auto"
          />
        ) : (
          <div className="stage-audio-only">
            <video ref={videoRef} controls preload="auto" muted />
            <p className="muted">Audio only — no video track in this recording.</p>
          </div>
        )}
      </div>

      {error !== null && <p className="error">{error}</p>}

      <SpeakerRail
        speakers={manifest.speakers}
        activeId={activeId}
        ready={ready}
        onSelect={select}
      />

      <CaptionView segment={segment} mediaMs={mediaMs} onSeek={seekToMs} />

      <div className="player-controls row">
        {naturalAvailable && (
          <div className="seg" role="group" aria-label="Audio mode">
            <button
              type="button"
              className={mode === 'faithful' ? 'seg-on' : ''}
              onClick={() => {
                setMode('faithful');
              }}
            >
              Faithful
            </button>
            <button
              type="button"
              className={mode === 'natural' ? 'seg-on' : ''}
              onClick={() => {
                setMode('natural');
              }}
            >
              Natural
            </button>
          </div>
        )}

        <label className="offset">
          Audio delay
          <input
            type="range"
            min={-MAX_AV_OFFSET_MS}
            max={MAX_AV_OFFSET_MS}
            step={10}
            value={offsetMs}
            onChange={(e) => {
              changeOffset(Number(e.target.value));
            }}
          />
          <output>{String(Math.round(offsetMs))} ms</output>
        </label>

        <span className="muted drift" title="Video minus audio, corrected continuously">
          drift {String(Math.round(drift))} ms
        </span>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- speaker rail

interface SpeakerRailProps {
  readonly speakers: readonly Speaker[];
  readonly activeId: TrackId;
  readonly ready: boolean;
  readonly onSelect: (id: TrackId) => void;
}

function SpeakerRail({ speakers, activeId, ready, onSelect }: SpeakerRailProps): React.JSX.Element {
  return (
    <div className="rail" role="radiogroup" aria-label="Speaker">
      {speakers.map((speaker) => (
        <button
          key={speaker.id}
          type="button"
          role="radio"
          aria-checked={activeId === speaker.id}
          className={'rail-item' + (activeId === speaker.id ? ' rail-on' : '')}
          disabled={!ready || !speaker.extraction_ok}
          onClick={() => {
            onSelect(speaker.id);
          }}
        >
          {speaker.thumbnail_url !== undefined ? (
            // A plain <img>, not next/image: these are signed CDN URLs with an
            // expiry, and the image optimiser would cache them past
            // signed_until and then serve 403s from its own cache.
            <img className="rail-face" src={speaker.thumbnail_url} alt="" />
          ) : (
            <span className="rail-face rail-face-none" aria-hidden="true">
              {String(speaker.ordinal)}
            </span>
          )}
          <span className="rail-label">{speaker.label}</span>
          <span className="rail-meta muted">
            {String(Math.round(speaker.speaking_ratio * 100))}% ·{' '}
            {speaker.modality === 'audio_only' ? 'no face' : 'face + voice'}
          </span>
        </button>
      ))}

      <button
        type="button"
        role="radio"
        aria-checked={activeId === MIXED_TRACK}
        className={'rail-item' + (activeId === MIXED_TRACK ? ' rail-on' : '')}
        disabled={!ready}
        onClick={() => {
          onSelect(MIXED_TRACK);
        }}
      >
        <span className="rail-face rail-face-none" aria-hidden="true">
          ∑
        </span>
        <span className="rail-label">Original mix</span>
        <span className="rail-meta muted">everyone, unprocessed</span>
      </button>
    </div>
  );
}

// --------------------------------------------------------------------- captions

interface CaptionViewProps {
  readonly segment: CaptionSegment | null;
  readonly mediaMs: number;
  readonly onSeek: (ms: number) => void;
}

function CaptionView({ segment, mediaMs, onSeek }: CaptionViewProps): React.JSX.Element {
  if (segment === null) {
    return (
      <p className="captions captions-idle muted" aria-live="off">
        —
      </p>
    );
  }

  const current = wordAt(segment, mediaMs);
  if (segment.words.length === 0) {
    return (
      <p className="captions" aria-live="polite">
        {segment.text}
      </p>
    );
  }

  return (
    <p className="captions" aria-live="polite">
      {segment.words.map((word, i) => (
        <button
          key={String(word.start_ms) + ':' + String(i)}
          type="button"
          className={'word' + (current === word ? ' word-on' : '')}
          // Confidence is disclosed rather than hidden: a word the model was
          // unsure of is exactly the word a reader should check against the
          // audio, and dimming it is the cheapest way to say so.
          style={word.probability < 0.6 ? { opacity: 0.55 } : undefined}
          onClick={() => {
            onSeek(word.start_ms);
          }}
        >
          {word.text}
        </button>
      ))}
    </p>
  );
}
