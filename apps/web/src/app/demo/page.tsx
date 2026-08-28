'use client';

/**
 * The player, running against real audio, with no API and no GPU.
 *
 * This page exists because everything else about the sync engine can pass —
 * types, lint, unit tests — while the thing itself has never decoded a byte.
 * Here it loads real AAC, crossfades real speech and renders real transcripts,
 * which is the only way to find out whether it actually works.
 *
 * The fixture is real recorded material with its true isolated sources — by
 * default an AMI meeting, where the sources are the participants' own headset
 * microphones. So what you hear when you pick a speaker is what a perfect
 * extractor would return: a target rather than a mock, and a direct comparison
 * once SEAVE has a checkpoint to point this page at.
 */

import { useEffect, useState } from 'react';
import { Player } from '@/components/Player';
import type { Manifest, PlaybackHint } from '@/lib/playback/manifest';

const FIXTURE = '/fixtures/manifest.json';
const GENERATE = 'uv run python scripts/make_player_fixtures.py';

/**
 * Point the fixture's absolute URLs at whoever is serving this page.
 *
 * The manifest is generated with a baked-in base URL because the contract
 * requires absolute URIs — a real manifest carries signed CDN links. That
 * would tie the demo to one port, so the fixture (and only the fixture) is
 * retargeted on load, which lets `pnpm dev` on 3000 and the Playwright suite
 * on 3210 read the same file.
 */
function retarget<T>(value: T, origin: string): T {
  if (typeof value === 'string') {
    return value.replace(/^https?:\/\/[^/]+\/fixtures\//, `${origin}/fixtures/`) as T;
  }
  if (Array.isArray(value)) return value.map((v: unknown) => retarget(v, origin)) as T;
  if (typeof value === 'object' && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, retarget(v, origin)])) as T;
  }
  return value;
}

type Load = { state: 'loading' } | { state: 'missing' } | { state: 'ready'; manifest: Manifest };

/**
 * Describe the fixture that actually loaded, rather than the one that was
 * loaded when this copy was written. The generator can emit either an AMI
 * meeting or a Libri3Mix clip, and prose that quietly describes the wrong one
 * is how a demo starts lying about itself.
 */
function FixtureNotes({ manifest }: { readonly manifest: Manifest }): React.JSX.Element {
  const meeting = manifest.speakers.some((s) => s.modality === 'audiovisual');
  const minutes = Math.round(manifest.duration_ms / 1000);
  const overlap = manifest.overlap_ratio;

  return (
    <div className="panel stack">
      <h2 style={{ margin: 0, fontSize: '1rem' }}>What you are listening to</h2>
      {meeting ? (
        <>
          <p className="muted">
            {String(manifest.speakers.length)} people in a real recorded meeting — {String(minutes)}{' '}
            seconds from the AMI corpus, the in-domain evaluation set this project is judged on.
            They talk over each other{' '}
            {overlap !== undefined ? `about ${String(Math.round(overlap * 100))}% of the time` : ''}
            . The rail shows each person&rsquo;s own camera; the video is their four closeups tiled.
          </p>
          <p className="muted">
            The isolated tracks are the participants&rsquo; own headset microphones, not SEAVE
            output. That makes this the target rather than a mock — headsets are the ground truth
            the model is scored against, and the same page will show real output once C1 produces a
            checkpoint. Some bleed between headsets is audible, which is exactly why meeting audio
            is the hard case.
          </p>
        </>
      ) : (
        <p className="muted">
          A Libri3Mix mixture with its true isolated sources, loudness-normalised to −16 LUFS and
          packaged exactly as S9 will package model output. The video is a test pattern with a
          running timestamp, so A/V sync can be checked by eye — there are no faces in it, which is
          why the speakers are marked audio-only.
        </p>
      )}
    </div>
  );
}

export default function DemoPage() {
  const [load, setLoad] = useState<Load>({ state: 'loading' });
  // Both engines against the same material, so the difference is audible:
  // WebAudio switches in 80 ms, HLS flushes its buffer and takes a few hundred.
  const [engine, setEngine] = useState<PlaybackHint>('webaudio');

  useEffect(() => {
    let cancelled = false;
    void fetch(FIXTURE)
      .then(async (response) => {
        if (!response.ok) throw new Error('no fixtures');
        const manifest = retarget((await response.json()) as Manifest, window.location.origin);
        if (!cancelled) setLoad({ state: 'ready', manifest });
      })
      .catch(() => {
        // The fixtures are build artifacts, not committed: they are derived
        // from a 69 GB dataset and regenerate in about two minutes.
        if (!cancelled) setLoad({ state: 'missing' });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="stack">
      <div className="panel stack">
        <div className="row">
          <h1 style={{ margin: 0 }}>Player demo</h1>
          <span className="badge">fixture</span>
        </div>
        <p className="muted">
          People talking over each other. Pick one and you hear only them, with the picture still in
          step and captions that follow whoever is selected.
        </p>
      </div>

      {load.state === 'loading' && <p className="muted">Loading fixture…</p>}

      {load.state === 'missing' && (
        <div className="panel stack">
          <p>No fixtures yet. Generate them, then reload:</p>
          <pre className="code-block">{GENERATE}</pre>
          <p className="muted">
            Needs Libri3Mix and the Whisper weights already on this machine. Runs on CPU, so it will
            not disturb a training run.
          </p>
        </div>
      )}

      {load.state === 'ready' && (
        <>
          <div className="seg" role="group" aria-label="Playback engine">
            <button
              type="button"
              className={engine === 'webaudio' ? 'seg-on' : ''}
              onClick={() => {
                setEngine('webaudio');
              }}
            >
              WebAudio
            </button>
            <button
              type="button"
              className={engine === 'hls' ? 'seg-on' : ''}
              onClick={() => {
                setEngine('hls');
              }}
            >
              HLS
            </button>
          </div>
          <Player key={engine} manifest={load.manifest} forceEngine={engine} />
          <FixtureNotes manifest={load.manifest} />
        </>
      )}
    </div>
  );
}
