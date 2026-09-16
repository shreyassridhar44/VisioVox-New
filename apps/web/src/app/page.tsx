/**
 * Landing — the expressive surface (docs/28 §D1).
 *
 * The 3D is loaded as its own chunk with `ssr: false`, so the page's first
 * paint never waits on three.js and a visitor who will never see the canvas
 * (reduced motion, no WebGL, Save-Data) does not download it at all. That is
 * what keeps LCP inside NFR-PERF-04 on a mid-tier phone.
 */

import Link from 'next/link';

import { HeroSlot } from '@/components/HeroSlot';
import { Badge, Button, Card } from '@/components/ui';

const CONTRIBUTIONS = [
  {
    title: 'Self-enrolment',
    body: 'Extraction normally needs a clean reference recording of each person. Nobody has one, so we mine it from the recording itself.',
  },
  {
    title: 'Works when the camera does not',
    body: 'Faces get occluded, people turn away, some are off-camera entirely. The model leans on whichever signal is actually there.',
  },
  {
    title: 'Silence is the target',
    body: 'Separation research optimises for similarity to the original. What you actually need is the other voices gone.',
  },
  {
    title: 'It never invents words',
    body: 'Restoration that sounds clean can fabricate speech. The faithful track is the default and both ship, labelled.',
  },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-16 py-4">
      {/* ---- hero ---- */}
      <section className="flex flex-col gap-6">
        <HeroSlot />

        <div className="flex flex-col gap-4">
          <Badge tone="accent">Audio-visual speaker extraction</Badge>
          <h1 className="font-display text-5xl leading-[1.05] tracking-tight sm:text-6xl">
            Hear one speaker
            <br />
            at a time.
          </h1>
          <p className="max-w-xl text-lg text-fg-secondary">
            Upload a recording where people talk over each other. Pick a speaker. Hear only them —
            in sync with the video, with their own captions.
          </p>
          <div className="flex flex-wrap gap-3">
            <Button size="lg" glow>
              <Link href="/upload">Upload a recording</Link>
            </Button>
            <Button size="lg" variant="secondary">
              <Link href="/demo">See it working first</Link>
            </Button>
          </div>
          <p className="text-sm text-fg-muted">
            The demo runs on a real meeting recording. No account needed.
          </p>
        </div>
      </section>

      {/* ---- the problem ---- */}
      <section className="flex flex-col gap-4">
        <h2 className="text-2xl">Why this is hard</h2>
        <p className="max-w-2xl text-fg-secondary">
          Meeting transcribers give you labelled <em>text</em> and leave the audio mixed. Stem
          separators split music, not people. Research models produce waveforms rather than
          products, and lose track of who is who over a long recording.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {CONTRIBUTIONS.map((c) => (
            <Card key={c.title} className="flex flex-col gap-2">
              <h3 className="font-medium">{c.title}</h3>
              <p className="text-sm text-fg-secondary">{c.body}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* ---- limits, stated up front ---- */}
      <section className="flex flex-col gap-4">
        <h2 className="text-2xl">What it will not do</h2>
        <p className="max-w-2xl text-fg-secondary">Said here rather than discovered later.</p>
        <ul className="flex max-w-2xl flex-col gap-2 text-sm text-fg-secondary">
          {[
            'Real-time or live separation — it works offline, on a finished recording.',
            'More than four speakers, and quality drops noticeably at four.',
            'Music or singing.',
            'Recover someone who is inaudible in the source. It separates; it does not invent.',
            'Identify who someone is, or match a voice across recordings. Deliberately not built.',
          ].map((line) => (
            <li key={line} className="flex gap-2">
              <span aria-hidden className="text-fg-muted">
                —
              </span>
              {line}
            </li>
          ))}
        </ul>
      </section>

      {/* ---- privacy ---- */}
      <section className="flex flex-col gap-3">
        <h2 className="text-2xl">Your recordings</h2>
        <p className="max-w-2xl text-fg-secondary">
          Voiceprints and face crops are deleted when the job finishes. There is no biometric
          database, and cross-recording identification is not implemented — by design, not by
          omission.
        </p>
      </section>
    </div>
  );
}
