/**
 * /styleguide — every primitive, both themes, one page.
 *
 * Exists so a visual regression is visible in one place rather than discovered
 * on a page someone happened to open. The theme toggle writes `data-theme` on
 * <html>, which is the same switch the app uses, so light mode is exercised
 * here rather than assumed to work.
 */
'use client';

import { useState } from 'react';

import {
  Alert,
  Badge,
  Button,
  Card,
  Field,
  Progress,
  Skeleton,
  SpeakerDot,
  Spinner,
} from '@/components/ui';

const SPEAKERS = ['--spk-1', '--spk-2', '--spk-3', '--spk-4', '--spk-mixed'];
const NEUTRALS = ['--bg-base', '--bg-surface', '--bg-elevated', '--bg-inset'];
const SEMANTIC = ['--accent', '--success', '--warning', '--danger', '--contested'];

function Swatches({ title, tokens }: { title: string; tokens: string[] }) {
  return (
    <div>
      <h3 className="mb-2 text-sm text-fg-secondary">{title}</h3>
      <div className="flex flex-wrap gap-2">
        {tokens.map((t) => (
          <div key={t} className="w-28">
            <div
              className="h-12 rounded-[var(--radius-sm)] border border-border-subtle"
              style={{ background: `var(${t})` }}
            />
            <code className="mt-1 block text-[0.68rem] text-fg-muted">{t}</code>
          </div>
        ))}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-4">
      <h2 className="text-lg font-medium">{title}</h2>
      {children}
    </section>
  );
}

export default function StyleguidePage() {
  const [light, setLight] = useState(false);
  const [progress, setProgress] = useState(42);

  function toggleTheme() {
    const next = !light;
    setLight(next);
    document.documentElement.setAttribute('data-theme', next ? 'light' : 'dark');
  }

  return (
    <div className="flex flex-col gap-10 py-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-medium">Design system</h1>
          <p className="text-sm text-fg-muted">
            docs/14. Contrast is enforced by <code>scripts/check-contrast.mjs</code>, not by review.
          </p>
        </div>
        <Button variant="secondary" onClick={toggleTheme}>
          {light ? 'Dark' : 'Light'} theme
        </Button>
      </header>

      <Section title="Colour">
        <Swatches title="Neutrals" tokens={NEUTRALS} />
        <Swatches title="Speakers — equal lightness, so none reads as louder" tokens={SPEAKERS} />
        <Swatches title="Semantic" tokens={SEMANTIC} />
      </Section>

      <Section title="Buttons">
        <div className="flex flex-wrap items-center gap-3">
          <Button>Primary</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="danger">Danger</Button>
          <Button loading>Working</Button>
          <Button disabled>Disabled</Button>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm">Small</Button>
          <Button size="md">Medium</Button>
          <Button size="lg" glow>
            Large, expressive
          </Button>
        </div>
        <p className="text-xs text-fg-muted">
          <code>glow</code> is for landing, auth, upload and waiting. In the player it is a bug.
        </p>
      </Section>

      <Section title="Fields">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Email" type="email" placeholder="you@example.com" />
          <Field label="Password" type="password" hint="At least 12 characters." />
          <Field label="Title" defaultValue="Board meeting" />
          <Field label="Broken" defaultValue="nope" error="That title is already in use." />
        </div>
      </Section>

      <Section title="Progress">
        <Progress value={progress} label="Upload" />
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              setProgress((p) => Math.max(0, p - 10));
            }}
          >
            −10
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              setProgress((p) => Math.min(100, p + 10));
            }}
          >
            +10
          </Button>
          <span className="tabular text-sm text-fg-muted">{progress}%</span>
        </div>
        <div>
          <p className="mb-2 text-sm text-fg-secondary">
            Indeterminate — used when the estimate is genuinely unknown
          </p>
          <Progress label="Working" />
        </div>
      </Section>

      <Section title="Status">
        <div className="flex flex-wrap items-center gap-3">
          <Badge>Queued</Badge>
          <Badge tone="accent">Processing</Badge>
          <Badge tone="success">Ready</Badge>
          <Badge tone="warning">Low confidence</Badge>
          <Badge tone="danger">Failed</Badge>
          <Spinner label="Loading" />
        </div>
        <Alert tone="danger" title="Upload refused">
          This file is 900.0 GB and the current limit is 50.0 GB. The limit reflects free space and
          uploads already in progress, so it may rise shortly.
        </Alert>
        <Alert tone="warning" title="Visual track unavailable">
          Two speakers were off camera, so their audio was separated without lip information.
        </Alert>
      </Section>

      <Section title="Speakers">
        <div className="flex flex-wrap gap-3">
          {[0, 1, 2, 3].map((i) => (
            <Card key={i} className="flex min-w-40 items-center gap-2 py-3">
              <SpeakerDot index={i} />
              <span className="text-sm">Speaker {i + 1}</span>
              <span className="tabular ml-auto text-xs text-fg-muted">{[41, 28, 19, 12][i]}%</span>
            </Card>
          ))}
        </div>
        <p className="text-xs text-fg-muted">
          The dot never appears without its label — colour is never the only identifier.
        </p>
      </Section>

      <Section title="Loading">
        <div className="flex flex-col gap-2">
          <Skeleton className="h-6 w-1/3" />
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      </Section>

      <Section title="Typography">
        <p className="font-display text-4xl">Hear one speaker at a time</p>
        <p className="text-base">
          Body copy at the default size. Timecodes and metrics use tabular figures so digits do not
          jitter as they change: <span className="mono tabular">00:14:32.480</span>
        </p>
        <p className="text-sm text-fg-secondary">Secondary text.</p>
        <p className="text-sm text-fg-muted">Muted text.</p>
      </Section>
    </div>
  );
}
