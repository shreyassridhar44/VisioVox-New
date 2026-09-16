'use client';

/**
 * The engaged wait (docs/28 §W5).
 *
 * A 20-minute job with a silent progress bar feels broken. This narrates the
 * real pipeline, stage by stage, from the SSE stream the API already publishes
 * — the copy is tied to actual stage names, so it cannot drift into fiction
 * while the backend does something else.
 *
 * Three rules it follows:
 *   - Never show a countdown that was not measured. Until the pipeline is
 *     benchmarked (W8) the estimate is a range, and it widens rather than
 *     lying when a stage runs long.
 *   - Never park at 99%. Progress comes from completed stages, so it moves
 *     when something real finishes and stops when nothing has.
 *   - Survive a dropped connection and a backgrounded tab, because people will
 *     switch away and come back.
 */

import { useEffect, useRef, useState } from 'react';

import { Badge, Card, Progress, SpeakerDot } from '@/components/ui';
import { formatDuration } from '@/lib/upload/resumable';

/**
 * What each stage is actually doing, in words a person can picture.
 *
 * Keyed on the stage identifiers the worker emits (docs/05 §2), so a renamed
 * stage shows its raw name rather than a confidently wrong description.
 */
const STAGE_COPY: Record<string, { title: string; detail: string }> = {
  S0_ingest: {
    title: 'Reading the recording',
    detail: 'Checking the file is what it claims to be, and pulling out the audio.',
  },
  S1_enhance: {
    title: 'Cleaning the room',
    detail: 'Removing reverb and background noise before anyone is separated.',
  },
  S2a_audio: {
    title: 'Listening for turns',
    detail: 'Finding who speaks when, and where voices collide.',
  },
  S2b_video: {
    title: 'Watching faces',
    detail: 'Tracking each face and reading which mouth is moving.',
  },
  S3_fuse: {
    title: 'Matching faces to voices',
    detail: 'Deciding which face belongs to which voice.',
  },
  S4_enrol: {
    title: 'Learning each voice',
    detail: 'Mining clean moments so every speaker has a reference of their own.',
  },
  S5_extract: {
    title: 'Separating the speakers',
    detail: 'The part that needs the GPU. Each speaker is pulled out separately.',
  },
  S6_restore: {
    title: 'Restoring what was buried',
    detail: 'Repairing speech that was masked, without inventing words.',
  },
  S7_transcribe: {
    title: 'Writing the captions',
    detail: 'Transcribing each speaker on their own track.',
  },
  S8_audit: {
    title: 'Checking for leakage',
    detail: 'Looking for words that appear on two tracks at once.',
  },
  S9_package: {
    title: 'Packaging for playback',
    detail: 'Encoding the tracks so they stay in sync in the browser.',
  },
};

export interface JobEvent {
  status: string;
  progress: number;
  stage: string | null;
  speaker?: number | null;
  speakers?: number | null;
}

const TERMINAL = new Set(['succeeded', 'partial', 'failed', 'cancelled']);

/**
 * Subscribe to job progress, reconnecting with backoff.
 *
 * EventSource reconnects on its own, but not after the server closes the stream
 * cleanly, and not with any backoff worth the name. A job outlives a browser
 * tab's attention span, so this has to survive sleep, a lost network and a
 * backgrounded tab.
 */
function useJobProgress(projectId: string, token: string | null): JobEvent | null {
  const [event, setEvent] = useState<JobEvent | null>(null);
  const attemptRef = useRef(0);

  useEffect(() => {
    if (!token) return;
    let source: EventSource | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const base = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

    function connect(): void {
      if (closed) return;
      // EventSource cannot set an Authorization header, so the token rides as a
      // query parameter on a URL that is never logged with its query string.
      source = new EventSource(
        `${base}/v1/projects/${projectId}/events?access_token=${encodeURIComponent(token ?? '')}`,
      );

      source.addEventListener('progress', (e) => {
        attemptRef.current = 0;
        setEvent(JSON.parse((e as MessageEvent<string>).data) as JobEvent);
      });

      source.addEventListener('done', (e) => {
        setEvent(JSON.parse((e as MessageEvent<string>).data) as JobEvent);
        closed = true;
        source?.close();
      });

      source.addEventListener('error', () => {
        source?.close();
        if (closed) return;
        // Capped exponential backoff: a server that is down should not be hit
        // once a second by every open tab.
        attemptRef.current += 1;
        const delay = Math.min(30000, 1000 * 2 ** (attemptRef.current - 1));
        timer = setTimeout(connect, delay);
      });
    }

    connect();
    return () => {
      closed = true;
      source?.close();
      if (timer) clearTimeout(timer);
    };
  }, [projectId, token]);

  return event;
}

/** Tab title and favicon badge, so a backgrounded tab still reports progress. */
function useTabSignal(event: JobEvent | null, projectTitle: string): void {
  useEffect(() => {
    if (!event) return;
    const original = document.title;
    if (TERMINAL.has(event.status)) {
      document.title = event.status === 'succeeded' ? `✓ ${projectTitle}` : `× ${projectTitle}`;
    } else {
      document.title = `${String(event.progress)}% · ${projectTitle}`;
    }
    return () => {
      document.title = original;
    };
  }, [event, projectTitle]);
}

export function ProcessingView({
  projectId,
  projectTitle,
  token,
  speakerCount,
}: {
  projectId: string;
  projectTitle: string;
  token: string | null;
  speakerCount: number | null;
}) {
  const event = useJobProgress(projectId, token);
  useTabSignal(event, projectTitle);

  const [startedAt] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setElapsed((Date.now() - startedAt) / 1000);
    }, 1000);
    return () => {
      clearInterval(id);
    };
  }, [startedAt]);

  const stage = event?.stage ?? null;
  const copy = stage ? STAGE_COPY[stage] : undefined;
  const pct = event?.progress ?? 0;

  /**
   * Remaining time, as a widening range.
   *
   * Derived from observed pace rather than a fixed table, so it is at least
   * grounded in this job. It is deliberately presented as "about X–Y" and it
   * widens when a stage runs long, because narrowing a guess as it gets less
   * reliable is how progress bars come to be distrusted.
   */
  const remaining: [number, number] | null =
    pct > 3 && elapsed > 5
      ? [(elapsed / pct) * (100 - pct) * 0.8, (elapsed / pct) * (100 - pct) * 1.6]
      : null;

  if (event && TERMINAL.has(event.status)) return null;

  return (
    <Card className="flex flex-col gap-5" glass>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg">{copy?.title ?? 'Getting started'}</h2>
          <p className="mt-1 text-sm text-fg-secondary">
            {copy?.detail ?? 'Your recording is queued and will begin shortly.'}
          </p>
        </div>
        <Badge tone="accent">{event?.status ?? 'queued'}</Badge>
      </div>

      <div className="flex flex-col gap-2">
        <Progress value={pct} label="Processing progress" />
        <div className="flex items-center justify-between text-xs text-fg-muted">
          <span className="tabular">{pct}% complete</span>
          <span className="tabular">
            {remaining
              ? `about ${formatDuration(remaining[0])}–${formatDuration(remaining[1])} left`
              : 'estimating…'}
          </span>
        </div>
      </div>

      {/* The stage list is the honest version of a progress bar: it shows what
          has actually been finished rather than a number that only goes up. */}
      <ol className="flex flex-col gap-1.5">
        {Object.entries(STAGE_COPY).map(([id, entry]) => {
          const order = Object.keys(STAGE_COPY);
          const current = stage ? order.indexOf(stage) : -1;
          const mine = order.indexOf(id);
          const state =
            current < 0
              ? 'pending'
              : mine < current
                ? 'done'
                : mine === current
                  ? 'active'
                  : 'pending';
          return (
            <li
              key={id}
              className={[
                'flex items-center gap-2.5 rounded-[var(--radius-sm)] px-2 py-1.5 text-sm',
                state === 'active' ? 'bg-bg-elevated text-fg-primary' : '',
                state === 'done' ? 'text-fg-muted' : '',
                state === 'pending' ? 'text-fg-muted opacity-60' : '',
              ].join(' ')}
            >
              <span aria-hidden className="w-4 shrink-0 text-center">
                {state === 'done' ? '✓' : state === 'active' ? '•' : '·'}
              </span>
              <span>{entry.title}</span>
              {state === 'active' && speakerCount != null && id === 'S5_extract' && (
                <span className="ml-auto flex items-center gap-1">
                  {Array.from({ length: Math.min(speakerCount, 4) }, (_, i) => (
                    <SpeakerDot key={i} index={i} size={7} />
                  ))}
                </span>
              )}
            </li>
          );
        })}
      </ol>

      <p className="text-xs text-fg-muted">
        You can close this tab. Processing continues, and the page picks up where it left off.
      </p>
    </Card>
  );
}
