'use client';

/**
 * Upload — the expressive surface (docs/28 §D1).
 *
 * The brief: tell the user what they can upload *before* they choose, and what
 * will happen *before* they commit. Both numbers are honest — the limit is
 * fetched live because it moves with free space and other uploads in flight,
 * and the processing estimate is shown as a range and labelled as an estimate,
 * because the pipeline has not been benchmarked yet (docs/track-w W5).
 */

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';

import { Alert, Badge, Button, Card, Progress } from '@/components/ui';
import {
  formatBytes,
  formatDuration,
  inspectLocally,
  startUpload,
  type UploadHandle,
  type UploadProgress,
} from '@/lib/upload/resumable';
import { api, useSession } from '@/lib/store';
import type { LimitsResponse } from '@visiovox/ts-client';

interface Inspected {
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
}

/**
 * Rough processing estimate.
 *
 * Deliberately a RANGE, and deliberately wide. Real per-stage timings are not
 * measured until the pipeline runs for real (W8), and a single confident number
 * derived from a guess is worse than an honest interval — people plan around
 * the number you show them.
 */
function estimateProcessing(durationSeconds: number | null): [number, number] | null {
  if (durationSeconds === null) return null;
  return [durationSeconds * 0.5, durationSeconds * 1.4];
}

export default function UploadPage() {
  const router = useRouter();
  const signedIn = useSession((s) => s.signedIn);
  const ready = useSession((s) => s.ready);
  const hydrate = useSession((s) => s.hydrate);

  const [limits, setLimits] = useState<LimitsResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [inspected, setInspected] = useState<Inspected | null>(null);
  const [title, setTitle] = useState('');
  const [attested, setAttested] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const handleRef = useRef<UploadHandle | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (ready && !signedIn) router.replace('/login');
  }, [ready, signedIn, router]);

  useEffect(() => {
    if (!signedIn) return;
    void api()
      .getLimits()
      .then(setLimits)
      .catch(() => {
        // A missing limits call should not block the page; the server still
        // enforces the real limit at init.
      });
  }, [signedIn]);

  const choose = useCallback((picked: File) => {
    setFile(picked);
    setError(null);
    setTitle((current) => current || picked.name.replace(/\.[^.]+$/, ''));
    setInspected(null);
    void inspectLocally(picked).then(setInspected);
  }, []);

  function onDrop(event: React.DragEvent) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped) choose(dropped);
  }

  async function begin() {
    if (!file) return;
    setError(null);
    try {
      const project = await api().createProject(title || file.name);
      handleRef.current = await startUpload({
        client: api(),
        projectId: project.id,
        file,
        onProgress: (p) => {
          setProgress(p);
          if (p.phase === 'done') router.push(`/projects/${project.id}`);
          if (p.phase === 'error') setError(p.error ?? 'Upload failed');
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start the upload');
    }
  }

  const maxBytes = limits?.max_upload_bytes ?? null;
  const maxSeconds = limits?.max_duration_seconds ?? null;
  const localDuration = inspected?.durationSeconds ?? null;

  const overBy = file !== null && maxBytes !== null && file.size > maxBytes ? maxBytes : null;
  const longBy =
    localDuration !== null && maxSeconds !== null && localDuration > maxSeconds ? maxSeconds : null;
  const tooBig = overBy !== null;
  const tooLong = longBy !== null;
  const busy =
    progress !== null && ['uploading', 'preparing', 'finalising'].includes(progress.phase);
  const estimate = estimateProcessing(localDuration);

  if (!ready) return null;

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 py-6">
      <header className="flex flex-col gap-1">
        <h1 className="font-display text-4xl">Upload a recording</h1>
        <p className="text-fg-secondary">
          A meeting, an interview, anything where people talk over each other.
        </p>
      </header>

      {limits && (
        <p className="text-sm text-fg-muted">
          Up to{' '}
          <strong className="tabular text-fg-primary">
            {formatBytes(limits.max_upload_bytes)}
          </strong>{' '}
          per file &middot; up to{' '}
          <strong className="tabular text-fg-primary">
            {Math.round(limits.max_duration_seconds / 60)} minutes
          </strong>{' '}
          of video. Based on free space and what is processing right now.
        </p>
      )}

      {/* ---- drop zone ---- */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => {
          setDragging(false);
        }}
        onDrop={onDrop}
        className={[
          'relative overflow-hidden rounded-[var(--radius-lg)] border-2 border-dashed p-10 text-center',
          'transition-[border-color,transform,box-shadow] duration-(--dur-normal) ease-(--ease-out)',
          dragging
            ? 'border-accent shadow-[var(--glow-accent)] scale-[1.01]'
            : 'border-border-strong hover:border-accent/60',
        ].join(' ')}
        style={{
          backgroundImage: dragging
            ? 'radial-gradient(60% 60% at 50% 40%, var(--mesh-1), transparent 70%)'
            : undefined,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*,audio/*"
          className="sr-only"
          onChange={(e) => {
            const picked = e.target.files?.[0];
            if (picked) choose(picked);
          }}
        />
        {file === null ? (
          <div className="flex flex-col items-center gap-3">
            <p className="text-lg">Drop a video here</p>
            <p className="text-sm text-fg-muted">or</p>
            <Button
              variant="secondary"
              onClick={() => {
                inputRef.current?.click();
              }}
            >
              Choose a file
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <p className="text-lg break-all">{file.name}</p>
            <p className="tabular text-sm text-fg-muted">
              {formatBytes(file.size)}
              {inspected?.durationSeconds != null &&
                ` · ${formatDuration(inspected.durationSeconds)}`}
              {inspected?.width != null &&
                ` · ${String(inspected.width)}×${String(inspected.height ?? 0)}`}
            </p>
            {!busy && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setFile(null);
                  setInspected(null);
                  setProgress(null);
                }}
              >
                Choose a different file
              </Button>
            )}
          </div>
        )}
      </div>

      {/* ---- what will happen, before committing ---- */}
      {file && !busy && (
        <Card className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-fg-secondary">Before you start</span>
            {tooBig || tooLong ? (
              <Badge tone="danger">Cannot process</Badge>
            ) : (
              <Badge tone="success">Ready</Badge>
            )}
          </div>

          {overBy !== null && (
            <Alert tone="danger" title="This file is too large right now">
              It is {formatBytes(file.size)} and the current limit is {formatBytes(overBy)}. The
              limit reflects free space and uploads already in progress, so it may rise shortly.
            </Alert>
          )}

          {longBy !== null && localDuration !== null && (
            <Alert tone="danger" title="This recording is too long">
              It runs {formatDuration(localDuration)} and the limit is {Math.round(longBy / 60)}{' '}
              minutes. Processing time scales with length, so please trim it first.
            </Alert>
          )}

          {!tooBig && !tooLong && (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
              <dt className="text-fg-muted">Upload</dt>
              <dd className="tabular">depends on your connection</dd>
              <dt className="text-fg-muted">Processing</dt>
              <dd className="tabular">
                {estimate
                  ? `about ${formatDuration(estimate[0])}–${formatDuration(estimate[1])}`
                  : 'estimated once we can read the file'}
              </dd>
            </dl>
          )}

          <p className="text-xs text-fg-muted">
            Processing time is an estimate and will get more accurate as we measure real runs.
          </p>

          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              checked={attested}
              onChange={(e) => {
                setAttested(e.target.checked);
              }}
              className="mt-1"
            />
            <span>I have the right to upload this recording and to process the voices in it.</span>
          </label>

          <Button
            size="lg"
            glow
            disabled={!attested || tooBig || tooLong}
            onClick={() => {
              void begin();
            }}
          >
            Start processing
          </Button>
        </Card>
      )}

      {/* ---- in flight ---- */}
      {progress && busy && (
        <Card className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <span className="text-sm text-fg-secondary">
              {progress.phase === 'finalising' ? 'Finishing up' : 'Uploading'}
            </span>
            <span className="tabular text-sm">
              {formatBytes(progress.uploadedBytes)} / {formatBytes(progress.totalBytes)}
            </span>
          </div>

          <Progress value={progress.fraction * 100} label="Upload progress" />

          <div className="flex items-center justify-between text-xs text-fg-muted">
            <span className="tabular">
              {progress.bytesPerSecond != null
                ? `${formatBytes(progress.bytesPerSecond)}/s`
                : 'measuring speed…'}
              {progress.secondsRemaining != null &&
                ` · ${formatDuration(progress.secondsRemaining)} left`}
            </span>
            <span className="tabular">
              part {progress.partsDone} of {progress.partsTotal}
            </span>
          </div>

          <div className="flex gap-2">
            {progress.phase === 'paused' ? (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  handleRef.current?.resume();
                }}
              >
                Resume
              </Button>
            ) : (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  handleRef.current?.pause();
                }}
              >
                Pause
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                void handleRef.current?.abort().then(() => {
                  setProgress(null);
                  setFile(null);
                });
              }}
            >
              Cancel
            </Button>
          </div>

          <p className="text-xs text-fg-muted">
            You can close this tab — the upload resumes from where it stopped when you come back.
          </p>
        </Card>
      )}

      {error !== null && (
        <Alert tone="danger" title="Upload problem">
          {error}
        </Alert>
      )}
    </div>
  );
}
