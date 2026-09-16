'use client';

/**
 * Resumable multipart upload from the browser (docs/28 §W2).
 *
 * Built for files far larger than memory and for connections that drop. The
 * whole file is never held at once: `File.slice` produces a view, and the
 * browser streams it straight to object storage. Bytes never pass through the
 * API (ADR-0012).
 *
 * Deliberately NOT here: a whole-file SHA-256. Hashing several gigabytes in the
 * main thread freezes the tab for minutes before a single byte moves, and the
 * object store already returns a per-part ETag that the server composes into an
 * integrity check.
 */

import type { CompletedPart, VisioVoxClient } from '@visiovox/ts-client';

/** Parallel part uploads. Four saturates a domestic connection without
 *  starving the page; more mostly adds retries on a flaky link. */
const CONCURRENCY = 4;
const MAX_ATTEMPTS = 5;
const BASE_BACKOFF_MS = 500;

export type UploadPhase =
  'idle' | 'preparing' | 'uploading' | 'paused' | 'finalising' | 'done' | 'error' | 'aborted';

export interface UploadProgress {
  phase: UploadPhase;
  /** 0–1 over the whole file. */
  fraction: number;
  uploadedBytes: number;
  totalBytes: number;
  /** Bytes per second, smoothed. Null until there is enough signal to mean it. */
  bytesPerSecond: number | null;
  /** Seconds remaining, or null while the rate is still meaningless. */
  secondsRemaining: number | null;
  partsDone: number;
  partsTotal: number;
  error?: string;
}

export interface UploadHandle {
  pause: () => void;
  resume: () => void;
  abort: () => Promise<void>;
}

interface StartOptions {
  client: VisioVoxClient;
  projectId: string;
  file: File;
  onProgress: (progress: UploadProgress) => void;
  signal?: AbortSignal;
}

/**
 * Exponentially-weighted throughput.
 *
 * A plain average over the whole upload makes the ETA stop responding once
 * enough samples accumulate, so a connection that dies reads as healthy for
 * minutes. Weighting recent samples keeps it honest.
 */
class RateMeter {
  private rate: number | null = null;
  private lastAt = performance.now();
  private lastBytes = 0;

  sample(totalBytes: number): number | null {
    const now = performance.now();
    const elapsed = (now - this.lastAt) / 1000;
    // Ignore very short windows: dividing by a few milliseconds produces wild
    // numbers that make the ETA jump around.
    if (elapsed < 0.4) return this.rate;

    const instant = (totalBytes - this.lastBytes) / elapsed;
    this.lastAt = now;
    this.lastBytes = totalBytes;
    this.rate = this.rate === null ? instant : this.rate * 0.7 + instant * 0.3;
    return this.rate;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * PUT one part, retrying with exponential backoff.
 *
 * `XMLHttpRequest` rather than `fetch`, because upload progress events are the
 * only way to show movement within a single 16 MiB part — with fetch, a part is
 * either 0% or 100%, and on a slow link that is a progress bar that sits still
 * for a minute at a time.
 */
function putPart(url: string, body: Blob, onBytes: (delta: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let lastLoaded = 0;

    xhr.upload.addEventListener('progress', (event) => {
      onBytes(event.loaded - lastLoaded);
      lastLoaded = event.loaded;
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const etag = xhr.getResponseHeader('ETag');
        if (etag) resolve(etag);
        else reject(new Error('storage did not return an ETag for this part'));
      } else {
        // Roll back this attempt's bytes so a retry cannot double-count.
        onBytes(-lastLoaded);
        reject(new Error(`part upload failed with ${String(xhr.status)}`));
      }
    });

    xhr.addEventListener('error', () => {
      onBytes(-lastLoaded);
      reject(new Error('network error while uploading a part'));
    });
    xhr.addEventListener('abort', () => {
      onBytes(-lastLoaded);
      reject(new DOMException('aborted', 'AbortError'));
    });

    xhr.open('PUT', url);
    xhr.send(body);
  });
}

export async function startUpload(options: StartOptions): Promise<UploadHandle> {
  const { client, projectId, file, onProgress } = options;

  let paused = false;
  let cancelled = false;
  // Read through functions: TypeScript narrows a plain boolean to its initial
  // value and keeps that narrowing across `await`, then reports every later
  // check as dead code. A call expression is never narrowed.
  const isCancelled = (): boolean => cancelled;
  const isPaused = (): boolean => paused;
  let uploadedBytes = 0;
  const meter = new RateMeter();

  const init = await client.uploadInit(projectId, {
    name: file.name,
    type: file.type,
    size: file.size,
  });

  const partSize = init.part_size_bytes;
  const partCount = init.part_count;
  const done = new Map<number, string>();
  let urls = new Map(init.parts.map((p) => [p.part_number, p.url]));

  function emit(phase: UploadPhase, error?: string): void {
    const rate = meter.sample(uploadedBytes);
    const remaining = file.size - uploadedBytes;
    onProgress({
      phase,
      fraction: file.size === 0 ? 0 : Math.min(1, uploadedBytes / file.size),
      uploadedBytes,
      totalBytes: file.size,
      bytesPerSecond: rate,
      // Only once the rate means something: an ETA computed from the first
      // half-second of an upload is a guess presented as a fact.
      secondsRemaining: rate !== null && rate > 0 ? remaining / rate : null,
      partsDone: done.size,
      partsTotal: partCount,
      ...(error === undefined ? {} : { error }),
    });
  }

  emit('preparing');

  /** Ask the server for the next batch, handing it what we have finished. */
  async function syncBatch(): Promise<void> {
    const completed: CompletedPart[] = [...done.entries()].map(([part_number, etag]) => ({
      part_number,
      etag,
    }));
    const next = await client.uploadParts(projectId, init.upload_id, completed, 0);
    urls = new Map(next.parts.map((p) => [p.part_number, p.url]));
  }

  async function uploadPart(partNumber: number): Promise<void> {
    const start = (partNumber - 1) * partSize;
    const blob = file.slice(start, Math.min(start + partSize, file.size));

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
      if (isCancelled()) throw new DOMException('aborted', 'AbortError');

      let url = urls.get(partNumber);
      if (url === undefined) {
        await syncBatch();
        url = urls.get(partNumber);
        if (url === undefined) throw new Error(`no upload URL for part ${String(partNumber)}`);
      }

      try {
        const etag = await putPart(url, blob, (delta) => {
          uploadedBytes += delta;
          emit('uploading');
        });
        done.set(partNumber, etag);
        return;
      } catch (err) {
        if (isCancelled() || (err instanceof DOMException && err.name === 'AbortError')) throw err;
        if (attempt === MAX_ATTEMPTS) throw err;
        // A presigned URL may simply have expired mid-upload; dropping it forces
        // a fresh one on the next attempt rather than retrying a dead link.
        urls.delete(partNumber);
        await sleep(BASE_BACKOFF_MS * 2 ** (attempt - 1));
      }
    }
  }

  const run = (async () => {
    // Resume: the server knows which parts it already holds, so a reload costs
    // nothing instead of restarting a multi-gigabyte transfer.
    const status = await client.uploadStatus(projectId, init.upload_id);
    for (const n of status.completed_parts) {
      done.set(n, 'resumed');
      uploadedBytes += Math.min(partSize, file.size - (n - 1) * partSize);
    }

    const queue: number[] = [];
    for (let n = 1; n <= partCount; n += 1) if (!done.has(n)) queue.push(n);

    emit('uploading');

    async function worker(): Promise<void> {
      for (;;) {
        while (isPaused() && !isCancelled()) await sleep(200);
        if (isCancelled()) throw new DOMException('aborted', 'AbortError');

        const next = queue.shift();
        if (next === undefined) return;
        await uploadPart(next);

        // Checkpoint each batch so a crash loses at most one batch of work.
        if (done.size % 10 === 0) await syncBatch();
      }
    }

    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, worker));

    emit('finalising');
    const parts: CompletedPart[] = [...done.entries()]
      .filter(([, etag]) => etag !== 'resumed')
      .map(([part_number, etag]) => ({ part_number, etag }));

    await client.uploadComplete(projectId, init.upload_id, init.key, parts);
    emit('done');
  })();

  run.catch((err: unknown) => {
    if (isCancelled()) emit('aborted');
    else emit('error', err instanceof Error ? err.message : 'Upload failed');
  });

  return {
    pause: () => {
      paused = true;
      emit('paused');
    },
    resume: () => {
      paused = false;
      emit('uploading');
    },
    abort: async () => {
      cancelled = true;
      try {
        await client.uploadAbort(projectId, init.upload_id);
      } finally {
        emit('aborted');
      }
    },
  };
}

// ------------------------------------------------------------- formatting --

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${String(bytes)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 || unit === 0 ? 0 : 1)} ${units[unit] ?? 'TB'}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `${String(Math.ceil(seconds))}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s === 0 ? `${String(m)} min` : `${String(m)} min ${String(s)}s`;
  const h = Math.floor(m / 60);
  return `${String(h)}h ${String(m % 60)} min`;
}

/**
 * Read duration and dimensions without uploading anything.
 *
 * This is what lets the upload screen say "18 minutes, ready by 14:35" before a
 * single byte moves. The server re-derives it authoritatively from a sandboxed
 * probe; this is for the user, not for admission.
 */
export function inspectLocally(
  file: File,
): Promise<{ durationSeconds: number | null; width: number | null; height: number | null }> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';

    const finish = (result: {
      durationSeconds: number | null;
      width: number | null;
      height: number | null;
    }): void => {
      URL.revokeObjectURL(url);
      resolve(result);
    };

    video.addEventListener('loadedmetadata', () => {
      finish({
        durationSeconds: Number.isFinite(video.duration) ? video.duration : null,
        width: video.videoWidth || null,
        height: video.videoHeight || null,
      });
    });
    // A container the browser cannot open is not necessarily one ffmpeg cannot:
    // resolve with nulls and let the server decide.
    video.addEventListener('error', () => {
      finish({ durationSeconds: null, width: null, height: null });
    });

    video.src = url;
  });
}
