'use client';

import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import type { JobResponse, ProjectResponse } from '@visiovox/ts-client';
import { api } from '@/lib/store';
import { Player } from '@/components/Player';
import type { Manifest } from '@/lib/playback/manifest';

interface Progress {
  status: string;
  progress: number;
  stage: string | null;
}

const TERMINAL = new Set(['succeeded', 'partial', 'failed', 'cancelled']);

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Read through a function: TypeScript narrows a plain boolean (or an
    // object property) to `false` after the first check and keeps that
    // narrowing across `await`, then reports every later check as dead.
    // A call expression is never narrowed, and the effect cleanup below is
    // what actually flips it.
    let cancelled = false;
    const stopped = () => cancelled;

    async function poll() {
      try {
        const p = await api().getProject(projectId);
        if (stopped()) return;
        setProject(p);

        const j: JobResponse | null = await api()
          .getJob(projectId)
          .catch(() => null);
        if (stopped() || j === null) return;
        setJob(j);
        setProgress({ status: j.status, progress: j.progress, stage: null });

        // Poll rather than SSE here: EventSource cannot attach an
        // Authorization header, and proxying the stream is Phase 6 work.
        // The mock finishes in seconds, so a 1 s tick is not a real cost yet.
        if (!TERMINAL.has(j.status)) {
          setTimeout(() => void poll(), 1000);
        }
      } catch (err) {
        if (!stopped()) setError(err instanceof Error ? err.message : 'Failed to load');
      }
    }

    void poll();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // The manifest carries signed URLs with an expiry, so it is fetched once the
  // project reaches `ready` rather than alongside the project record — there is
  // no point holding URLs that lapse before anyone presses play.
  useEffect(() => {
    if (project?.status !== 'ready') return;
    let cancelled = false;
    void api()
      .getManifest(projectId)
      .then((body) => {
        if (!cancelled) setManifest(body as Manifest);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load manifest');
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, project?.status]);

  if (error !== null) return <p className="error">{error}</p>;
  if (!project) return <p className="muted">Loading…</p>;

  const pct = progress?.progress ?? 0;

  return (
    <div className="stack">
      <div className="panel stack">
        <div className="row">
          <h1 style={{ margin: 0 }}>{project.title}</h1>
          <span className={`badge${project.status === 'ready' ? ' ready' : ''}`}>
            {project.status}
          </span>
        </div>
        <div className="bar">
          <span style={{ width: `${String(pct)}%` }} />
        </div>
        <p className="muted">
          {project.status === 'ready'
            ? `${String(project.speaker_count ?? 0)} speakers · overlap ${
                project.overlap_ratio !== null
                  ? `${String(Math.round(project.overlap_ratio * 100))}%`
                  : '—'
              } · ${project.difficulty ?? 'unrated'}`
            : `Processing — ${String(pct)}%`}
        </p>
        {project.warnings.length > 0 && (
          <ul className="muted">
            {project.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </div>

      {manifest !== null && <Player manifest={manifest} />}

      {job && (
        <div className="panel">
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Status</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {job.stages.map((s) => (
                <tr key={s.stage}>
                  <td>{s.stage}</td>
                  <td>{s.status}</td>
                  <td>{s.duration_ms !== null ? `${String(s.duration_ms)} ms` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
