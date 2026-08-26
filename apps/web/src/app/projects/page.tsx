'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import type { ProjectResponse } from '@visiovox/ts-client';
import { api } from '@/lib/store';

export default function ProjectsPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const list = await api().listProjects();
      setProjects(list.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load projects');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function upload(file: File) {
    setBusy(true);
    setError(null);
    try {
      const project = await api().createProject(file.name);
      const init = await api().uploadInit(project.id, {
        name: file.name,
        type: file.type,
        size: file.size,
      });
      const part = init.parts[0];
      if (!part) throw new Error('no upload URL returned');

      const put = await fetch(part.url, { method: 'PUT', body: file });
      if (!put.ok) throw new Error(`upload failed (${String(put.status)})`);
      const etag = put.headers.get('ETag') ?? '';

      await api().uploadComplete(project.id, init.upload_id, init.key, [{ part_number: 1, etag }]);
      router.push(`/projects/${project.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div className="panel stack">
        <h1>Your projects</h1>
        <label htmlFor="file">Upload a recording</label>
        <input
          id="file"
          type="file"
          accept="video/*,audio/*"
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
          }}
        />
        {busy && <p className="muted">Uploading…</p>}
        {error !== null && <p className="error">{error}</p>}
      </div>

      <div className="panel">
        {projects.length === 0 ? (
          <p className="muted">Nothing here yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Status</th>
                <th>Speakers</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => (
                <tr key={p.id}>
                  <td>
                    <a href={`/projects/${p.id}`}>{p.title}</a>
                  </td>
                  <td>
                    <span className={`badge${p.status === 'ready' ? ' ready' : ''}`}>
                      {p.status}
                    </span>
                  </td>
                  <td>{p.speaker_count ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
