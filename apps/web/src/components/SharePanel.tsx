'use client';

/**
 * Create and revoke share links (docs/28 §W7, §D3).
 *
 * The WhatsApp button is a `wa.me` link, which is a real, documented deep link.
 * There is no Instagram button, because a web app cannot post to Instagram —
 * the Graph API needs a business account and app review and does not accept
 * third-party uploads. On mobile the native share sheet offers Instagram
 * anyway, which is the honest route to it.
 */

import { useCallback, useEffect, useState } from 'react';

import { Alert, Badge, Button, Card } from '@/components/ui';
import { api } from '@/lib/store';
import type { ShareResponse } from '@visiovox/ts-client';

export function SharePanel({
  projectId,
  projectTitle,
  speakers,
}: {
  projectId: string;
  projectTitle: string;
  speakers: { ordinal: number; label: string }[];
}) {
  const [shares, setShares] = useState<ShareResponse[]>([]);
  const [speaker, setSpeaker] = useState<string>('all');
  const [fresh, setFresh] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const refresh = useCallback(async () => {
    const list = await api().listShares(projectId);
    setShares(list.items);
  }, [projectId]);

  useEffect(() => {
    void refresh().catch(() => {
      setError('Could not load existing links.');
    });
  }, [refresh]);

  function linkFor(token: string): string {
    return `${window.location.origin}/s/${token}`;
  }

  async function create() {
    setBusy(true);
    setError(null);
    setCopied(false);
    try {
      const row = await api().createShare(projectId, {
        speaker_ordinal: speaker === 'all' ? null : Number(speaker),
        password: null,
        expires_in_days: 30,
      });
      // Shown once: only the hash is stored, so it cannot be recovered later.
      if (row.token) setFresh(linkFor(row.token));
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create the link');
    } finally {
      setBusy(false);
    }
  }

  async function revoke(id: string) {
    setError(null);
    try {
      await api().revokeShare(projectId, id);
      setFresh(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not revoke the link');
    }
  }

  async function copy(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
    } catch {
      setError('Could not reach the clipboard. Select the link and copy it.');
    }
  }

  const live = shares.filter((s) => s.revoked_at === null);

  return (
    <Card className="flex flex-col gap-4">
      <h2 className="text-lg">Share</h2>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="text-fg-secondary">Who to share</span>
          <select
            value={speaker}
            onChange={(e) => {
              setSpeaker(e.target.value);
            }}
            className="rounded-[var(--radius-sm)] border border-border-strong bg-bg-inset px-3 py-2"
          >
            <option value="all">Everyone</option>
            {speakers.map((s) => (
              <option key={s.ordinal} value={String(s.ordinal)}>
                {s.label} only
              </option>
            ))}
          </select>
        </label>
        <Button onClick={() => void create()} loading={busy}>
          Create link
        </Button>
      </div>

      {error !== null && <Alert tone="danger">{error}</Alert>}

      {fresh !== null && (
        <Alert tone="accent" title="Link created">
          <p className="mb-2 break-all font-mono text-xs">{fresh}</p>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="secondary" onClick={() => void copy(fresh)}>
              {copied ? 'Copied' : 'Copy link'}
            </Button>
            <Button size="sm" variant="secondary">
              <a
                href={`https://wa.me/?text=${encodeURIComponent(`${projectTitle} — ${fresh}`)}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                WhatsApp
              </a>
            </Button>
          </div>
          <p className="mt-2 text-xs">
            Copy it now — only a hash is stored, so it cannot be shown again.
          </p>
        </Alert>
      )}

      {live.length > 0 && (
        <ul className="flex flex-col gap-2">
          {live.map((row) => (
            <li
              key={row.id}
              className="flex flex-wrap items-center gap-3 rounded-[var(--radius-sm)] border border-border-subtle px-3 py-2 text-sm"
            >
              <span>
                {row.speaker_ordinal !== null
                  ? `Speaker ${String(row.speaker_ordinal)}`
                  : 'Everyone'}
              </span>
              <Badge>{row.access_count} views</Badge>
              {row.expires_at !== null && (
                <span className="text-xs text-fg-muted">
                  expires {new Date(row.expires_at).toLocaleDateString()}
                </span>
              )}
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto"
                onClick={() => void revoke(row.id)}
              >
                Revoke
              </Button>
            </li>
          ))}
        </ul>
      )}

      <p className="text-xs text-fg-muted">
        Anyone with a link can watch without an account. Revoking takes effect immediately and
        cannot be undone. Shared pages are never indexed by search engines.
      </p>
    </Card>
  );
}
