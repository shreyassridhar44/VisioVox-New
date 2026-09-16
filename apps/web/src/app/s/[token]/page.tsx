/**
 * /s/[token] — the public share page (docs/28 §W7).
 *
 * A Server Component so the Open Graph tags are real HTML: an unfurl in
 * WhatsApp or Slack is fetched by a bot that runs no JavaScript, so metadata
 * rendered on the client is metadata nobody sees.
 *
 * Deliberately `noindex`. A link someone sent to three people should not turn
 * up in a search result, and `robots` here is belt to the API's `X-Robots-Tag`
 * braces.
 */

import type { Metadata } from 'next';

import { SharePlayer } from '@/components/SharePlayer';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface SharePayload {
  title: string;
  duration_ms: number | null;
  speaker_count: number;
  speaker_ordinal: number | null;
  manifest: unknown;
}

async function load(token: string): Promise<SharePayload | null> {
  try {
    const response = await fetch(`${API}/v1/shared/${encodeURIComponent(token)}`, {
      // Never cached: revocation has to take effect immediately, and a cached
      // copy of a revoked share is precisely the failure this must not have.
      cache: 'no-store',
    });
    if (!response.ok) return null;
    return (await response.json()) as SharePayload;
  } catch {
    return null;
  }
}

function minutes(ms: number | null): string {
  if (ms === null) return '';
  const m = Math.round(ms / 60000);
  return m <= 1 ? '1 minute' : `${String(m)} minutes`;
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ token: string }>;
}): Promise<Metadata> {
  const { token } = await params;
  const data = await load(token);

  if (!data) {
    return { title: 'Link unavailable', robots: { index: false, follow: false } };
  }

  const who =
    data.speaker_ordinal !== null
      ? `Speaker ${String(data.speaker_ordinal)} isolated`
      : `${String(data.speaker_count)} speakers, separated`;
  const description = `${who}${data.duration_ms ? ` · ${minutes(data.duration_ms)}` : ''}`;

  return {
    title: `${data.title} — VisioVox`,
    description,
    robots: { index: false, follow: false },
    openGraph: {
      title: data.title,
      description,
      type: 'video.other',
    },
    twitter: { card: 'summary_large_image', title: data.title, description },
  };
}

export default async function SharePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  const data = await load(token);

  if (!data) {
    return (
      <div className="mx-auto flex max-w-lg flex-col gap-3 py-16 text-center">
        <h1 className="font-display text-3xl">This link is not available</h1>
        <p className="text-fg-secondary">
          It may have been revoked by whoever shared it, or it may have expired. Links are not
          recoverable once revoked.
        </p>
        <p className="text-sm text-fg-muted">
          <a href="/" className="text-accent">
            What is VisioVox?
          </a>
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-5 py-6">
      <header className="flex flex-col gap-1">
        <p className="text-sm text-fg-muted">Shared with you</p>
        <h1 className="font-display text-3xl">{data.title}</h1>
        <p className="text-sm text-fg-secondary">
          {data.speaker_ordinal !== null
            ? `Speaker ${String(data.speaker_ordinal)}, isolated from the other voices`
            : `${String(data.speaker_count)} speakers, each separated onto their own track`}
          {data.duration_ms ? ` · ${minutes(data.duration_ms)}` : ''}
        </p>
      </header>

      <SharePlayer manifest={data.manifest} />

      <footer className="flex flex-col gap-2 border-t border-border-subtle pt-4 text-sm text-fg-muted">
        <p>
          Separated with{' '}
          <a href="/" className="text-accent">
            VisioVox
          </a>{' '}
          — upload a recording where people talk over each other, and hear one of them at a time.
        </p>
        <p className="text-xs">
          Whoever shared this can revoke the link at any time. Voiceprints and face crops from the
          original recording were deleted when it finished processing.
        </p>
      </footer>
    </div>
  );
}
