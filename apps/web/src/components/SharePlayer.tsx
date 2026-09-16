'use client';

/**
 * Client boundary for the player on a public share page.
 *
 * The page itself is a Server Component so its Open Graph tags are real HTML;
 * the player needs Web Audio and the DOM, so it lives behind this boundary.
 */

import { Player } from '@/components/Player';
import type { Manifest } from '@/lib/playback/manifest';

export function SharePlayer({ manifest }: { manifest: unknown }) {
  // The shape is validated by the player's own manifest parsing; a share that
  // arrives malformed should show the page and fail on playback rather than
  // blanking the whole route.
  return <Player manifest={manifest as Manifest} />;
}
