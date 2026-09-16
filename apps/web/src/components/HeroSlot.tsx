'use client';

/**
 * Client boundary for the 3D hero.
 *
 * The App Router forbids `ssr: false` inside a Server Component, and the scene
 * genuinely cannot be server-rendered — it needs a WebGL context. So the
 * dynamic import lives here, in the smallest possible client island, and the
 * landing page around it stays a Server Component.
 *
 * The placeholder reserves the hero's exact height. Without it the text below
 * jumps when the chunk lands, which is both unpleasant and a CLS penalty.
 */

import dynamic from 'next/dynamic';

const RibbonHero = dynamic(() => import('./RibbonHero').then((m) => m.RibbonHero), {
  ssr: false,
  loading: () => (
    <div
      aria-hidden
      className="h-[52vh] min-h-72 w-full rounded-[var(--radius-lg)] bg-bg-surface"
    />
  ),
});

export function HeroSlot() {
  return <RibbonHero />;
}
