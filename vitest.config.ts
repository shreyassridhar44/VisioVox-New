import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

/**
 * TypeScript unit tests. The Python suite (`pytest`) still owns the pipeline,
 * the API and the contracts; this covers the browser-side code those cannot
 * reach — currently the playback sync arithmetic and caption indexing.
 *
 * The environment is plain Node, deliberately. Everything tested here is a
 * pure function precisely so it does not need a DOM, and the parts that do
 * need one need a *real* browser with real audio hardware rather than a
 * simulated one (docs/12 §7) — jsdom would only give false confidence.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./apps/web/src', import.meta.url)),
    },
  },
  test: {
    include: ['apps/web/src/**/*.test.ts'],
    environment: 'node',
  },
});
