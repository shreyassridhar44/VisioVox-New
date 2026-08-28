import { defineConfig, devices } from '@playwright/test';

/**
 * Browser tests for the playback engines (docs/12 §7).
 *
 * These exist because everything else about the sync engines can pass while
 * the engines have never decoded a byte in a real browser. Unit tests cover
 * the arithmetic; this covers whether Web Audio, AAC decoding, HLS and the
 * DOM actually cooperate — which is not something a pure function can tell us.
 *
 * The port is fixed rather than random: the fixture manifest carries absolute
 * URLs, and the demo page retargets them to its own origin so the suite and a
 * hand-run `pnpm dev` can both work.
 */

const PORT = 3210;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: Boolean(process.env['CI']),
  retries: 0,
  workers: 1,
  reporter: [['list']],
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://127.0.0.1:${String(PORT)}`,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            // Playback has to start without a click, and headless Chromium has
            // no output device — the null sink still runs the graph, which is
            // what these tests observe.
            '--autoplay-policy=no-user-gesture-required',
            '--use-fake-device-for-media-stream',
          ],
        },
      },
    },
  ],
  webServer: {
    command: `pnpm --filter @visiovox/web exec next start -p ${String(PORT)}`,
    url: `http://127.0.0.1:${String(PORT)}/demo`,
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
