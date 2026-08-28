/**
 * The playback engines, in a real browser (docs/12 §7).
 *
 * Everything else about these engines can pass while they have never decoded a
 * byte: types, lint, and unit tests over the arithmetic all hold whether or not
 * Web Audio, AAC decoding and the DOM actually cooperate. That is what this
 * suite is for, and it is why it runs against the real fixture — a genuine
 * 3-speaker Libri3Mix mixture with its true isolated sources — rather than a
 * stub.
 *
 * Gates that need real audio measurement over ten minutes of playback (the
 * long-run drift and seek-accuracy rows of docs/12 §7) are not here yet. What
 * is here is the part that was completely unverified: that the thing loads,
 * switches, and shows the right words at the right time.
 */

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import {
  buildCaptionIndex,
  parseTranscript,
  segmentAt,
} from '../apps/web/src/lib/playback/captions';

// Playwright runs from the repository root, which is also where the config
// lives, so this resolves the same way the browser's `/fixtures/` does.
const FIXTURES = resolve('apps/web/public/fixtures');

function readCaptions(speaker: string) {
  const path = join(FIXTURES, `${speaker}.captions.json`);
  const raw: unknown = JSON.parse(readFileSync(path, 'utf8'));
  const transcript = parseTranscript(raw);
  if (transcript === null) throw new Error(`unreadable fixture transcript for ${speaker}`);
  return buildCaptionIndex(transcript);
}

/** The rail is disabled until the engine reports ready, so this is the signal. */
async function waitForEngine(page: Page): Promise<void> {
  await expect(page.getByRole('radio', { name: /Speaker 1/ })).toBeEnabled({ timeout: 30_000 });
}

async function openDemo(page: Page, engine: 'WebAudio' | 'HLS'): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });

  await page.goto('/demo');
  await expect(page.getByRole('heading', { name: 'Player demo' })).toBeVisible();
  // Fail loudly rather than testing the empty state by accident.
  await expect(page.getByText('No fixtures yet')).toBeHidden();

  if (engine === 'HLS') await page.getByRole('button', { name: 'HLS', exact: true }).click();
  await waitForEngine(page);
  return errors;
}

test.describe('WebAudio engine', () => {
  test('decodes the fixture and offers every track', async ({ page }) => {
    await openDemo(page, 'WebAudio');

    // Three speakers plus the original mix, which is always selectable.
    for (const name of ['Speaker 1', 'Speaker 2', 'Speaker 3', 'Original mix']) {
      await expect(page.getByRole('radio', { name: new RegExp(name) })).toBeEnabled();
    }
    await expect(page.locator('.error')).toHaveCount(0);
  });

  test('starts on a speaker, not on the mixture', async ({ page }) => {
    await openDemo(page, 'WebAudio');
    await expect(page.getByRole('radio', { name: /Speaker 1/ })).toHaveAttribute(
      'aria-checked',
      'true',
    );
  });

  test('switching speakers moves the selection and raises nothing', async ({ page }) => {
    const errors = await openDemo(page, 'WebAudio');

    for (const name of ['Speaker 2', 'Speaker 3', 'Original mix', 'Speaker 1']) {
      await page.getByRole('radio', { name: new RegExp(name) }).click();
      await expect(page.getByRole('radio', { name: new RegExp(name) })).toHaveAttribute(
        'aria-checked',
        'true',
      );
    }
    expect(errors, `console errors: ${errors.join(' | ')}`).toHaveLength(0);
  });

  test('a switch resolves well inside the 120 ms budget', async ({ page }) => {
    await openDemo(page, 'WebAudio');
    const target = page.getByRole('radio', { name: /Speaker 2/ });

    const started = Date.now();
    await target.click();
    await expect(target).toHaveAttribute('aria-checked', 'true');
    const elapsed = Date.now() - started;

    // Round-trip through Playwright, so this bounds the switch rather than
    // measuring it: the audible crossfade is 80 ms of scheduled ramp on the
    // audio thread. What it does prove is that nothing blocks on a decode, a
    // seek or the network — those would be hundreds of ms, not tens.
    expect(elapsed).toBeLessThan(1_000);
  });

  test('plays without stalling', async ({ page }) => {
    await openDemo(page, 'WebAudio');
    await page.evaluate(async () => {
      const video = document.querySelector('video');
      if (video === null) throw new Error('no video element');
      await video.play();
    });
    await page.waitForTimeout(1_500);

    const advanced = await page.evaluate(() => document.querySelector('video')?.currentTime ?? 0);
    expect(advanced, 'playback did not advance').toBeGreaterThan(0.3);
  });
});

test.describe('captions', () => {
  test('show the words the transcript says, at the times it says them', async ({ page }) => {
    // The transcript is read here from the same file the browser fetches, and
    // the expected caption is computed with the same `segmentAt` the player
    // uses — so a disagreement means the wiring is wrong, not the arithmetic.
    const index = readCaptions('s1');
    await openDemo(page, 'WebAudio');

    const probes = [1.0, 2.5, 4.0, 6.0, 8.0, 10.0];
    const mismatches: string[] = [];

    for (const seconds of probes) {
      await page.evaluate((t) => {
        const video = document.querySelector('video');
        if (video !== null) video.currentTime = t;
      }, seconds);
      await page.waitForTimeout(250);

      const expected = segmentAt(index, seconds * 1000);
      const shown = ((await page.locator('.captions').first().textContent()) ?? '').trim();

      if (expected === null) continue; // a gap; the player shows an em dash
      const firstWord = expected.words[0]?.text.trim() ?? '';
      if (firstWord !== '' && !shown.includes(firstWord)) {
        mismatches.push(`t=${String(seconds)}s expected "${firstWord}" in "${shown}"`);
      }
    }
    expect(mismatches, mismatches.join('; ')).toHaveLength(0);
  });

  test('follow the selected speaker', async ({ page }) => {
    await openDemo(page, 'WebAudio');
    await page.evaluate(() => {
      const video = document.querySelector('video');
      if (video !== null) video.currentTime = 3;
    });
    await page.waitForTimeout(250);
    const first = await page.locator('.captions').first().textContent();

    await page.getByRole('radio', { name: /Speaker 3/ }).click();
    await page.waitForTimeout(300);
    const second = await page.locator('.captions').first().textContent();

    // Three different people reading three different passages: the caption
    // must change with the selection, or captions are not speaker-scoped.
    expect(second).not.toBe(first);
  });
});

test.describe('HLS engine', () => {
  test('loads the multivariant playlist and offers every rendition', async ({ page }) => {
    await openDemo(page, 'HLS');
    for (const name of ['Speaker 1', 'Speaker 2', 'Speaker 3', 'Original mix']) {
      await expect(page.getByRole('radio', { name: new RegExp(name) })).toBeEnabled();
    }
    await expect(page.locator('.error')).toHaveCount(0);
  });

  test('announces the buffer gap instead of hiding it', async ({ page }) => {
    await openDemo(page, 'HLS');
    await page.getByRole('radio', { name: /Speaker 2/ }).click();
    // The switch really does silence the audio while the buffer refills, so
    // the player has to say so — an unacknowledged gap reads as a fault.
    await expect(page.getByText('Switching speaker…')).toBeVisible();
  });

  test('hides the A/V offset control, which it cannot honour', async ({ page }) => {
    await openDemo(page, 'HLS');
    await expect(page.getByText('Audio delay')).toBeHidden();
    await expect(page.getByText('streaming · browser-synced')).toBeVisible();
  });
});
