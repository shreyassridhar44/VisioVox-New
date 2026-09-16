#!/usr/bin/env node
/**
 * Contrast gate (docs/14-design-system.md §2).
 *
 * "A CI script parses the token file, computes contrast for every defined
 * foreground/background pairing, and fails the build below threshold. Contrast
 * is a test, not a review comment."
 *
 * Parses tokens.css directly rather than importing a JS palette, because the
 * CSS file is what the browser actually renders. A duplicated palette in JS is
 * a palette that can drift from the one users see.
 *
 * OKLCH is converted to sRGB here rather than pulled from a library so the
 * check has no runtime dependency and runs anywhere Node does.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const TOKENS = resolve(HERE, '../apps/web/src/app/tokens.css');

/** WCAG 2.2 AA: 4.5 for body text, 3.0 for large text and UI boundaries. */
const AA_TEXT = 4.5;
const AA_LARGE = 3.0;

// ---------------------------------------------------------------- colour --

function oklchToSrgb(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);

  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ ** 3;
  const m = m_ ** 3;
  const s = s_ ** 3;

  const lin = [
    +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];

  // Clamp: a token can specify a colour outside the sRGB gamut, and the browser
  // clips it too. Measuring the unclipped value would flatter the result.
  return lin.map((v) => Math.min(1, Math.max(0, v)));
}

/** WCAG relative luminance, from LINEAR sRGB (no second transfer function). */
function luminance([r, g, b]) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(fg, bg) {
  const a = luminance(fg);
  const b = luminance(bg);
  const [hi, lo] = a > b ? [a, b] : [b, a];
  return (hi + 0.05) / (lo + 0.05);
}

// ----------------------------------------------------------------- parse --

/** Extract `--name: oklch(L C H ...)` pairs from one CSS block. */
function parseBlock(css) {
  const out = new Map();
  const re = /--([a-z0-9-]+)\s*:\s*oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)/gi;
  let m;
  while ((m = re.exec(css)) !== null) {
    out.set(m[1], oklchToSrgb(Number(m[2]), Number(m[3]), Number(m[4])));
  }
  return out;
}

function blockFor(css, selector) {
  const start = css.indexOf(selector);
  if (start === -1) return '';
  const open = css.indexOf('{', start);
  const end = css.indexOf('\n}', open);
  return css.slice(open, end);
}

// ----------------------------------------------------------------- pairs --

/**
 * Every pairing the interface actually renders. Listing them explicitly rather
 * than testing the cross product keeps the failure message meaningful: a
 * violation names a real combination someone can go and look at.
 */
const PAIRS = [
  ['fg-primary', 'bg-base', AA_TEXT, 'body text on the page'],
  ['fg-primary', 'bg-surface', AA_TEXT, 'body text on a card'],
  ['fg-primary', 'bg-elevated', AA_TEXT, 'body text in a popover'],
  ['fg-secondary', 'bg-base', AA_TEXT, 'secondary text on the page'],
  ['fg-secondary', 'bg-surface', AA_TEXT, 'secondary text on a card'],
  ['fg-muted', 'bg-base', AA_LARGE, 'muted text on the page'],
  ['fg-muted', 'bg-surface', AA_LARGE, 'muted text on a card'],

  ['accent', 'bg-base', AA_LARGE, 'accent on the page'],
  ['accent', 'bg-surface', AA_LARGE, 'accent on a card'],
  ['success', 'bg-surface', AA_LARGE, 'success state'],
  ['warning', 'bg-surface', AA_LARGE, 'warning / low confidence'],
  ['danger', 'bg-surface', AA_LARGE, 'error state'],
  ['contested', 'bg-surface', AA_LARGE, 'contested attribution'],
  ['confidence-low', 'bg-surface', AA_LARGE, 'low confidence marker'],

  // Speaker colours carry identity, so they must be legible on every surface
  // they can land on — card, caption bar and timeline bed.
  ['spk-1', 'bg-surface', AA_LARGE, 'speaker 1'],
  ['spk-2', 'bg-surface', AA_LARGE, 'speaker 2'],
  ['spk-3', 'bg-surface', AA_LARGE, 'speaker 3'],
  ['spk-4', 'bg-surface', AA_LARGE, 'speaker 4'],
  ['spk-1', 'bg-inset', AA_LARGE, 'speaker 1 on the timeline bed'],
  ['spk-2', 'bg-inset', AA_LARGE, 'speaker 2 on the timeline bed'],
  ['spk-3', 'bg-inset', AA_LARGE, 'speaker 3 on the timeline bed'],
  ['spk-4', 'bg-inset', AA_LARGE, 'speaker 4 on the timeline bed'],

  ['border-strong', 'bg-base', AA_LARGE, 'strong border'],
];

// ------------------------------------------------------------------ run --

const css = readFileSync(TOKENS, 'utf8');

const themes = [
  ['dark', parseBlock(blockFor(css, ':root {'))],
  [
    'light',
    new Map([
      ...parseBlock(blockFor(css, ':root {')),
      ...parseBlock(blockFor(css, "[data-theme='light']")),
    ]),
  ],
];

let failures = 0;
let checked = 0;

for (const [theme, tokens] of themes) {
  if (tokens.size === 0) {
    console.error(`no tokens parsed for the ${theme} theme — has tokens.css moved?`);
    process.exit(1);
  }

  for (const [fg, bg, threshold, label] of PAIRS) {
    const a = tokens.get(fg);
    const b = tokens.get(bg);
    if (!a || !b) {
      console.error(`  MISSING  ${theme}  --${fg} / --${bg}`);
      failures += 1;
      continue;
    }
    const ratio = contrast(a, b);
    checked += 1;
    if (ratio < threshold) {
      console.error(
        `  FAIL  ${theme.padEnd(5)}  ${label.padEnd(34)} ${ratio.toFixed(2)}:1 < ${threshold}:1  (--${fg} on --${bg})`,
      );
      failures += 1;
    }
  }
}

if (failures > 0) {
  console.error(`\ncontrast gate: ${failures} failing pairing(s) of ${checked + failures} checked`);
  console.error(
    'Adjust the L value in tokens.css — OKLCH lightness maps almost directly to contrast.',
  );
  process.exit(1);
}

console.log(`contrast gate: ${checked} pairings pass in both themes`);
