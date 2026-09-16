# W3 — Design system build-out

**State:** ✅ Done
**Plan of record:** [`../28-product-delivery-plan.md`](../28-product-delivery-plan.md) §W3
**Depends on:** none

---

## Tasks

- [x] Tailwind v4 wired to the OKLCH tokens from docs/14 §2, one palette via `@theme`
- [x] Expressive-surface tokens (glow, gradient mesh, glass) added rather than substituted
- [x] Typography tokens; tabular figures on every timecode and metric
- [x] Primitives: Button, Card, Field, TextArea, Progress, Badge, Alert, Skeleton, Spinner, SpeakerDot
- [x] CI contrast gate — `scripts/check-contrast.mjs`, wired into `make check`
- [x] Dark default, light fully supported, reduced motion honoured
- [x] `/styleguide` renders every primitive in both themes
- [ ] Amend ADR-0011 to record the expressive/restrained split — the split is implemented and
      documented in DECISIONS.md, but the ADR itself has not been edited

**Verified 2026-09-16:** contrast gate 46 pairings pass in both themes; build clean across all
routes; 23 KB of CSS served with the tokens present in the DOM.

**Found by the gate on first run:** `--border-strong` was 2.11:1 against the page background in
dark and 1.95:1 in light, against the 3:1 WCAG 1.4.11 requires for UI boundaries. That token draws
input edges, so a border nobody can see is an accessibility fault. The token was corrected; the
test was not weakened.

---

## Decisions made while building

_None yet._ Append dated entries. Anything load-bearing also goes in
[`DECISIONS.md`](./DECISIONS.md) Part 3, so it is visible without opening this file.

---

## Measurements

_None yet._ Every number records the date and the command that produced it. Hardware state drifts —
the disk figures in W0 were stale within a month.

---

## Gotchas

_None yet._ Write these down when you hit them, not at the end of the phase.
