# ADR-0011 — React Three Fiber for the landing visuals

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`13-frontend-architecture.md`](../13-frontend-architecture.md) §5, G8, NFR-PERF-04/07, NFR-A11Y-04

## Context

G8 requires a landing page that people describe as good-looking, with animation, while meeting
LCP ≤ 2.5 s on mid-tier mobile and ≥ 50 fps desktop / ≥ 30 fps mobile — and remaining fully
functional under `prefers-reduced-motion` and without WebGL.

The concept: a tangled waveform ribbon that **separates into distinct coloured ribbons** on scroll —
the product's function, shown rather than described.

## Options considered

### A — CSS/SVG animation only
**Pros:** Tiny, universally supported, accessible by default.
**Cons:** Cannot express the depth and volume that makes the separation metaphor land. Fails G8's
ambition.

### B — Raw three.js
**Pros:** Full control; no abstraction overhead.
**Cons:** Imperative lifecycle fights React; manual cleanup is error-prone (leaked contexts, orphaned
geometry); harder to gate on reduced-motion; harder to code-split cleanly.

### C — React Three Fiber + drei ✅
**Pros:** Declarative scene graph; React lifecycle handles mount/unmount and disposal; SSR-safe with
`ssr: false` dynamic import; composes with the existing animation stack; `useFrame` gives direct
per-frame access without re-rendering React; large ecosystem.
**Cons:** ~150 kB on top of three.js. An abstraction to learn. Easy to write accidentally slow code
if `useFrame` triggers state updates.

### D — Spline / Rive / a prebuilt 3D tool
**Pros:** Fast to produce; designer-friendly.
**Cons:** Runtime dependency on a third-party player; less control; awkward to bind to scroll
progress and theme tokens; conflicts with a strict CSP.

## Decision

**React Three Fiber**, lazily loaded as a separate chunk, with mandatory fallbacks.

Performance rules (non-negotiable):
- `InstancedMesh` — one draw call, never N meshes
- Mutate matrices inside `useFrame`; **never** `setState` per frame
- `dpr={[1, 1.5]}` — cap device pixel ratio
- `frameloop="demand"` when idle
- Pause on `IntersectionObserver` exit and on `visibilitychange`
- FPS watchdog: below 25 fps for 2 s → reduce particle count, then fall back to poster

Fallback matrix:

| Condition | Behaviour |
|---|---|
| `prefers-reduced-motion` | Static poster; **WebGL never initialises** |
| No WebGL2 | Static poster |
| `Save-Data` or slow connection | Static poster |
| Sustained low FPS | Degrade, then poster |

## Rationale

R3F's decisive advantage over raw three.js is **lifecycle correctness**. A 3D scene mounted and
unmounted as users navigate must dispose geometries, materials and the WebGL context reliably.
Leaked contexts hit the browser's limit and break rendering after a few navigations — the same class
of bug as the leaked `AudioContext` in the player. R3F handles disposal as part of React unmount;
with raw three.js it is manual and routinely forgotten.

The static-poster fallback is a first-class path, not a degraded afterthought. It is what
`prefers-reduced-motion` users, WebGL-less browsers and weak devices actually see, and the page must
be good with it. A landing page that requires WebGL to be legible has failed NFR-A11Y-04.

**The 3D hero is not the most important thing on the landing page.** The live interactive demo
([`13-frontend-architecture.md`](../13-frontend-architecture.md) §5.2) is — it lets a visitor hear the
product work in ten seconds. The 3D is a metaphor; the demo is proof. Build the demo first.

## Consequences

**Positive** — Striking hero that communicates the product without text. Correct lifecycle handling.
Clean code-splitting keeps it out of the initial bundle. Composes with the design tokens.

**Negative** — ~400 kB lazy chunk. Ongoing performance vigilance. Two versions of the hero (3D and
poster) to design and maintain. Requires verifying that no CSP `unsafe-eval` is needed for shader
compilation.

**Neutral** — Three.js reads colours from the same CSS custom properties as the DOM, so 3D and UI stay
in sync across themes — implemented by parsing computed styles at init.

## Revisit when

- The 3D chunk measurably harms LCP or conversion → poster-only.
- Mobile FPS cannot reach 30 on target devices after optimisation.
- A CSS-only approach becomes expressive enough (e.g. broad CSS 3D / Houdini support).
