# 13 — Frontend Architecture

---

## 1. Stack

| Concern | Choice | Why |
|---|---|---|
| Framework | Next.js 15 (App Router), React 19 | RSC for a fast landing page; mature ecosystem |
| Language | TypeScript, `strict` | Generated API types make the contract enforceable |
| Styling | Tailwind CSS v4 + CSS variables | Zero runtime cost; tokens live in CSS so 3D and DOM share them |
| Components | shadcn/ui (Radix primitives) | Accessible by default; we own the source |
| Animation | Motion (Framer Motion) + GSAP ScrollTrigger | Declarative for UI, timeline control for scroll |
| 3D | React Three Fiber + drei | Declarative three.js, SSR-safe, code-splittable |
| Client state | Zustand | Player state changes 60×/s — Redux/Context re-render cost is not acceptable here |
| Server state | TanStack Query | Caching, retries, SSE integration |
| Forms | React Hook Form + Zod | Zod schemas shared with the API contract |
| Testing | Vitest, Testing Library, Playwright | Unit → integration → E2E |

---

## 2. Route structure

```
app/
├── (marketing)/                    ← static/ISR, no auth
│   ├── page.tsx                    landing (3D hero + live demo)
│   ├── how-it-works/
│   ├── pricing/
│   └── legal/{privacy,terms,dpa,aup}/
│
├── (auth)/
│   ├── login/  register/  verify/  reset/
│
├── (app)/                          ← authenticated shell
│   ├── layout.tsx                  nav, user menu, quota meter
│   ├── projects/
│   │   ├── page.tsx                grid
│   │   ├── new/                    upload
│   │   └── [id]/
│   │       ├── page.tsx            ⭐ player
│   │       ├── processing/         live job progress
│   │       └── transcript/
│   └── settings/{account,privacy,sessions,usage}/
│
├── share/[token]/                  ← public, read-only
└── api/                            ← BFF route handlers only
    ├── auth/[...nextauth]/
    └── proxy/[...path]/            server-side token attachment
```

**Route-group boundaries are also security boundaries.** `(marketing)` is fully static and
CDN-cached; `(app)` is dynamic, `no-store`, and auth-gated in middleware. Nothing under `(app)` is
ever cached at the edge.

---

## 3. The BFF pattern

Access tokens never reach client JavaScript.

```
Browser ──cookie(httpOnly, Secure, SameSite=Lax)──► Next.js route handler
                                                          │ reads session
                                                          │ attaches Bearer JWT
                                                          ▼
                                                     FastAPI /v1/…
```

Consequences:
- XSS cannot exfiltrate an API token — there isn't one in the document
- CSRF is handled by `SameSite=Lax` plus a double-submit token on mutations
- Token refresh happens server-side, invisibly
- Media URLs are the one exception: they are pre-signed, short-lived and go straight to the CDN

---

## 4. Player architecture

The most complex part of the client.

```
┌──────────────────────────────────────────────────────────────┐
│  <ProjectPlayer>                                             │
│                                                              │
│  ┌─────────────────────────┐  ┌──────────────────────────┐   │
│  │  <VideoStage>           │  │  <SpeakerRail>           │   │
│  │   ├ <video muted>       │  │   ├ <SpeakerCard × N>    │   │
│  │   ├ <ActiveSpeakerRing> │  │   ├ <MixedCard>          │   │
│  │   └ <CaptionOverlay>    │  │   └ <ModeToggle F/N>     │   │
│  └─────────────────────────┘  └──────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ <Timeline>  multi-lane waveform, overlap shading,    │    │
│  │             low-confidence markers, scrubber         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ <TranscriptPane>  word-highlighted, click-to-seek     │   │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
             │                          │
             ▼                          ▼
    usePlayerStore (Zustand)   PlaybackEngine (docs/12)
```

### 4.1 State — and the discipline that keeps it fast

```ts
interface PlayerState {
  // High-frequency (~60 Hz) — components subscribe with narrow selectors
  currentTimeMs: number;
  isPlaying: boolean;
  driftMs: number;

  // Low-frequency
  activeTrackId: TrackId;
  audioMode: 'faithful' | 'natural';
  volume: number;
  playbackRate: number;
  captionsVisible: boolean;

  // Derived, memoised
  activeCaption: CaptionSegment | null;
  activeSpeakerIds: SpeakerId[];    // who is speaking now, for the video ring
}
```

**The rule:** no component subscribes to the whole store.

```ts
// ✅ re-renders only when the caption text actually changes
const caption = usePlayerStore(s => s.activeCaption?.text);

// ❌ re-renders 60 times a second
const { activeCaption } = usePlayerStore();
```

`currentTimeMs` updates at 60 Hz. A single careless subscription re-renders the transcript pane
(potentially thousands of nodes) on every frame and drops the page to 10 fps. This is the main
performance hazard in the app, and it is why Zustand with selectors is used rather than Context.

The scrubber and caption highlight are additionally driven **outside React** — direct DOM/style
writes from the rAF loop — so the hot path does not reconcile at all.

### 4.2 Engine integration

```ts
function usePlaybackEngine(manifest: Manifest) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const engineRef = useRef<PlaybackEngine>();

  useEffect(() => {
    const Engine = manifest.playback_hint === 'hls'
      ? HlsSyncEngine : WebAudioSyncEngine;
    const e = new Engine();
    e.load(manifest, videoRef.current!).then(() => setReady(true));
    engineRef.current = e;
    return () => e.destroy();                 // teardown matters: leaked
  }, [manifest.project_id]);                  // AudioContexts exhaust the browser limit

  const selectTrack = useCallback(async (id: TrackId) => {
    const t0 = performance.now();
    await engineRef.current!.selectTrack(id);
    telemetry.timing('player.switch_ms', performance.now() - t0);   // NFR-PERF-03
    usePlayerStore.setState({ activeTrackId: id });
  }, []);
}
```

Switch latency is measured in production, not only in tests. A regression shows up as a metric, not
as a support ticket.

### 4.3 Speaker selection UX

| State | Presentation |
|---|---|
| Available, AV | Face thumbnail, speaker colour ring, speaking-time bar |
| Available, audio-only | Generated monogram avatar + "no face detected" tooltip |
| Currently speaking | Animated ring on the card **and** on the video overlay |
| Selected | Filled card, elevated, colour-saturated |
| Low mean confidence | Amber dot + tooltip explaining what that means |
| Extraction failed | Disabled card with the reason |

Never colour-only (NFR-A11Y-06): every speaker is identified by **colour + label + position +
avatar shape**. Colour is redundant reinforcement, not the carrier of meaning.

---

## 5. Landing page

The landing page has one job: make the value obvious in five seconds, without reading.

### 5.1 Hero — 3D

**Concept:** a single tangled waveform ribbon in 3D that, on scroll or on hover, **separates into
two or three coloured ribbons** that flow apart. That is the product, shown rather than described.

```tsx
// components/hero/SeparationRibbons.tsx
export function SeparationRibbons({ progress }: { progress: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    for (let i = 0; i < COUNT; i++) {
      const speaker = i % SPEAKERS;
      const mixedY  = sumOfWaves(i, t);                 // tangled
      const sepY    = wave(i, t, speaker) + OFFSET[speaker];
      dummy.position.set(x(i), lerp(mixedY, sepY, progress),
                              lerp(0, DEPTH[speaker], progress));
      dummy.updateMatrix();
      ref.current!.setMatrixAt(i, dummy.matrix);
    }
    ref.current!.instanceMatrix.needsUpdate = true;
  });
  return <instancedMesh ref={ref} args={[geo, mat, COUNT]} />;
}
```

Performance rules (NFR-PERF-07):
- **InstancedMesh**, one draw call — never N meshes
- Mutate matrices in `useFrame`; never `setState` per frame
- `dpr={[1, 1.5]}` — cap device pixel ratio; retina at 3× is invisible and triples fragment cost
- `frameloop="demand"` when idle
- Pause rendering when off-screen (`IntersectionObserver`) and on `visibilitychange`
- Dynamic import with `ssr: false`; the 3D bundle is a separate chunk excluded from the initial JS budget

Fallbacks:
| Condition | Behaviour |
|---|---|
| `prefers-reduced-motion` | Static rendered poster image; no WebGL initialised (NFR-A11Y-04) |
| No WebGL2 | Poster image (NFR-COMPAT-04) |
| Save-Data / slow connection | Poster image |
| Low FPS detected (< 25 for 2 s) | Degrade particle count, then fall back to poster |

### 5.2 Live demo — the section that actually sells it

Below the hero: a real, preloaded 20-second clip with 3 speakers and the actual player. Visitors
switch speakers and hear it work, with no signup.

This is worth more than the 3D hero. The 3D is a metaphor; this is the product. It should be the
first thing built and the last thing cut.

Implementation: pre-baked static assets on the CDN, the same `WebAudioSyncEngine`, no API calls.
Consented demo footage only ([`06-datasets.md`](./06-datasets.md) §6).

### 5.3 Scroll narrative

| Section | Content | Motion |
|---|---|---|
| Hero | Ribbons separate | Scroll-linked progress 0→1 |
| Problem | Overlapping waveform, unintelligible caption | Fade + waveform draw |
| Solution | Live demo player | Reveal on enter |
| How it works | Pipeline stages, animated in sequence | Staggered, ScrollTrigger |
| Novelty | The five contributions, plainly stated | Card stagger |
| Honest limits | What it can't do | Deliberately plain — no animation |
| CTA | Sign up | — |

The "honest limits" section is unusual on a landing page and is a deliberate choice: it matches the
product's core principle (Charter §7.1), and it is more persuasive to the technical audience this
project is aimed at than another feature grid.

---

## 6. Performance budget

| Metric | Budget |
|---|---|
| Initial JS (landing, gzipped, excl. 3D) | ≤ 250 kB |
| 3D chunk (lazy) | ≤ 400 kB |
| LCP (mid-tier mobile, 4G) | ≤ 2.5 s |
| INP | ≤ 200 ms |
| CLS | ≤ 0.1 |
| Player route JS | ≤ 180 kB |

Techniques: RSC by default (`'use client'` only where interactivity requires it), `next/image` with
AVIF/WebP, `next/font` self-hosted with `font-display: swap`, route-level code splitting, dynamic
import for three.js / hls.js / the waveform renderer, ISR for marketing routes.

CI enforces the budget via `@next/bundle-analyzer` + a size-limit check and Lighthouse CI. A PR that
exceeds budget fails.

---

## 7. Accessibility implementation

| Requirement | Implementation |
|---|---|
| Keyboard | Roving tabindex on the speaker rail; `1–4` selects; `space` play/pause; `←/→` ±5 s; `m` mute; `c` captions; `f` fullscreen |
| Screen reader | `aria-live="polite"` announces "Now playing Speaker 2 only"; the transcript is a semantic list |
| Focus | Visible ring using a dedicated token; never `outline: none` without a replacement |
| Reduced motion | `useReducedMotion()` gates every animation; 3D never initialises |
| Contrast | Tokens verified at build time by a contrast script in CI |
| Captions | User-adjustable size, font, background opacity, position |
| Non-colour identification | Speaker = colour + label + avatar + position |

Tested with axe-core in CI (zero violations gate) and manually with NVDA and VoiceOver.

---

## 8. Error and empty states

Every state is designed. Failure states get the same attention as the happy path, because for this
product they are common and informative.

| State | Design |
|---|---|
| Upload rejected | Explain *which* limit and what to do (trim it, compress it) |
| Processing | Live stage progress, plain-English labels, ETA, cancel |
| Processing warning | Inline, non-blocking: "No face found for Speaker 2 — using audio-only" |
| Partial success | Show the speakers that worked, mark the one that failed with a reason |
| Job failed | Reason, correlation ID, retry button if retryable |
| Low confidence segment | Amber caption underline + tooltip |
| Contested span | Both transcripts show it, marked "contested" |
| Expired project | Clear message, re-upload CTA |
| Offline | Toast; playback continues from buffered media |
| Signed URL expiring | Silent manifest re-fetch before expiry |

---

## 9. Testing

| Level | Tool | Scope |
|---|---|---|
| Unit | Vitest | Stores, hooks, engine logic (mocked `AudioContext`) |
| Component | Testing Library | Speaker rail, captions, upload form |
| Integration | Vitest + MSW | BFF routes, TanStack Query flows |
| E2E | Playwright | Upload → process (mocked) → play → switch |
| Visual | Playwright screenshots | Landing, player, key states |
| A11y | axe-core in Playwright | Zero violations |
| Perf | Lighthouse CI | Budget enforcement |
| **Audio sync** | Playwright + Web Audio analysis | [`12-media-pipeline-and-sync.md`](./12-media-pipeline-and-sync.md) §7 |

The audio-sync suite is the unusual one and the most valuable: it measures the actual product claim
(≤ 120 ms switch, ≤ 40 ms drift) in a real browser, rather than trusting that the implementation
matches the design.
