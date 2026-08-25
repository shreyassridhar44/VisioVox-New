# ADR-0006 — Split Next.js web tier and FastAPI control plane

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`09-system-design.md`](../09-system-design.md) §3, [`02-approach-review.md`](../02-approach-review.md) §F4

## Context

The archived roadmap proposed a local-first, single-user app with "plain HTML/JS for speed" and
local filesystem storage. The actual requirement is a production-grade, authenticated, deployed,
visually rich application (G6, G8).

Two hard constraints shape the topology:
- The ML stack is **Python-only** (PyTorch, pyannote, SpeechBrain, ffmpeg bindings)
- The frontend needs **React Three Fiber, RSC and a modern animation stack** for G8

## Options considered

### A — Monolithic Python (FastAPI + Jinja / HTMX)
**Pros:** One language, one deploy, simplest ops.
**Cons:** Cannot deliver the player or the 3D landing page to the required standard. Rejected on G8.

### B — Monolithic Node/Next.js, ML via subprocess
**Pros:** One language for the app.
**Cons:** Bridging to Python for ML is fragile; loses Pydantic/typed API generation; GPU work inside a
web process is wrong on every axis.

### C — Next.js web tier + FastAPI API + separate workers ✅
**Pros:** Each tier is idiomatic. Python where the ML lives; React where the UI lives. Independent
scaling — the web tier scales on request volume, workers on queue depth. A clear, testable contract
boundary. The BFF pattern keeps tokens out of client JS.
**Cons:** Two runtimes, two deploy pipelines, two dependency ecosystems. Contract drift risk. More
infrastructure for a solo maintainer.

### D — Next.js + serverless functions + managed ML API
**Pros:** Minimal ops.
**Cons:** No serverless platform suits 10-minute GPU jobs. Managed ASR/separation APIs cannot run our
own model — which is the entire project.

## Decision

**Option C.** Four deployables: `web` (Next.js), `api` (FastAPI), `worker-cpu`, `worker-gpu`.

Contract enforced by `packages/contracts` — OpenAPI spec plus the artifact-manifest JSON Schema —
with generated clients and a blocking CI drift check.

## Rationale

The split follows a real boundary rather than an arbitrary one: **rendering and interaction** on one
side, **data, orchestration and inference** on the other. Attempting to collapse it forces one side
into a poor tool.

The contract package is what makes the cost of option C acceptable. Generated clients plus a CI
drift check convert "two services can disagree" from a chronic integration risk into a build
failure. It is also what allows the parallel ML/app tracks in
[`21-implementation-plan.md`](../21-implementation-plan.md) — the frontend builds against a frozen
manifest schema from week 3, months before a trained model exists.

Separating CPU and GPU workers is a security and cost decision, not just organisational: untrusted
media decoding stays in a cheap, disposable, network-isolated container (ADR-0009), while expensive
GPU workers holding model weights never touch attacker-controlled input directly.

## Consequences

**Positive** — Each tier uses the right tool. Independent scaling and deployment. Strong contract
boundary. Enables parallel development tracks.

**Negative** — Two runtimes to maintain, patch and secure. Two CI pipelines. Cross-service tracing
requires deliberate instrumentation. More moving parts for one person to operate.

**Neutral** — Web and API can be co-deployed for local development; the split is logical before it is
physical.

## Revisit when

- Contract drift causes repeated integration failures despite CI → the boundary is in the wrong place.
- Operational burden dominates feature work → consider co-locating web and API.
