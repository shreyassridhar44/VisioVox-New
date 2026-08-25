# AGENTS.md

Task-specific playbooks for AI agents working on VisioVox.
Read [`CLAUDE.md`](./CLAUDE.md) first — it holds the invariants and conventions that apply everywhere.

> ⛔ **Attribution:** commits are authored solely by the repository owner. No `Co-Authored-By`
> trailers, no "Generated with" footers, no AI-tool references in commits, PRs or docs. Full policy
> in [`CLAUDE.md`](./CLAUDE.md).

> 🖥️ **Hardware:** app work runs on the laptop with `PIPELINE_MODE=mock` (no GPU). Training and real
> inference run on the college RTX A5000. See
> [`docs/25-compute-and-hardware.md`](./docs/25-compute-and-hardware.md).

---

## How to pick up work

1. Find the phase in [`docs/21-implementation-plan.md`](./docs/21-implementation-plan.md)
2. Read the linked design doc for that area
3. Check whether an ADR already decides the approach
4. Confirm the phase's **exit criteria** — that's the definition of done, not "it runs"

---

## Playbooks

### Adding an API endpoint

```
1. Define Pydantic request/response models (extra='forbid')
2. Add the route; use the require_project / require_owner dependency — never inline the check
3. Regenerate the OpenAPI spec:      make contracts
4. Regenerate clients:               make clients
5. Tests: unit (validation) + integration (DB) + ownership is auto-generated from the spec
6. Rate limit if it mutates or is unauthenticated
7. Audit-log it if it touches auth, media, sharing or deletion
```

Gotchas: the contract check is blocking — if the generated spec differs from the committed one, CI
fails. Never hand-edit `packages/ts-client` or `packages/py-client`.

### Adding a pipeline stage

```
1. Implement in ml/pipeline/stages/, subclassing Stage
2. Declare inputs/outputs as storage keys — stages communicate through storage, not memory
3. Make it idempotent: key on (job_id, stage, pipeline_version); skip if outputs exist with matching
   checksums
4. Emit progress events to Redis
5. Add to the Celery chain in the right position
6. Add a fixture that exercises it end to end
7. Instrument: span, duration histogram, failure counter
8. Define the failure behaviour — what happens downstream if this stage fails?
```

**Every stage needs a fallback.** The pipeline must never return nothing
([`00-charter.md`](./docs/00-charter.md) §7.3). If your stage can't produce output, define what
happens instead.

### Changing the model

```
1. Config change, not code change, where possible
2. make eval-quick — must not regress (SI-SDRi 0.5 dB, SIR 1.0 dB, WER 1.0 pt)
3. If it's an architecture change, C0 smoke test FIRST: overfit 100 samples in 200 steps
4. Full training → make eval-full → update the model card
5. Deploy: staging → 10% canary → 24h watch → promote (docs/23 §5)
```

Model versions deploy independently of code. A model rollback is a config change.

### Adding a player feature

```
1. Read docs/12 — the sync design is subtle and the constraints are not obvious
2. Narrow Zustand selectors. Always.
3. If it touches the audio graph, run the sync suite: make test-sync
4. Verify: play, pause, seek, scrub, rate change, tab background, device change
5. Test on a real iOS device — AudioContext behaviour differs from every simulator
6. Keyboard accessible; announced to screen readers
```

The sync suite measures the actual product claim. If it fails, the feature isn't done.

### Adding a UI component

```
1. Check packages/ui first — it may exist
2. Use design tokens; never hardcode a colour or duration
3. Ship all states: default, hover, focus, active, disabled, loading, error, empty
4. Keyboard accessible, visible focus, correct ARIA
5. Respect prefers-reduced-motion
6. Storybook story + visual regression snapshot
7. Speaker identification: colour + label + shape, never colour alone
```

### Investigating a failed job

```
make job-info JOB=job_01HX…        # state, stages, timings, error
# → Tempo trace by job_id           which stage, how long
# → Loki by correlation_id          what it logged
# → job_stages table                what state it reached
```

Then: was it input-driven (add a validation rule), infrastructure (retry), or a model regression
(roll back the model version)?

### Security-sensitive changes

Anything touching auth, uploads, media processing, artifact access or sharing:

```
1. Re-read the relevant section of docs/15
2. Ownership check at the data layer, not the route
3. 404 not 403 for cross-tenant access
4. ffmpeg → sandbox, always, no credentials, no network
5. Add an audit log event
6. Run: make test-security
7. Flag it in the PR description so it gets a security-focused review
```

---

## Work that needs human sign-off

Do not do these autonomously:

- Deploying to production
- Changing retention, deletion or biometric-handling behaviour
- Relaxing a security control (sandbox, CSP, rate limit, ownership check)
- Changing published quality targets or landing-page claims
- Anything touching the VVX corpus or consent handling
- Adding a dependency with a restrictive or unclear licence
- Marking an ADR superseded

---

## Sub-agent decomposition

Useful splits when parallelising:

| Track | Independent because |
|---|---|
| ML pipeline | Depends only on the artifact manifest contract |
| API + workers | Depends on the OpenAPI contract |
| Frontend | Runs entirely against the mock pipeline |
| Infrastructure | Depends only on container interfaces |

The contract in `packages/contracts` is what makes these independent. **A change to the contract is
not a local change** — it affects every track and needs coordination.

Bad splits: separating a model change from its evaluation; separating a player change from its sync
tests; separating an endpoint from its ownership test.

---

## Definition of done

A task is done when:

- [ ] Code works and is tested at the right levels
- [ ] `make lint typecheck test` passes
- [ ] Relevant regression gates pass (`eval-quick` for ML, sync suite for player)
- [ ] Documentation updated **in the same PR** if behaviour changed
- [ ] An ADR exists if an architectural choice was made
- [ ] Instrumented (span, metric, log) if it can fail
- [ ] Requirement ID referenced in the commit
- [ ] Failure behaviour defined, not just the happy path

"It runs on my machine and I'll add tests later" is not done.

---

## Reporting results

- State numbers, not adjectives. "+0.3 dB across 3 seeds, within noise" beats "improved."
- If something regressed, lead with it.
- If you couldn't complete part of the task, say which part and why — don't silently narrow scope.
- If a design document turned out to be wrong, say so explicitly and propose the correction.
