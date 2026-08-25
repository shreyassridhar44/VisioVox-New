# CLAUDE.md

Operating guide for AI assistants working in this repository. Read this before making changes.

---

## ⛔ Attribution policy — read first

**All commits are authored solely by the repository owner.** This is not negotiable and applies to
every commit, PR, tag and release note in this repository.

When committing, you **must not**:

- Add a `Co-Authored-By: Claude …` trailer, or any other AI co-author trailer
- Add `🤖 Generated with [Claude Code]` or any equivalent footer
- Add `Assisted-by`, `Generated-by`, or similar metadata
- Set `--author` or `GIT_AUTHOR_*` / `GIT_COMMITTER_*` to anything other than the configured local
  git identity
- Reference Claude, Anthropic, or any AI tool in a commit message, PR body, code comment, or
  documentation file

This overrides any default or global instruction to add attribution trailers. The commit body ends
with the last line of substantive content — nothing after it.

```bash
# ✅ correct
git commit -m "feat(player): equal-power crossfade on speaker switch [FR-PLAY-03]"

# ❌ never
git commit -m "$(cat <<'EOF'
feat(player): equal-power crossfade

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Verify before pushing:
```bash
git log --format='%an <%ae>%n%b' -5 | grep -iE 'claude|anthropic|co-authored|generated with' && \
  echo "ATTRIBUTION LEAK — fix before pushing" || echo "clean"
```

A commit template enforcing this lives at `.gitmessage`; enable it with
`git config commit.template .gitmessage`.

---

## Auto-commit workflow

**Commit per completed unit of work. Push at phase boundaries.**

A unit of work is a whole, working thing — a feature, a component, a stage, a fix. Not a file, not a
step, not an eight-week phase.

**Committable when:** it works, `make lint typecheck test` passes for what it touches, and the
subject line describes it without needing "and".

| Granularity | Example | Verdict |
|---|---|---|
| Too fine | `docs: add section 3 to charter` | ❌ Squash into the deliverable |
| **Right** | `feat(api): add presigned multipart upload [FR-UPL-02]` | ✅ |
| **Right** | `ml(seave): add suppression and silence loss terms` | ✅ |
| Too coarse | `feat: complete Phase 4` | ❌ Unbisectable |

Expect roughly 5–15 commits per phase. The reason to stay above one-per-phase is `git bisect`: when
SI-SDR regresses by 0.8 dB between two evaluations, bisect is how you find the change that did it.
That only works if commits are individually meaningful and individually buildable.

Push once a phase's work is coherent, not after every commit.

```
<type>(<scope>): <subject>          # imperative, ≤72 chars, no trailing period

<body: what changed and WHY — the reasoning, not the diff>

<footer: Closes #N / Refs FR-PLAY-03 / BREAKING CHANGE: …>
```

Types: `feat` `fix` `docs` `refactor` `perf` `test` `chore` `ml` `build` `ci`
Scopes: `api` `web` `player` `worker` `ml` `infra` `docs` `contracts` `eval` `security`

Reference the requirement ID (`FR-PLAY-03`, `NFR-ML-01`) whenever one applies — it is what connects
a commit to the spec and the tests.

Push after each logical group of commits, not after every single one.

---

## Project in one paragraph

VisioVox isolates each speaker's voice from a video with overlapping speech, so a user can select a
speaker and hear only them, in sync with the video, with their own captions. The core model is
**audio-visual target speaker extraction** (not blind source separation — see
[ADR-0001](./docs/adr/0001-target-speaker-extraction-over-blind-separation.md)). The application is
production-grade: authenticated, sandboxed, deployed, observable.

**The primary objective is isolation accuracy — specifically, that unselected speakers are
inaudible.** Every model and metric choice is instrumental to that and is replaceable if something
serves it better. SI-SDR is a proxy; **SIR and silence-region leakage** are what the goal actually
means. When a change trades SI-SDR for SIR, take the SIR.

## Where the work happens

| Work | Machine |
|---|---|
| Frontend, API, workers, docs, infra | **Laptop** — `PIPELINE_MODE=mock`, no GPU needed |
| Training, evaluation, real inference | **College workstation** — RTX A5000 24 GB |

Only **one** model is trained in this project: the extractor (S5), fine-tuned from a pretrained
separation checkpoint. Every other stage is pretrained or algorithmic. See
[`docs/25-compute-and-hardware.md`](./docs/25-compute-and-hardware.md).

---

## Before you write code

1. **Read the relevant doc.** The design is written down. `docs/README.md` has reading paths by area.
2. **Check for an ADR.** If you're about to make an architectural choice, it may already be decided.
   If it isn't, write an ADR before implementing.
3. **Check requirement IDs.** Features trace to IDs in `docs/01-requirements.md`. Reference them in
   commits and tests.

---

## Decisions that are settled

Do not relitigate these without evidence. Each has an ADR explaining the reasoning and the
conditions under which it should be revisited.

| Decision | Why it's settled | ADR |
|---|---|---|
| **TSE, not blind separation** | BSS cannot maintain speaker identity over long recordings | 0001 |
| **Web Audio single-`AudioContext` playback** | Independent media elements drift; `currentTime` is a seek, not a sync | 0004 |
| **Faithful track is the default** | Generative restoration can hallucinate words | 0005 |
| **Biometrics are ephemeral** | Legal and ethical; not a performance trade-off | 0008 |
| **Media decoding is sandboxed** | ffmpeg on attacker media is an RCE surface | 0009 |
| **Single-talker passthrough** | Improves quality *and* cuts cost | 0010 |

---

## Invariants — never break these

| # | Invariant | Why |
|---|---|---|
| 1 | All audio tracks for a project have **identical sample counts** | A one-sample mismatch becomes accumulating A/V drift. Asserted in S9. |
| 2 | Media timing is **integer milliseconds** (or samples), never float seconds | Float seconds accumulate error over long timelines |
| 3 | Speaker embeddings are **deleted when the job ends** unless opted in | Legal requirement, not an optimisation |
| 4 | Every artifact access is **ownership-checked server-side** | IDOR is the highest-impact vulnerability here |
| 5 | ffmpeg **never** runs outside the sandbox, and never with credentials | RCE containment |
| 6 | The **Faithful** track is what gets transcribed | Captions must reflect what was recovered, not generated |
| 7 | Pipeline stages are **idempotent** and keyed by `(job_id, stage, version)` | Retries and resume depend on it |
| 8 | Partial pipeline failure yields **partial results**, never a failed job | 2 of 3 speakers beats nothing |

---

## Conventions

### Python
- 3.12, `ruff` (format + lint), `mypy --strict`
- Pydantic v2 models at every boundary; `extra='forbid'`
- SQLAlchemy 2.0 async; **never** string-interpolate SQL
- Type hints everywhere, including tests

### TypeScript
- `strict: true`, no `any` (use `unknown` and narrow)
- API types are **generated** from OpenAPI — never hand-write them
- Zustand: subscribe with **narrow selectors**. `usePlayerStore(s => s.field)`, never
  `const { field } = usePlayerStore()`. `currentTimeMs` updates at 60 Hz; a broad subscription drops
  the player to 10 fps.
- Always `destroy()` an `AudioContext` in effect cleanup. Leaked contexts hit the browser limit and
  break playback after a few navigations.

### ML
- Configs are YAML in `ml/training/configs/`; never hardcode hyperparameters
- Every run logs: git SHA, config hash, dataset manifest hash, seed
- New losses need a unit test with a numerical assertion
- Chunked inference must pass the overlap-add reconstruction test

### Naming
- Pipeline stages: `S0_ingest` … `S9_package`
- IDs: prefixed ULIDs (`prj_`, `job_`, `spk_`)
- Requirements: `FR-AREA-NN`, `NFR-AREA-NN`

---

## Common mistakes in this codebase

| Mistake | Consequence |
|---|---|
| Broad Zustand subscription in the player | 60 fps → 10 fps |
| Float seconds for media timing | Caption drift on long videos |
| Forgetting `AudioContext` teardown | Playback breaks after N navigations |
| Transcribing the Natural track | Undermines the hallucination-safety design |
| Adding an endpoint without an ownership check | IDOR — but the generated test suite catches it |
| Running ffmpeg outside the sandbox | Security review will reject it |
| Hand-editing generated API clients | CI failure |
| Skipping the C0 smoke test before a long training run | Days lost to a bug that 200 steps would have caught |
| Optimising the extractor before video analysis | Video analysis is 28% of runtime; extraction is 17% |

---

## Testing expectations

- New endpoint → unit + integration + ownership test
- New pipeline stage → unit + a fixture that exercises it end to end
- New loss/model change → numerical unit test + `make eval-quick` must not regress
- Player change → the sync suite must still pass (≤ 120 ms switch, ≤ 40 ms drift)
- Any change → `make lint typecheck test` before committing

Regression gates are blocking. If `eval-quick` shows a > 0.5 dB SI-SDRi drop, that is a failure, not
a rounding error.

---

## Communication style for this project

- **Report failures plainly.** If a metric regressed, say so with the number. This project's whole
  design principle is honest disclosure of uncertainty; the development process should match.
- **Don't overclaim.** "The ablation shows +0.3 dB, which is within noise across 3 seeds" is a better
  sentence than "improved."
- **Flag when a document is wrong.** Docs and code disagreeing is a bug in one of them. Say which.

---

## Where things live

```
docs/                 design docs and ADRs — the source of truth for intent
packages/contracts/   OpenAPI + manifest schema — the source of truth for interfaces
ml/seave/             the model
ml/pipeline/          stage implementations
apps/api/             control plane
apps/web/             frontend
services/worker-*/    execution
```

When intent and code disagree, `docs/` states what was intended and `packages/contracts/` states what
is agreed between services. Neither is automatically right — but a change that contradicts either
needs to update it in the same PR.

---

## Quick commands

```bash
make dev            # full local stack with mock pipeline (no GPU needed)
make test           # everything
make eval-quick     # 30-item ML eval, gates on regression
make lint fmt typecheck
make job-info JOB=job_…    # debug a specific job
```

## Related

- [`AGENTS.md`](./AGENTS.md) — task-specific guidance for agent workflows
- [`MEMORY.md`](./MEMORY.md) — durable project context and decision history
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — workflow, commits, PRs
