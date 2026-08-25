# Contributing

---

## Setup

```bash
# Inside WSL2 — NOT /mnt/c (see docs/23-runbook.md §2)
git clone <repo> ~/projects/visiovox && cd ~/projects/visiovox
cp .env.example .env.local
make dev          # postgres, redis, minio, api, web, mock worker
make seed
```

→ http://localhost:3000 · `demo@visiovox.local` / `demo1234`

No GPU required — the mock pipeline returns fixture results.

For ML work: `nvidia-smi` must work inside WSL2, and
`python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

---

## Workflow

```bash
git checkout -b feat/speaker-rename        # or fix/, docs/, refactor/, perf/, test/, chore/
# ... work ...
make lint fmt typecheck test
git commit
gh pr create
```

Small PRs. One logical change. A PR that touches the model, the API and the frontend should usually
be three PRs.

**Commit granularity:** one commit per completed unit of work — a feature, component, stage or fix.
Not per file, and not per phase. Push at phase boundaries. Roughly 5–15 commits per phase is normal;
the floor exists so `git bisect` can isolate a regression, which matters here because model
regressions are subtle and metric-only.

---

## Commits

### Attribution

All commits are authored solely by the repository owner. Do **not** add `Co-Authored-By` trailers,
"Generated with" footers, or any AI-tool attribution to commits, PRs, tags or release notes. This
applies to commits made with AI assistance.

```bash
git config commit.template .gitmessage     # enable the template with the policy reminder
```

### Format

Conventional Commits, with a requirement ID where one applies:

```
feat(player): equal-power crossfade on speaker switch [FR-PLAY-03]

Replaces the linear gain ramp, which dipped ~3dB at the midpoint because
uncorrelated signals sum in power rather than amplitude. Measured dip is
now under 0.4dB.

Closes #142
```

Types: `feat` `fix` `docs` `refactor` `perf` `test` `chore` `ml`
Scopes: `api` `web` `player` `worker` `ml` `infra` `docs` `contracts`

`ml` is a separate type because model changes have their own review criteria and their own gates.

---

## Pull requests

**Required in the description:**
- What changed and why
- Requirement IDs addressed
- Test evidence (which suites, which numbers)
- For ML changes: `make eval-quick` output, before and after
- For player changes: sync suite results
- For security-relevant changes: say so explicitly

**Checklist:**
- [ ] `make lint typecheck test` passes
- [ ] New code is tested at the appropriate levels
- [ ] Docs updated **in this PR** if behaviour changed
- [ ] ADR added if an architectural choice was made
- [ ] Instrumented if it can fail
- [ ] Failure behaviour defined, not just the happy path
- [ ] No secrets, no hardcoded config
- [ ] Migration is reversible (if any)

---

## Code standards

### Python
- 3.12 · `ruff` format + lint · `mypy --strict`
- Pydantic v2 at boundaries, `extra='forbid'`
- SQLAlchemy 2.0 async, parameterised queries only
- Docstrings on public functions; explain *why*, not *what*

### TypeScript
- `strict: true`, no `any`
- API types generated from OpenAPI — never hand-written
- Zustand with narrow selectors (see `CLAUDE.md`)
- Components in `packages/ui` ship all states + a Storybook story

### ML
- Hyperparameters in YAML configs, never inline
- Every run logs git SHA, config hash, dataset manifest hash, seed
- New losses need a numerical unit test
- Report mean ± std over 3 seeds for anything headline

---

## Testing

| Change | Required |
|---|---|
| API endpoint | unit + integration (+ ownership test is auto-generated) |
| Pipeline stage | unit + end-to-end fixture |
| Model / loss | numerical unit test + `make eval-quick` no regression |
| Player | sync suite (`make test-sync`) |
| UI component | Storybook + a11y (zero axe violations) |
| Anything | existing suites still pass |

```bash
make test              # everything
make test-unit
make test-integration
make test-e2e
make test-sync         # player timing — measures the actual product claim
make test-security
make eval-quick        # 30-item ML eval (GPU)
```

Regression gates are blocking. A 0.6 dB SI-SDRi drop fails CI; it is not a rounding error.

---

## Documentation

Docs ship with the code that changes behaviour. Specifically:

| Change | Update |
|---|---|
| Architectural decision | New ADR in `docs/adr/` |
| New requirement | `docs/01-requirements.md` |
| Pipeline change | `docs/05-ml-architecture.md` |
| API change | Regenerate contracts; update `docs/11-api-spec.md` if semantics changed |
| Operational change | `docs/23-runbook.md` |
| Something learned the hard way | `MEMORY.md` §Lessons |

A doc that contradicts the code is a bug. Fix it in the same PR.

---

## Adding a dependency

Justify it in the PR. Check: licence (must be permissive), maintenance status, size (frontend budget
in `docs/13` §6), whether it's already solved by something we have, and transitive dependency count.

Model weights: verify the SHA-256 against a manifest, prefer `safetensors`, never load an unverified
pickle.

---

## Security

Do **not** open a public issue for a vulnerability — see [`SECURITY.md`](./SECURITY.md).

Changes touching auth, uploads, media processing, artifact access or sharing get a security-focused
review. Flag them in the PR description.

---

## Working with AI assistants

[`CLAUDE.md`](./CLAUDE.md) and [`AGENTS.md`](./AGENTS.md) are the operating guides. If you find an
assistant repeatedly making the same mistake, add it to the "Common mistakes" table in `CLAUDE.md` —
that's what the table is for.

---

## Review

Reviewers look for: correctness, security (especially ownership checks and sandbox boundaries),
whether the invariants in `CLAUDE.md` still hold, test adequacy, and whether the docs match.

Disagreement is resolved by evidence — a measurement, a test, or an ADR. "I prefer it this way"
without one of those isn't a blocking review comment.
