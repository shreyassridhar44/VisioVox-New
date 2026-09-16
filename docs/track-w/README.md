# Track W — working folder

The **plan of record** is [`../28-product-delivery-plan.md`](../28-product-delivery-plan.md).
That file says *what* to build and *why*. This folder is the **working memory**: what was actually
done, what was measured, what went wrong, and what the next person (or the next session) needs to
pick up without re-deriving anything.

## Read this first, every session

1. **[`STATUS.md`](./STATUS.md)** — one screen. Where the work is right now, what is blocked, what
   is next. If you read nothing else, read this.
2. **[`DECISIONS.md`](./DECISIONS.md)** — what was decided and what rules bind. Read before writing
   any code, so settled questions are not reopened and constraints are not accidentally broken.
3. The phase file for the phase named in STATUS.
4. `../28-product-delivery-plan.md` only if you need the *argument* behind a decision — DECISIONS.md
   carries the conclusion.

## Files

| File | Purpose |
|---|---|
| [`STATUS.md`](./STATUS.md) | Live resume pointer. **Update at the end of every working session.** |
| [`DECISIONS.md`](./DECISIONS.md) | Decision register, standing rules, and the build log of choices made while implementing |
| `W0…W9-*.md` | One working log per phase |

## How the three layers fit

```
  28-product-delivery-plan.md   the plan   — what to build, why, in what order
  track-w/DECISIONS.md          the rules  — what was settled, what binds you
  track-w/W*.md                 the work   — what was actually done and measured
  track-w/STATUS.md             the cursor — where to resume
```

Keep them in that relationship. The plan changes rarely and deliberately; the work logs change
constantly; STATUS changes every session.

## Phase log format

Each phase file has four sections, and they earn their place:

- **Tasks** — mirrors the checklist in doc 28, ticked as work lands. The source of truth for "is
  this done".
- **Decisions made while building** — choices taken *during* implementation that doc 28 did not
  anticipate. These are the ones that get forgotten and then re-litigated.
- **Measurements** — real numbers from this machine: free space, throughput, RTF, timings. Never
  guesses. A number here is quotable; a number in a plan is not.
- **Gotchas** — things that cost time. Written so they cost it once.

## Rules

- **A task is ticked only when its verification command has been run and passed.** Not when the
  code looks right.
- **Every measurement records the date and the command that produced it**, because hardware state
  drifts — the disk numbers in W0 were already stale within a month.
- **Write the gotcha down when you hit it**, not at the end of the phase. It will be less accurate
  later and it may prevent the next hour being lost.
- If a phase's plan turns out to be wrong, fix `../28-product-delivery-plan.md` too. Two documents
  disagreeing is worse than either being wrong alone.

## Environment reminder

The live repo is `~/visiovox/VisioVox-New` inside the **`VisioVox`** WSL distro
(`wsl.exe -d VisioVox`). The default distro is `Ubuntu` and is a bare scratch box.
`C:\Users\Admin\Desktop\visiovox-claude\VisioVox-New` is a **stale clone** — edits there are
invisible to every build, test and run.
