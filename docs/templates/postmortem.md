# Post-mortem — `<short title>`

> Blameless. The goal is to find what in the **system** allowed this, not who did it.
> Required for every P1 and any P2 that recurs. Due within 5 working days.

| | |
|---|---|
| Date of incident | |
| Duration | |
| Severity | P1 / P2 / P3 |
| Author | |
| Status | Draft / Reviewed / Actions tracked |

---

## Summary

Two or three sentences. What broke, who was affected, how long, how it was resolved.

---

## Impact

| | |
|---|---|
| Users affected | |
| Jobs failed / delayed | |
| Data loss | none / … |
| Error budget consumed | |
| Cost impact | |
| External communication | status page / email / none |

---

## Timeline

All times UTC. Include detection, not just cause and fix — how long it took to *notice* is usually
the most actionable number here.

| Time | Event |
|---|---|
| | Change deployed / condition began |
| | First user impact |
| | **Detected** (by whom/what — alert, user report, chance?) |
| | Investigation started |
| | Root cause identified |
| | Mitigation applied |
| | Fully resolved |

**Time to detect:** ___  **Time to mitigate:** ___  **Time to resolve:** ___

---

## Root cause

What actually happened, mechanically. Go past the proximate cause.

Not: *"the GPU worker OOMed."*
But: *"chunk length is derived from available VRAM at process start, but a second model was loaded
lazily during S6, so the S5 chunk size was computed against memory that was no longer free."*

**Contributing factors:**
- What made this possible?
- What made it hard to detect?
- What made it hard to diagnose?

---

## Detection

- How was it detected?
- Was there an alert? Did it fire? Was it actionable?
- If a user reported it before monitoring did — **why?** That's usually the most important finding.

---

## Resolution

What was done to mitigate, and what was done to fix. Note if the mitigation was a workaround that
still needs a real fix.

---

## What went well

Genuinely — good instrumentation, a clean rollback, a runbook that worked. Worth recording so it
isn't accidentally removed later.

---

## What went badly

- Missing or noisy alerting
- A runbook that was wrong or absent
- A test that should have caught it
- An assumption in a design doc that turned out to be false

---

## Where we got lucky

Things that could have made it much worse and didn't. These are latent risks, and they are the most
valuable output of a post-mortem — they identify the next incident before it happens.

---

## Action items

| # | Action | Type | Owner | Due | Issue |
|---|---|---|---|---|---|
| 1 | | prevent / detect / mitigate / document | | | |

Prefer **prevent** over **detect** over **mitigate**. An action item that is only "add an alert"
means the same failure will happen again, just visibly.

Every action item gets a tracked issue. A post-mortem with untracked actions is a document, not a fix.

---

## Documentation updates

- [ ] Runbook (`docs/23-runbook.md`)
- [ ] Risk register (`docs/22-risk-register.md`) — **add the risk if it wasn't there**
- [ ] `MEMORY.md` §Lessons if a design assumption was wrong
- [ ] ADR if a decision needs revisiting
- [ ] Alert definitions

The risk-register line matters: a register that never gains entries isn't being used. Every surprise
is a gap in it.
