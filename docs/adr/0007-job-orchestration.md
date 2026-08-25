# ADR-0007 — Celery + Redis for job orchestration

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`09-system-design.md`](../09-system-design.md) §5, NFR-REL-03, FR-PIPE-14

## Context

A job is a 10-stage DAG taking 5–20 minutes, mixing CPU and GPU stages, some parallel (S2A ∥ S2B).
Requirements: per-stage retry, resume-from-last-completed-stage after worker loss, cancellation
within 10 s, live progress, and idempotency. GPU workers run on preemptible instances, so
**interruption is normal, not exceptional**.

## Options considered

### A — Celery + Redis ✅
**Pros:** Mature; `chain`/`chord`/`group` express the DAG directly; excellent Python ecosystem fit;
Redis already needed for cache, rate limits and SSE pub/sub. Well-documented failure modes.
**Cons:** Durability depends on the broker and on our own checkpointing. No built-in workflow
history. Celery's operational sharp edges (visibility timeout, late acks, prefetch) must be
understood and configured deliberately.

### B — Temporal
**Pros:** Purpose-built for durable long-running workflows. Automatic state persistence, replay,
built-in retry semantics, workflow history, native cancellation. Would handle preemption natively.
**Cons:** Substantial operational surface — server, database, workers, UI. Significant learning
curve. For a solo maintainer this is a second system to run and debug alongside the product.

### C — Dramatiq / RQ / arq
**Pros:** Simpler than Celery.
**Cons:** Weaker DAG primitives; would require hand-rolling the orchestration Celery already provides.

### D — Cloud-native (Step Functions, Cloud Workflows)
**Pros:** Managed durability.
**Cons:** Vendor lock-in; awkward fit for GPU containers; poor local development story.

## Decision

**Celery + Redis**, with durability provided by our own design rather than the broker:

1. Every stage writes outputs to object storage and records completion in `job_stages`
2. Stages are keyed by `(job_id, stage, pipeline_version)` and skipped if outputs exist with matching
   checksums — retries are therefore free
3. Resume reads `job_stages` and restarts from the first incomplete stage
4. `acks_late=True` with a visibility timeout exceeding the longest stage
5. `preStop` hook drains gracefully: stop consuming, finish the current stage, checkpoint, exit

Temporal is documented as the scale path.

## Rationale

The decisive observation: **checkpoint-to-storage plus idempotent stages gives us most of what
Temporal provides, for this specific workload, at a fraction of the operational cost.** Our stages
are naturally checkpointable — each produces large artifacts that must be written to storage anyway.
Durability is therefore nearly free.

Temporal's advantages are largest for workflows with complex in-memory state and fine-grained steps.
Ours is a linear chain of coarse, artifact-producing stages. That is the shape where the simpler
approach holds up.

For a solo maintainer, "one fewer distributed system to operate" is a genuine engineering
consideration, not laziness. Redis is already a dependency; Temporal would be a new one.

The design deliberately keeps nothing durable in Redis. On Redis loss, in-flight jobs are re-enqueued
from Postgres `job_stages` — the source of truth is the database, not the broker.

## Consequences

**Positive** — Fewer moving parts. Idiomatic Python. Redis serves multiple purposes. Retries are free
because stages are idempotent. Preemption costs one stage, not one job.

**Negative** — Durability is our responsibility, and a bug in the idempotency keys silently breaks it.
No workflow-history UI — debugging relies on traces and `job_stages`. Celery's configuration sharp
edges must be got right.

**Neutral** — The stage-checkpoint design is the migration path: it maps cleanly onto Temporal
activities if we ever move.

## Revisit when

- Workflows gain branching, human-in-the-loop steps, or long waits (Celery models these poorly).
- Jobs regularly fail to resume correctly → our durability layer is not holding.
- Scale exceeds ~30k jobs/month, where workflow observability becomes worth its operational cost.
