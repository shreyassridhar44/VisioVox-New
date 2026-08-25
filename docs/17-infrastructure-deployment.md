# 17 — Infrastructure & Deployment

---

## 1. Topology

```
                        ┌──────────────────────┐
                        │  Cloudflare          │
                        │  DNS · WAF · CDN     │
                        └──┬────────────────┬──┘
                           │                │
              ┌────────────▼──────┐   ┌─────▼───────────────┐
              │  WEB              │   │  R2 (object store)  │
              │  Next.js          │   │  media · artifacts  │
              │  container, 2+    │   │  zero egress fees   │
              └────────┬──────────┘   └─────▲───────────────┘
                       │                    │
              ┌────────▼──────────┐         │
              │  API              │         │
              │  FastAPI, 2+      │─────────┤
              └───┬───────────┬───┘         │
                  │           │             │
        ┌─────────▼──┐   ┌────▼──────┐      │
        │ Postgres   │   │ Redis     │      │
        │ (managed,  │   │ (managed) │      │
        │  HA, PITR) │   └────┬──────┘      │
        └────────────┘        │             │
                     ┌────────▼─────────────┴───┐
                     │  WORKERS                 │
                     │  CPU pool (2–8)          │
                     │  GPU pool (0–N, →zero)   │
                     └──────────────────────────┘
```

**Cloudflare R2 for object storage** is the decision with the largest cost effect: media egress is
the dominant bandwidth cost in a video product, and R2 charges none. On S3 with CloudFront, egress
alone would likely exceed all other infrastructure spend combined.
[ADR-0012](./adr/0012-object-storage-and-delivery.md).

---

## 2. Environments

| | local | ci | staging | production |
|---|---|---|---|---|
| Compute | Docker Compose | GH Actions | 1 replica each | 2+ replicas |
| GPU | Optional; mock by default | Mock | 1, scale-to-zero | 0–N, capped |
| DB | Postgres container | Ephemeral | Managed, small | Managed HA + PITR |
| Storage | MinIO | MinIO | R2 bucket | R2 bucket |
| Domain | localhost | — | staging.visiovox.app | visiovox.app |
| Data | Seeded fixtures | Ephemeral | Synthetic + demo | Real |

**The mock pipeline is a first-class artifact.** It returns a pre-computed manifest from fixtures,
so the whole application can be developed, tested and demoed without a GPU. This is what allows the
parallel ML/app tracks in [`21-implementation-plan.md`](./21-implementation-plan.md).

---

## 3. Containers

Multi-stage, distroless where possible, pinned by digest.

```dockerfile
# services/worker-gpu/Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3-pip libsndfile1 && rm -rf /var/lib/apt/lists/*

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

FROM base AS runtime
COPY --from=deps /app/.venv /app/.venv
COPY services/worker-gpu /app
COPY ml /app/ml
# Weights are NOT baked in — mounted from a cache volume, checksum-verified at load
ENV MODEL_CACHE=/models HF_HUB_OFFLINE=1
USER 10001:10001
ENTRYPOINT ["/app/.venv/bin/celery","-A","worker","worker","-Q","gpu","-c","1"]
```

Two deliberate choices:

- **Weights are not in the image.** They are gigabytes, they change independently of code, and
  baking them makes every deploy a multi-GB pull. Mounted from a persistent volume and verified by
  SHA-256 against a manifest before load ([`15-security.md`](./15-security.md) §8).
- **`HF_HUB_OFFLINE=1`.** A production worker must never reach out to download a model at runtime.
  It is a network dependency, a supply-chain risk, and a cold-start hazard.

The **ffmpeg sandbox** is a separate minimal image with no Python, no credentials and no network,
invoked by the CPU worker.

---

## 4. Orchestration

Kubernetes for production. Compose for local.

```yaml
# GPU worker — the part that matters for cost
apiVersion: apps/v1
kind: Deployment
metadata: { name: worker-gpu }
spec:
  replicas: 0                       # KEDA scales from zero
  template:
    spec:
      runtimeClassName: nvidia
      terminationGracePeriodSeconds: 120     # let the current stage finish
      containers:
      - name: worker
        image: ghcr.io/…/worker-gpu@sha256:…
        resources:
          limits: { nvidia.com/gpu: 1, memory: 24Gi, cpu: 8 }
        volumeMounts:
        - { name: models, mountPath: /models, readOnly: true }
        lifecycle:
          preStop: { exec: { command: ["/app/graceful-stop.sh"] } }
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
spec:
  scaleTargetRef: { name: worker-gpu }
  minReplicaCount: 0
  maxReplicaCount: 4                # ⭐ hard cap — this is the budget control
  cooldownPeriod: 300
  triggers:
  - type: redis
    metadata: { listName: "celery:gpu", listLength: "1" }
```

`maxReplicaCount` is the single most important number in the infrastructure. Without it,
autoscaling on queue depth converts a traffic spike directly into an unbounded GPU bill.

**Graceful shutdown** matters more than usual: GPU nodes are often preemptible/spot. `preStop` stops
accepting new tasks, finishes the current stage, checkpoints to storage, and exits — so a preemption
costs one stage, not one job (NFR-REL-03).

---

## 5. CI/CD

```
PR opened
  ├─ lint (ruff, eslint, prettier)
  ├─ typecheck (mypy --strict, tsc --noEmit)
  ├─ unit tests (pytest, vitest)          ─┐
  ├─ contract check (OpenAPI drift)        ├─ parallel
  ├─ security (gitleaks, trivy, pip-audit)─┘
  ├─ build images
  ├─ integration tests (compose)
  ├─ E2E (Playwright, mock pipeline)
  ├─ ml quick-eval (if ml/ touched)  ← 30 items, gates on regression
  ├─ lighthouse CI (if web/ touched)
  └─ preview deploy → ephemeral env

merge to main
  ├─ full test suite
  ├─ build + sign (cosign) + SBOM (syft)
  ├─ deploy staging
  ├─ smoke tests on staging
  ├─ ⏸ manual approval
  └─ deploy production (rolling, canary 10% → 100%)

nightly
  ├─ full ML eval on VVX-Eval
  ├─ dependency audit
  ├─ backup restore verification
  └─ deletion verification job
```

**Deployment strategy:** rolling for Web and API (stateless, fast). GPU workers **drain, don't
roll** — stop consuming, finish in-flight stages, then replace. Migrations run as a pre-deploy job
using expand/migrate/contract so old and new code coexist safely
([`10-data-model.md`](./10-data-model.md) §4).

**Rollback:** `kubectl rollout undo` for code. Database rollbacks are avoided entirely by the
expand/contract discipline — there is never a schema state that only one version can read.

---

## 6. Terraform

```
infra/terraform/
├── modules/{network,database,redis,storage,k8s,workers,monitoring,dns}/
└── envs/{staging,production}/
```

Rules: remote state with locking; no manual console changes (drift detection in CI fails the build);
secrets referenced from the secret manager, never in `.tf` or `.tfvars`; every `plan` reviewed in the
PR; `apply` only from CI.

**Policy tests** (Checkov/tfsec) enforce the invariants that must never regress:
- No public bucket ACLs
- Encryption at rest on every store
- No `0.0.0.0/0` ingress except on the load balancer
- All logs shipped to the central sink

Invariant I5 ("no artifact is publicly readable") is verified by a policy test rather than by
inspection, because it is exactly the kind of thing that regresses silently.

---

## 7. Networking

| Path | Exposure |
|---|---|
| Internet → Cloudflare | 443 only |
| Cloudflare → Web/API | Authenticated origin pull |
| Web → API | Internal service, mTLS |
| API → DB/Redis | Private subnet, no public route |
| Workers → DB/Redis/R2 | Private + workload identity |
| **Sandbox → anything** | **none** |
| Egress | Restricted allowlist |

Default-deny NetworkPolicies. The sandbox namespace has no egress at all
([`15-security.md`](./15-security.md) §4).

---

## 8. Backup & DR

| Asset | Method | RPO | RTO |
|---|---|---|---|
| Postgres | Automated + PITR | 5 min | 1 h |
| R2 media | Versioning + lifecycle | 0 | — |
| Redis | Ephemeral by design | n/a | n/a |
| Model weights | Versioned in registry + object store | 0 | 30 min |
| Config/IaC | Git | 0 | 15 min |
| **VVX corpus** | **Encrypted offsite, 3-2-1** | 24 h | 4 h |

Redis holds only queue state and cache; on loss, in-flight jobs are re-enqueued from Postgres
`job_stages`. That is a deliberate design property — nothing durable lives in Redis.

**VVX raw recordings are irreplaceable.** Everything else is reconstructible. Restores are tested
quarterly, not assumed.

**Disaster runbook:** in [`23-runbook.md`](./23-runbook.md) §7. Rehearsed at least once before
launch, because an untested restore is not a backup.

---

## 9. Cost

Estimate at ~1000 jobs/month, 10-minute average.

| Item | Monthly |
|---|---|
| GPU (scale-to-zero, ~150 h) | $90–180 |
| CPU workers | $40 |
| Web + API | $50 |
| Postgres (managed HA) | $60 |
| Redis | $20 |
| R2 storage (~2 TB) | $30 |
| **R2 egress** | **$0** ⭐ |
| Cloudflare | $20 |
| Observability | $0–50 (free tier) |
| **Total** | **≈ $310–450** |

On S3 + CloudFront, egress for the same traffic would add roughly $200–400/month, more than doubling
the bill. That single line is the justification for ADR-0012.

**Controls:** budget alerts at 50/80/100%; automatic job-admission pause at 100%; per-user GPU-second
quotas; `maxReplicaCount`; 7-day lifecycle on work artifacts; scale-to-zero on non-production
environments outside working hours.

---

## 10. Deployment checklist

- [ ] Migrations applied and reversible
- [ ] Feature flags default to off for anything unfinished
- [ ] Staging smoke tests green
- [ ] Rollback verified (an actual `rollout undo` in staging, not a plan on paper)
- [ ] Model weights present and checksum-verified on GPU nodes
- [ ] Terraform plan clean (no drift)
- [ ] Secrets present in the target environment
- [ ] Dashboards and alerts updated for anything new
- [ ] Runbook updated if operational behaviour changed
- [ ] Status page ready if downtime is expected
