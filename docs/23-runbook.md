# 23 — Operations Runbook

---

## 1. Quick reference

| | |
|---|---|
| Dashboards | Grafana → Service Health / Pipeline / Model Quality / Cost |
| Logs | `{service="api"} \| json \| correlation_id="…"` |
| Traces | Tempo, search by `job_id` |
| Errors | Sentry |
| Status page | status.visiovox.app |

**Given only a user complaint**, start here:
```
job_id or correlation_id
  → Tempo trace       (which stage, how long)
  → Loki logs         (what it said)
  → job_stages table  (what state it reached)
```

---

## 2. Development environment

### First-time WSL2 + CUDA setup

```powershell
# Windows (elevated) — install the NVIDIA driver on WINDOWS, not inside WSL
wsl --install -d Ubuntu-22.04
wsl --update
```
```bash
# Inside WSL2
nvidia-smi                                # must work — proves passthrough
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install -y ffmpeg libsndfile1 build-essential

git clone <repo> ~/projects/visiovox      # ⭐ WSL2 filesystem, NOT /mnt/c
cd ~/projects/visiovox && uv sync
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

> ⭐ **Never put the repo or datasets on `/mnt/c`.** The 9p bridge is several times slower and will
> bottleneck the dataloader before the GPU is saturated. This is the archived roadmap's advice and
> it is correct.

### Daily
```bash
make dev        # compose up: postgres, redis, minio, mailhog + web + api + mock worker
make seed       # demo user + fixture projects
make test       # full suite
make eval-quick # 30-item ML eval
make lint fmt typecheck
```

---

## 3. Common operations

### Inspect a job
```bash
make job-info JOB=job_01HX…       # state, stages, timings, errors
make job-logs JOB=job_01HX…
make job-artifacts JOB=job_01HX…  # list storage keys + sizes
```

### Retry a failed job
```bash
# check retryability first — permanent failures should not be retried
make job-info JOB=… | grep error_code
kubectl exec deploy/api -- python -m cli jobs retry job_01HX…
```

### Cancel a stuck job
```bash
kubectl exec deploy/api -- python -m cli jobs cancel job_01HX… --force
```

### Drain GPU workers (before deploy)
```bash
kubectl annotate deploy/worker-gpu drain=true    # stop consuming new tasks
watch kubectl get pods -l app=worker-gpu         # wait for in-flight stages to finish
kubectl rollout restart deploy/worker-gpu
```

### Reprocess with a new model
```bash
kubectl exec deploy/api -- python -m cli jobs reprocess job_01HX… \
        --from-stage S5_extract --pipeline-version seave-1.1.0
```
Stages before S5 are reused from storage — reprocessing after a model update is cheap.

---

## 4. Alert runbooks

### RB-01 · APIDown
1. `kubectl get pods -l app=api` — crash-looping?
2. `kubectl logs deploy/api --tail=200`
3. Check DB connectivity and pool saturation
4. Recent deploy? → `kubectl rollout undo deploy/api`
5. Update the status page if > 15 min

### RB-02 · JobFailureSpike
1. Pipeline dashboard → **which stage** is failing?
2. `job_stage_failures_total{stage}` grouped by `error_code`
3. Common causes:
   - `S0_ingest` → a new input format, or a sandbox misconfiguration
   - `S5_extract` → GPU OOM (check `gpu_memory_used_bytes`) or a bad model deploy
   - `S7_transcribe` → model cache empty on a new node
4. Bad model deploy → roll back the model version, not the code
5. If input-driven → identify the pattern, add a validation rule

### RB-03 · QueueBacklog
1. `job_queue_depth` and `job_queue_wait_seconds`
2. GPU replicas at `maxReplicaCount`? → capacity limit, working as designed
3. Workers healthy but idle? → broker connectivity
4. One job hogging? → check for a stuck stage, cancel it
5. Sustained → raise `maxReplicaCount` **only after checking the budget dashboard**

### RB-04 · GPUUnavailable
1. `kubectl get nodes -l gpu=true` — nodes present?
2. Spot/preemptible reclaim? → usually self-heals; confirm KEDA is scaling
3. `nvidia-smi` on the node; check the device plugin daemonset
4. Model cache volume mounted?
5. Queue drains automatically on recovery — no manual replay needed

### RB-07 · BudgetExceeded
1. Cost dashboard → which component?
2. Job admission auto-pauses at 100% — verify it did
3. Identify heavy users via `gpu_seconds_total{user_plan}`
4. Abuse? → suspend the account, review the audit log
5. Legitimate growth? → raise the budget deliberately, not reflexively

### RB-08 · ExtractionQualityDrop ⭐
The alert that matters most for a product like this.
1. Model Quality dashboard → confidence distribution over time
2. Correlate with: a model deploy, a pipeline version change, or an input-distribution shift
3. Check `modality_used_total` — a rise in `audio_only` means the vision path is degrading
4. Recent model deploy → roll back the model version
5. No deploy → likely input shift (new user segment, new device type). Sample recent jobs and
   listen. Feed findings back into training data.

**Always listen to samples.** Metrics tell you something changed; only listening tells you what.

### RB-12 · SecurityAnomaly
1. Audit log for the user: `{action=~"authz.deny|auth.fail"}`
2. Enumeration pattern? → suspend the account, block the IP
3. Any successful cross-tenant access? → **escalate to SEV1 immediately**
4. Preserve logs; follow [`15-security.md`](./15-security.md) §11

---

## 5. Model deployment

```bash
# 1. Validate
make eval-full MODEL=seave-1.1.0
# gates: SI-SDRi, SIR, WER, hallucination rate, ECE, RTF

# 2. Publish weights + checksum
make model-publish MODEL=seave-1.1.0

# 3. Staging
kubectl set env deploy/worker-gpu -n staging EXTRACTOR_VERSION=seave-1.1.0
make smoke-test ENV=staging

# 4. Canary — 10% of production jobs
kubectl exec deploy/api -- python -m cli config set extractor.canary_pct=10 \
                                                    extractor.canary_version=seave-1.1.0

# 5. Watch for 24h: confidence distribution, failure rate, RTF, user reports
# 6. Promote or roll back
kubectl exec deploy/api -- python -m cli config set extractor.version=seave-1.1.0 \
                                                    extractor.canary_pct=0
```

Model versions are deployed **independently of code**. A model rollback is a config change, not a
redeploy — which matters because model regressions are more common than code regressions and need a
fast, low-risk reversal.

---

## 6. Database operations

```bash
make migrate                      # apply
make migrate-down N=1             # revert one
make db-console                   # psql
```

Rules: never DDL directly on production; expand/migrate/contract for anything breaking;
`CREATE INDEX CONCURRENTLY`; test on a production-sized restore before applying.

### Slow query triage
```sql
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 20;
```

---

## 7. Disaster recovery

### Postgres restore
```bash
make db-restore-verify SNAPSHOT=<id>   # ⭐ ALWAYS to a scratch instance first
# verify row counts and a sample project, then:
make db-restore SNAPSHOT=<id> ENV=production
kubectl rollout restart deploy/api deploy/web
make smoke-test ENV=production
```
**RTO 1 h · RPO 5 min.** Rehearsed quarterly.

### Region failure
1. Declare SEV1, update the status page
2. Restore Postgres to the secondary region from the latest snapshot
3. R2 is multi-region — no action needed
4. Repoint DNS
5. Redeploy workers; re-enqueue interrupted jobs from `job_stages`

### Total loss of a training GPU
Checkpoints are in object storage. Provision a cloud GPU, restore the last checkpoint, resume.
Loss is bounded to one epoch.

### VVX corpus loss
No recovery path — it cannot be re-recorded identically. **Restore from the encrypted offsite
backup.** Verify that backup quarterly; this is the one dataset where an untested backup is
unacceptable (R-26).

---

## 8. Maintenance

| Task | Cadence |
|---|---|
| Dependency updates | Weekly (Dependabot) |
| Security patches | Immediate for CRITICAL |
| Backup restore verification | Quarterly |
| DR drill | Semi-annually |
| Access review | Quarterly |
| Secret rotation | Per [`15-security.md`](./15-security.md) §7 |
| Cost review | Monthly |
| Model quality review | Monthly — listen to samples, not just charts |
| Risk register review | Per phase boundary |
| SLO review | Monthly |

---

## 9. Escalation

| Situation | Action |
|---|---|
| Data breach suspected | SEV1 · contain · preserve evidence · 72 h clock starts |
| Cross-tenant access confirmed | SEV1 · disable the endpoint · full audit review |
| Sandbox escape | SEV1 · take workers offline · do not process new media until fixed |
| Budget runaway | Pause admission · investigate before raising limits |
| Legal / takedown request | Preserve, do not delete · route to counsel |
| Model producing harmful output | Roll back the model version immediately; investigate after |
