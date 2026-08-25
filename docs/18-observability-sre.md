# 18 — Observability & SRE

---

## 1. Principle

A job takes minutes and passes through ten stages across three services. When a user says *"my video
failed"*, you must be able to answer **which stage, why, and on what input** from one identifier,
without reproducing anything.

That identifier is the **correlation ID**, minted at the browser and carried through every log,
span, metric exemplar and error to the end of the pipeline.

---

## 2. The three signals

| Signal | Tool | Retention | Answers |
|---|---|---|---|
| Traces | OpenTelemetry → Tempo | 7 d (30 d sampled) | Where did the time go? |
| Metrics | Prometheus → Grafana | 15 mo | Is it healthy? Trending? |
| Logs | Structured JSON → Loki | 30 d | What exactly happened? |
| Errors | Sentry | 90 d | What broke, for whom, how often? |

---

## 3. Tracing

One trace spans the whole job.

```
trace: job_01HX8…
├── browser: upload.init                            120 ms
├── api: POST /uploads/init                          45 ms
│   ├── db: quota check                               3 ms
│   └── storage: presign                             12 ms
├── browser: upload.parts                          42.0 s
├── api: POST /uploads/complete                      80 ms
└── job: process                                   8m 42s
    ├── worker-cpu: S0_ingest                       25.1 s
    │   └── sandbox: ffmpeg                         23.8 s
    ├── worker-gpu: S1_enhance                      41.0 s
    ├── worker-gpu: S2A_analyse_audio               55.2 s   ─┐ parallel
    ├── worker-gpu: S2B_analyse_video              152.4 s   ─┘
    ├── worker-gpu: S3_fuse                          1.8 s
    ├── worker-gpu: S4_enrol                        14.6 s
    ├── worker-gpu: S5_extract                      89.7 s   ← per-speaker child spans
    ├── worker-gpu: S6_restore                      44.9 s
    ├── worker-gpu: S7_transcribe                   71.3 s
    ├── worker-gpu: S8_audit                        19.4 s
    └── worker-cpu: S9_package                      44.8 s
```

Trace context propagates through the Celery message headers, so the worker spans join the same trace
as the HTTP request that started it — this is the part that is easy to get wrong and the part that
makes the whole thing useful.

Every stage span carries: `job_id`, `stage`, `attempt`, `speaker_count`, `duration_ms`,
`gpu_peak_vram_mb`, `model_version`, `pipeline_version`.

**Sampling:** 100% of jobs (low volume, high value), 10% of routine API reads, 100% of errors and
of anything slower than p99.

---

## 4. Metrics

### Services (RED)
```
http_requests_total{service,route,method,status}
http_request_duration_seconds{service,route}          # histogram
http_requests_in_flight{service}
```

### Jobs (the ones that matter)
```
job_duration_seconds{stage,speaker_count}             # histogram
job_stage_failures_total{stage,error_code}
job_queue_depth{queue}
job_queue_wait_seconds                                # histogram — drives ETA
job_rtf{speaker_count}                                # real-time factor  ⭐ NFR-PERF-01
job_outcome_total{outcome}                            # ready|failed|cancelled
job_partial_success_total                             # some speakers ok, some not
gpu_seconds_total{user_plan}                          # ⭐ cost attribution
```

### GPU (USE)
```
gpu_utilization_percent{node}
gpu_memory_used_bytes{node}
gpu_temperature_celsius{node}
worker_saturation{queue}
```

### Quality — the unusual ones
```
extraction_confidence{speaker_index}                  # histogram
leakage_repairs_total
unresolved_spans_total
speaker_count_detected{count}
modality_used_total{modality}                         # audiovisual|audio_only|visual_only
restoration_gate_decision_total{decision}
```

These are **model behaviour in production**, which nothing in a training loop can tell you. A rising
share of `audio_only` means face detection is degrading on real uploads. Falling
`extraction_confidence` means the input distribution has shifted. This is the early-warning system
for silent model degradation, and it is the observability that a purely research project never
builds.

### Client
```
player_switch_latency_ms                              # ⭐ NFR-PERF-03, measured live
player_drift_ms                                       # ⭐ FR-PLAY-05
player_engine_total{engine}                           # webaudio|hls
web_vitals{metric}                                    # LCP, INP, CLS
hero_fps                                              # 3D performance in the wild
```

Measuring switch latency and drift **in production** rather than only in tests is what turns the
central product claim into something continuously verified.

---

## 5. Logging

```jsonc
{
  "ts": "2026-08-25T14:03:22.481Z",
  "level": "info",
  "service": "worker-gpu",
  "correlation_id": "01HX8ZQ…",
  "trace_id": "4bf92f…",
  "job_id": "job_01HX…",
  "user_id": "usr_01HX…",
  "stage": "S5_extract",
  "event": "stage.completed",
  "duration_ms": 89712,
  "speaker_id": "spk_2",
  "modality": "audio_only",
  "mean_confidence": 0.71
}
```

Rules:
- JSON only; no free-form string logs in production paths
- `correlation_id` on **every** line
- Levels: ERROR (needs action), WARN (degraded but handled), INFO (state changes), DEBUG (off in prod)
- **Never logged:** media content, transcript text, embeddings, tokens, passwords, full IPs
- Log lines are events, not sentences: `"event": "stage.completed"`, not `"Finished stage 5"`

---

## 6. SLOs

| SLO | Target | Window | Error budget |
|---|---|---|---|
| API availability | 99.5% | 30 d | 3.6 h |
| API latency p95 < 200 ms | 99% | 30 d | — |
| Job success rate | 98% | 30 d | 2% |
| Job completes < 3× duration | 95% | 30 d | 5% |
| Player switch < 120 ms p95 | 99% | 30 d | — |
| Upload success | 99% | 30 d | — |

**Burn-rate alerting**, not threshold alerting:

| Burn rate | Window | Severity | Meaning |
|---|---|---|---|
| 14.4× | 1 h | Page | Budget gone in 2 days |
| 6× | 6 h | Page | Budget gone in 5 days |
| 3× | 1 d | Ticket | Budget gone in 10 days |
| 1× | 3 d | Ticket | Trending badly |

This avoids paging for a brief blip while still catching a slow bleed — which threshold alerts do
backwards.

---

## 7. Alerts

Every alert has: a clear name, a severity, a runbook link, and an owner. **An alert without a
runbook is deleted** — it will be ignored at 3 a.m. anyway.

| Alert | Condition | Sev | Runbook |
|---|---|---|---|
| APIDown | availability < 99% for 5 m | P1 | RB-01 |
| JobFailureSpike | failure rate > 10% for 15 m | P1 | RB-02 |
| QueueBacklog | queue wait > 30 m | P2 | RB-03 |
| GPUUnavailable | 0 healthy GPU workers, queue > 0, 10 m | P1 | RB-04 |
| DBConnectionsHigh | > 80% pool for 5 m | P2 | RB-05 |
| StorageErrors | 5xx from R2 > 1% | P2 | RB-06 |
| **BudgetExceeded** | spend > 100% of monthly budget | **P1** | RB-07 |
| **ExtractionQualityDrop** | median confidence < 0.6 over 24 h | **P2** | RB-08 |
| **ModalityShift** | audio_only share > 40% over 24 h | P3 | RB-09 |
| PlayerDriftHigh | p95 drift > 60 ms | P2 | RB-10 |
| CertExpiring | < 14 d | P3 | RB-11 |
| SecurityAnomaly | authz denials > 100/h from one user | P1 | RB-12 |

The two starred quality alerts are the ones a conventional setup would omit. They detect the model
getting quietly worse in production — the failure mode that damages a product like this most,
because nothing crashes and no error is logged.

---

## 8. Dashboards

| Dashboard | Audience | Panels |
|---|---|---|
| **Service health** | On-call | RED per service, error budget burn, active alerts |
| **Pipeline** | On-call, ML | Stage durations (stacked), failure by stage, queue depth, RTF trend, GPU utilisation |
| **Model quality** | ML | Confidence distribution, modality mix, speaker-count histogram, leakage repairs, restoration gate decisions |
| **Client** | Frontend | Web vitals, switch latency, drift, engine mix, hero FPS, browser breakdown |
| **Business** | All | Jobs/day, active users, media hours, GPU-seconds by plan, cost per job |
| **Cost** | All | Spend by component, budget burn-down, cost per job trend |

"Cost per job" is deliberately a headline number. It is the metric that determines whether the
product is viable, and it moves in response to engineering decisions (single-talker passthrough,
face-detection frame rate) in ways that are otherwise invisible.

---

## 9. On-call

Solo project → simplified but real:

- P1 → push notification, respond within 30 min waking hours, best effort overnight
- P2 → notification, respond same day
- P3 → ticket, next working session
- Status page updated for anything user-visible lasting > 15 min
- Post-mortem for every P1 and any P2 that recurs. Blameless, with tracked action items.

**Post-mortem template:** `docs/templates/postmortem.md` — timeline, impact, root cause, what went
well, what didn't, action items with owners and dates.

---

## 10. Instrumentation checklist

For every new endpoint or stage:
- [ ] Traced with meaningful span name and attributes
- [ ] Duration histogram
- [ ] Error counter with an error-code label
- [ ] Structured logs at boundaries with `correlation_id`
- [ ] Errors reported to Sentry with context
- [ ] Dashboard panel if it is a new subsystem
- [ ] Alert if it can fail in a way users notice
- [ ] Runbook entry for that alert
