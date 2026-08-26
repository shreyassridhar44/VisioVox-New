# VisioVox Documentation

Complete documentation set. Start with the reading paths below rather than reading top to bottom.

---

## Start here

| If you want to… | Read |
|---|---|
| Understand the goal | [`00-charter.md`](./00-charter.md) |
| **Know what changed from the original plan and why** | ⭐ [`02-approach-review.md`](./02-approach-review.md) |
| **Know what needs training and on which machine** | ⭐ [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) |
| **Set up or reproduce the workstation** | [`26-workstation-as-built.md`](./26-workstation-as-built.md) |
| **See the measured Tier 0 baseline** | ⭐ [`27-phase1-baseline-report.md`](./27-phase1-baseline-report.md) |
| Understand the model | [`03`](./03-research-landscape.md) → [`04`](./04-novelty.md) → [`05`](./05-ml-architecture.md) |
| Understand the system | [`09-system-design.md`](./09-system-design.md) |
| Start building | [`21-implementation-plan.md`](./21-implementation-plan.md) |
| Look up a term | [`24-glossary.md`](./24-glossary.md) |

---

## Reading paths

**New to the project (2 hours)**
`00-charter` → `02-approach-review` → `09-system-design` → `21-implementation-plan`

**ML track**
`03-research-landscape` → `04-novelty` → `05-ml-architecture` → `06-datasets` →
`07-training-playbook` → `08-evaluation-protocol` → ADRs 0001, 0002, 0003, 0005, 0010, 0013

**Application track**
`09-system-design` → `10-data-model` → `11-api-spec` → `12-media-pipeline-and-sync` →
`13-frontend-architecture` → `14-design-system` → ADRs 0004, 0006, 0007, 0011, 0014

**Security & compliance**
`15-security` → `16-privacy-compliance` → ADRs 0008, 0009 → `22-risk-register`

**Operations**
`17-infrastructure-deployment` → `18-observability-sre` → `23-runbook`

---

## Full index

### Foundation
| | Document | Contents |
|---|---|---|
| 00 | [Charter](./00-charter.md) | Vision, goals, non-goals, users, principles, constraints |
| 01 | [Requirements](./01-requirements.md) | Functional + non-functional requirements, acceptance scenarios |
| 02 | [Approach Review](./02-approach-review.md) | ⭐ Audit of the original roadmap; what changed and why |

### Machine learning
| | Document | Contents |
|---|---|---|
| 03 | [Research Landscape](./03-research-landscape.md) | BSS vs TSE, prior art, where the real gap is |
| 04 | [Novelty](./04-novelty.md) | Five contributions, each with a falsifiable claim and ablation |
| 05 | [ML Architecture](./05-ml-architecture.md) | The 10-stage pipeline, SEAVE model, artifact manifest |
| 06 | [Datasets](./06-datasets.md) | Four tiers, simulation, AMI-Eval, licensing |
| 07 | [Training Playbook](./07-training-playbook.md) | Curriculum, config, monitoring, failure modes, compute budget |
| 08 | [Evaluation Protocol](./08-evaluation-protocol.md) | Metrics, harness, ablation suite, listening test, report structure |

### System
| | Document | Contents |
|---|---|---|
| 09 | [System Design](./09-system-design.md) | C4 views, request flows, orchestration, scaling, tech decisions |
| 10 | [Data Model](./10-data-model.md) | Schema, invariants, migrations, retention jobs |
| 11 | [API Spec](./11-api-spec.md) | Endpoints, manifest, SSE, rate limits, contract testing |
| 12 | [Media & Sync](./12-media-pipeline-and-sync.md) | ⭐ Dual playback engines, crossfade, drift correction, packaging |

### Frontend
| | Document | Contents |
|---|---|---|
| 13 | [Frontend Architecture](./13-frontend-architecture.md) | Routes, BFF, player, landing page, performance |
| 14 | [Design System](./14-design-system.md) | Colour, type, motion, signature components, tone |

### Security & compliance
| | Document | Contents |
|---|---|---|
| 15 | [Security](./15-security.md) | STRIDE, authn/z, media sandboxing, CSP, supply chain |
| 16 | [Privacy & Compliance](./16-privacy-compliance.md) | Biometric handling, GDPR/BIPA/AI Act, DPIA |

### Operations
| | Document | Contents |
|---|---|---|
| 17 | [Infrastructure](./17-infrastructure-deployment.md) | Topology, containers, K8s, CI/CD, Terraform, DR, cost |
| 18 | [Observability](./18-observability-sre.md) | Tracing, metrics, SLOs, alerts, dashboards |
| 19 | [Testing Strategy](./19-testing-strategy.md) | Test pyramid, sync tests, ML gates, release gate |
| 20 | [Performance & Cost](./20-performance-cost.md) | Bottlenecks, optimisations, unit economics, capacity |

### Planning
| | Document | Contents |
|---|---|---|
| 21 | [Implementation Plan](./21-implementation-plan.md) | ⭐ Phases, milestones, dependencies, contingencies |
| 22 | [Risk Register](./22-risk-register.md) | Risks with triggers and fallbacks |
| 23 | [Runbook](./23-runbook.md) | Dev setup, operations, alert runbooks, DR |
| 24 | [Glossary](./24-glossary.md) | Terms, metrics, models, abbreviations |
| 25 | [Compute & Hardware](./25-compute-and-hardware.md) | ⭐ Which machine does what · **what actually needs training** · A5000 tuning |
| 26 | [Workstation as built](./26-workstation-as-built.md) | Pre-flight results, WSL2 setup, verified GPU baseline, what the smoke test found |
| 27 | [Phase 1 baseline report](./27-phase1-baseline-report.md) | ⭐ Tier 0 measured on AMI · permutation-error rate · the empirical test of ADR-0001 |

---

## Architecture Decision Records

| ADR | Decision | Status |
|---|---|---|
| [0001](./adr/0001-target-speaker-extraction-over-blind-separation.md) | ⭐ TSE over blind source separation | Accepted |
| [0002](./adr/0002-tfgridnet-backbone.md) | TF-GridNet backbone | Accepted |
| [0003](./adr/0003-self-enrolment.md) | Self-enrolment from diarization | Accepted |
| [0004](./adr/0004-dual-engine-playback.md) | ⭐ Dual-engine playback | Accepted |
| [0005](./adr/0005-gated-generative-restoration.md) | Gated restoration, dual delivery | Accepted |
| [0006](./adr/0006-service-topology.md) | Next.js + FastAPI split | Accepted |
| [0007](./adr/0007-job-orchestration.md) | Celery + Redis | Accepted |
| [0008](./adr/0008-ephemeral-biometrics.md) | ⭐ Ephemeral biometrics | Accepted |
| [0009](./adr/0009-sandboxed-media-processing.md) | ⭐ Sandboxed media processing | Accepted |
| [0010](./adr/0010-single-talker-passthrough.md) | Single-talker passthrough | Accepted |
| [0011](./adr/0011-landing-visual-stack.md) | React Three Fiber | Accepted |
| [0012](./adr/0012-object-storage-and-delivery.md) | Cloudflare R2 | Accepted |
| [0013](./adr/0013-dataset-licensing.md) | Dual-track dataset licensing | Accepted |
| [0014](./adr/0014-authentication.md) | Self-hosted auth + BFF bridge | Accepted |

Template: [`adr/0000-template.md`](./adr/0000-template.md)

---

## Templates

| Template | Use |
|---|---|
| [Model Card](./templates/model-card.md) | Required per trained checkpoint; gates deployment on licence status |
| [DPIA](./templates/dpia.md) | GDPR Art. 35 assessment — complete before beta |
| [VVX Consent Form](./templates/vvx-consent-form.md) | Participant consent for corpus recording |
| [Post-mortem](./templates/postmortem.md) | Required for every P1 incident |

---

## Archive

[`archive/original-roadmap.md`](./archive/original-roadmap.md) — the original planning document,
preserved. Its feasibility analysis (§2), WSL2 rationale (§3), pipeline skeleton (§4) and risk table
(§9) are carried forward; its model architecture, player design, novelty claim and application scope
are superseded. See [`02-approach-review.md`](./02-approach-review.md).

---

## Conventions

- **Requirement IDs** (`FR-PLAY-03`, `NFR-ML-01`) are stable and referenced from tests and ADRs
- **Findings** (`F1`…`F13`) refer to [`02-approach-review.md`](./02-approach-review.md) §3–4
- **Novelty axes** are numbered 1–6 in [`04-novelty.md`](./04-novelty.md)
- **Stages** `S0`…`S9` refer to [`05-ml-architecture.md`](./05-ml-architecture.md)
- **Risks** (`R-01`…) are in [`22-risk-register.md`](./22-risk-register.md)
- ⭐ marks the decisions and documents most central to the project

## Maintenance

Documentation changes ship in the same PR as the code they describe. Any architectural decision gets
an ADR before implementation. If a document and the code disagree, that is a bug in one of them —
fix it, don't work around it.
