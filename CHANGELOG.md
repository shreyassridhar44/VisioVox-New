# Changelog

Notable changes to VisioVox. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

Model versions (`seave-x.y.z`) are tracked separately from application versions and deploy
independently — see [`docs/23-runbook.md`](./docs/23-runbook.md) §5.

---

## [Unreleased]

### Added
- Complete documentation set: 25 design documents and 14 ADRs
- Project charter, requirements with stable IDs, and acceptance scenarios
- SEAVE architecture specification — audio-visual target speaker extraction
- Five novelty contributions, each with a falsifiable claim and a designed ablation
- Dual-engine playback design (Web Audio + HLS) replacing the original sync approach
- Security model: STRIDE threat model, media sandboxing spec, authn/authz design
- Privacy model: ephemeral biometrics, DPIA, GDPR/BIPA/AI-Act mapping
- Phased implementation plan with parallel ML and application tracks

### Changed — from the original roadmap
| Area | From | To | Why |
|---|---|---|---|
| Core model | Blind source separation (SepFormer/Libri2Mix) | Audio-visual target speaker extraction | Permutation instability over long recordings; fixed speaker count; ignores video ([ADR-0001](./docs/adr/0001-target-speaker-extraction-over-blind-separation.md)) |
| Playback sync | N `<audio>` elements synced via `currentTime` | Single `AudioContext`, gain crossfade, rate-based drift correction | `currentTime` is a seek, not a sync; independent elements drift ([ADR-0004](./docs/adr/0004-dual-engine-playback.md)) |
| Novelty | Fine-tuning a pretrained checkpoint | Five contributions targeting the deployment gap | Fine-tuning a checkpoint on its own dataset is a reproduction |
| App scope | Local-first, single-user, no auth | Production: auth, sandboxing, CDN, observability, CI/CD | Contradicted the stated goal |
| Metrics | SI-SNRi, WER, DER | + SIR, silence leakage, DNSMOS/UTMOS, leakage word rate, ECE | SI-SDR cannot distinguish leakage from artifact |
| Timeline | 12 weeks sequential | 24 weeks, two parallel tracks | App was hostage to model convergence |

### Added — pipeline stages absent from the original plan
- Dereverberation + denoising front-end (reverb is the dominant real-world degradation)
- Single-talker passthrough routing (quality **and** cost — 80–95% of a timeline is single-talker)
- Loudness normalisation to −16 LUFS / −1 dBTP (prevents level jumps on speaker switch)
- Cross-stream leakage audit and repair
- Per-segment confidence with calibration

### Deferred
- Cross-video speaker identification — deliberately not built
  ([ADR-0008](./docs/adr/0008-ephemeral-biometrics.md))

---

## Version scheme

| Component | Scheme | Example |
|---|---|---|
| Application | SemVer | `1.2.0` |
| Model | SemVer, independent | `seave-1.0.3` |
| Pipeline | SemVer, independent | `pipeline-1.1.0` |
| Artifact manifest | Major.Minor | `1.0` |
| API | Path-versioned | `/v1` |

**Breaking changes** — an API `/v2`, a manifest major bump, or a model change that regresses a
published quality target. Each requires a migration note here and a 6-month overlap for the API.

---

## Entry conventions

Model releases record the measured deltas, not adjectives:

```markdown
## [seave-1.1.0] — YYYY-MM-DD
### Changed
- Increased suppression loss weight 0.3 → 0.45

### Metrics (VVX-Eval, mean ± 95% CI over 3 seeds)
| Metric | Previous | New | Δ |
|---|---|---|---|
| SI-SDRi (2spk) | 14.2 ± 0.3 | 14.0 ± 0.3 | −0.2 |
| SIR (2spk)     | 20.1 ± 0.5 | 23.4 ± 0.4 | **+3.3** |
| WER            | 14.1%      | 14.0%      | −0.1 |

### Notes
Trades 0.2 dB SI-SDRi for 3.3 dB SIR. Accepted: SIR is the metric that corresponds to
audible leakage, which is the product requirement (G2).
```

Regressions are stated plainly, including when they were accepted as a deliberate trade.
