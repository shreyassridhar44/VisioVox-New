# Roadmap

High-level view. Detailed plan with exit criteria:
[`docs/21-implementation-plan.md`](./docs/21-implementation-plan.md).

---

## Now — v0.1 → v1.0 (~24 weeks)

Two parallel tracks joined by a frozen contract.

```
 Week   1    4    8    12   16   20   24
        │    │    │    │    │    │    │
 ML     ├─P1─┼─P3─┼───P4────┼─P5─┼─P7─┤
 APP    ├─P2─┼──────P6──────┼─P8─┼─P9─┤
```

| Phase | Weeks | Track | Outcome |
|---|---|---|---|
| P0 Foundations | 1 | both | Stack runs; all pretrained models smoke-tested |
| P1 Baseline pipeline | 2–3 | ML | Honest "before" measurement; **manifest contract frozen** |
| P2 App skeleton | 2–4 | APP | Upload → mock processing → ready, no GPU needed |
| P3 Data | 4–6 | ML | LibriMix, VoxCeleb2, AMI, **VVX corpus recorded** |
| P4 Core model | 6–14 | ML | SEAVE meets quality floors on VVX-Val |
| P5 Novelty stages | 14–17 | ML | Self-enrolment, passthrough, restoration, leakage audit |
| P6 App build-out | 5–16 | APP | Player, landing page, security, accessibility |
| P7 Evaluation | 16–19 | ML | Baselines, ablations, listening test, report |
| P8 Integration | 19–21 | both | Real pipeline behind the real app |
| P9 Hardening & launch | 21–24 | both | Deployed, secure, observable, documented |

### Milestones

| # | Week | Milestone |
|---|---|---|
| M1 | 3 | Baseline pipeline + frozen contract |
| M2 | 4 | App running on the mock pipeline |
| M3 | 6 | VVX corpus recorded |
| M4 | 7 | **Sync engine proven** — highest app risk retired |
| M5 | 10 | Audio-only TSE ≥ 13 dB SI-SDRi |
| M6 | 12 | AV conditioning beats audio-only |
| M7 | 14 | Meets quality floors on VVX |
| M8 | 16 | Application complete |
| M9 | 19 | Ablations complete |
| M10 | 21 | Real end-to-end |
| M11 | 24 | **Production launch** |

---

## v1.0 — launch scope

**In**
- Upload video (≤ 60 min, ≤ 2 GB), 2–4 speakers
- Automatic speaker detection with face thumbnails
- Per-speaker isolated audio, full length, video-aligned
- Instant speaker switching in the player (≤ 120 ms)
- Per-speaker captions, word-aligned, click-to-seek
- Faithful / Natural audio modes
- Per-segment confidence disclosure
- Downloads: audio, captions, transcript
- Expiring share links
- Accounts, quotas, privacy controls, verifiable deletion
- Animated landing page with a live interactive demo

**Out** — real-time separation · > 4 speakers · non-English captions · music · native mobile apps ·
cross-video speaker identification (deliberately, see
[ADR-0008](./docs/adr/0008-ephemeral-biometrics.md))

---

## Next — v1.1 (post-launch, ~8 weeks)

Ordered by expected value:

| Feature | Why |
|---|---|
| **Multi-language captions** | Whisper is already multilingual; only evaluation scope gated it. Largest reach increase for the least work. |
| **Speaker mixer** (hear any subset together) | Falls out of the Web Audio architecture almost free |
| **A/B compare** (mixed ↔ isolated, instant) | Same — and it's the most persuasive demo interaction |
| **MP4 export with burned-in captions** | Most-requested export format in similar products |
| Manual speaker-count correction | Handles the diarization miscount case users will hit |
| Longer media (2 h) | Requires HLS engine hardening |
| Batch upload | Workflow for researchers with many sessions |

---

## Later — v1.2+

| Theme | Items |
|---|---|
| **Quality** | Distilled model (2–3× faster) · 4-speaker improvements · music/noise robustness |
| **Editing** | Transcript correction that re-syncs · speaker merge/split · trim and clip export |
| **Collaboration** | Shared workspaces · comments on timestamps · team plans |
| **Integrations** | Zoom/Meet/Teams recording import · API for programmatic access · webhooks |
| **Accessibility** | Live caption styling presets · translated captions · audio descriptions |

---

## Explicitly not planned

| Item | Reason |
|---|---|
| Cross-video speaker identification | [ADR-0008](./docs/adr/0008-ephemeral-biometrics.md) — privacy and regulatory |
| Voice cloning or synthesis | Out of scope; adjacent to misuse |
| Speaker demographic inference | Not our business |
| Real-time live separation | Materially harder problem (no lookahead); would compromise quality targets |
| Training on customer media by default | Opt-in only, always |

---

## Research directions

If the project continues past v1.0 as research:

- **Causal / streaming SEAVE** for real-time use — the hard version of the problem
- **Joint diarization + extraction** — currently sequential, so diarization errors propagate
- **Self-supervised pretraining** on unlabelled conversational video
- **Perceptually-weighted objectives** — the SI-SDR/audibility gap deserves a loss, not just a metric
- **Cross-lingual robustness** — the visual pathway should transfer better than the audio one
- **Beyond speech** — the same architecture may apply to isolating one instrument or one sound source

---

## How this changes

Roadmap items move when evidence says so — a milestone slips, an ablation falsifies a claim, or usage
data contradicts a priority. Changes are recorded in
[`CHANGELOG.md`](./CHANGELOG.md) and, where they alter an architectural decision, in a new or
superseding ADR.
