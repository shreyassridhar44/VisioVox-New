# 22 — Risk Register

Severity = Impact × Likelihood. Reviewed at every phase boundary.
Risks marked ⭐ are carried forward from the archived roadmap §9, which identified them correctly.

---

## 1. Technical — ML

| ID | Risk | Imp | Lik | Sev | Mitigation | Trigger | Fallback |
|---|---|---|---|---|---|---|---|
| **R-01** ⭐ | 3–4 speaker quality below listenable threshold | H | H | 🔴 | AV conditioning; suppression loss; per-segment confidence | SI-SDRi < 8 dB at 3 spk on AMI-Val | Publish the degradation curve; cap the demo at 3; warn in-product at 4 |
| **R-02** | Benchmark gains don't transfer to real recordings | H | H | 🔴 | Realistic simulation; dereverb front-end; VVX in-domain fine-tune | AMI-Eval < 60% of Libri2Mix gain | More aggressive simulation; more VVX data; report the gap honestly |
| **R-03** | Visual conditioning shows no measurable gain | H | M | 🟠 | Verify ROI/audio alignment frame-by-frame at C2 start | C2 < +0.5 dB over C1 | Debug alignment first (usual cause); else ship audio-only TSE — Novelties 1,3,4,5 survive |
| **R-04** ⭐ | Isolated audio is artifact-y despite good SI-SDR | M | H | 🟠 | MRSTFT loss; gated restoration; listening test | DNSMOS < 3.0 | Enable restoration more aggressively; document as a limitation |
| **R-05** | Generative restoration hallucinates words | H | M | 🟠 | Fidelity gate refuses on very low φ; Faithful is default; ASR cross-check | Hallucination rate > 0.5% | Tighten the gate; if unfixable, disable S6 entirely |
| **R-06** | Self-enrolment fails on heavily overlapped recordings | M | M | 🟡 | Purity gate; visual-only fallback | > 15% of speakers lack usable enrolment | Visual-only conditioning; surface a warning |
| **R-07** | Whisper hallucinates on silent isolated tracks | M | H | 🟠 | VAD-gated ASR; `no_speech_prob` filter; energy check; diarization cross-check | Spurious captions in fixtures | Tighten thresholds; `near_silence.mp4` regression test |
| **R-08** | Diarization miscounts speakers | M | M | 🟡 | Audio+video cross-check; confidence surfaced | Count accuracy < 85% | Show detected count with confidence; allow manual correction (post-v1) |
| **R-09** | Training doesn't converge in budget | H | M | 🟠 | C0 smoke test; staged curriculum; early diagnosis | C1 misses 13 dB by week 10 | Init from a published TSE checkpoint |
| **R-10** | Same-gender pairs remain poor | M | M | 🟡 | This is precisely what visual conditioning targets | No improvement at C2 | Report as a limitation; it is the known hard case |

---

## 2. Technical — Application

| ID | Risk | Imp | Lik | Sev | Mitigation | Trigger | Fallback |
|---|---|---|---|---|---|---|---|
| **R-11** ⭐ | A/V sync drifts on switch or seek | H | M | 🟠 | Shared `AudioContext`; rate-based correction; automated sync suite | Drift > 40 ms in tests | Documented in [`12-…`](./12-media-pipeline-and-sync.md); built in week 5 to fail early |
| **R-12** | HLS switch gap misses the 120 ms target | M | H | 🟡 | Prefer WebAudio for typical durations | Measured > 200 ms | Accept for long content; document; show a transition state |
| **R-13** | iOS audio restrictions break playback | H | M | 🟠 | `resume()` in the user gesture; silent-switch detection | Fails on a real device | Device matrix testing before launch, not after |
| **R-14** | 3D hero tanks mobile performance | M | M | 🟡 | InstancedMesh; dpr cap; FPS-triggered degradation | < 30 fps mobile | Static poster fallback |
| **R-15** | Zustand over-subscription drops the player to 10 fps | M | M | 🟡 | Selector discipline; rAF outside React | Profiler shows 60 Hz re-renders | Documented in [`13-…`](./13-frontend-architecture.md) §4.1 |
| **R-16** | Leaked `AudioContext` breaks playback after N projects | M | M | 🟡 | Explicit `destroy()` in effect cleanup | Playback fails after several navigations | Covered by an E2E navigation test |

---

## 3. Security & privacy

| ID | Risk | Imp | Lik | Sev | Mitigation | Trigger | Fallback |
|---|---|---|---|---|---|---|---|
| **R-17** | RCE via ffmpeg CVE on uploaded media | 🔴 Critical | M | 🔴 | gVisor sandbox, no network, no credentials, non-root, read-only FS, resource caps | Any sandbox escape in testing | Blocks launch. Non-negotiable. |
| **R-18** | IDOR exposes another user's private recording | 🔴 Critical | M | 🔴 | Data-layer ownership checks; spec-generated cross-tenant test suite; 404 not 403 | Any suite failure | Blocks launch |
| **R-19** | Cost DoS via GPU abuse | H | M | 🟠 | Quotas, concurrency cap, `maxReplicaCount`, budget auto-pause | Spend > 80% budget | Automatic admission pause |
| **R-20** | Biometric data breach | 🔴 Critical | L | 🟠 | ⭐ Ephemeral by default — there is no database to breach | Any persisted voiceprint for a non-opted-in user | Nightly assertion test |
| **R-21** | User uploads a recording they had no right to make | H | M | 🟠 | Attestation, terms, takedown process | Complaint received | Not technically preventable; documented in the DPIA |
| **R-22** | Model weights supply-chain compromise | H | L | 🟡 | SHA-256 manifest verification; safetensors; offline hub | Checksum mismatch | Refuse to load; alert |
| **R-23** | Regulatory reclassification under the EU AI Act | H | L | 🟡 | Ephemeral biometrics keeps us out of biometric *identification* | Any proposal to add cross-video ID | Do not build it ([`16-…`](./16-privacy-compliance.md) §4) |

---

## 4. Data

| ID | Risk | Imp | Lik | Sev | Mitigation | Fallback |
|---|---|---|---|---|---|---|
| **R-24** ⭐ | Dataset licensing blocks commercial use | H | M | 🟠 | Production checkpoint trained only on permissive data; per-checkpoint model card | LRS3 for research ablations only |
| **R-25** | ~~VVX recording slips~~ | — | — | ✅ | **Realised 2026-08-26; contingency executed** | AMI is now the eval basis ([ADR-0015](./adr/0015-ami-replaces-vvx.md)) |
| **R-26** | ~~VVX data loss~~ | — | — | ✅ | **Closed** — no self-recorded data exists | Every dataset is re-downloadable |
| **R-27** | ~~Participant withdraws consent~~ | — | — | ✅ | **Closed** — no participants are recorded | — |
| **R-28** | Speaker leakage between train and eval splits | H | M | 🟠 | Automated disjointness check in CI | Results are invalid until fixed — check runs on every split change |

R-28 deserves emphasis: split leakage silently inflates every number in the report and is only
discovered when someone else tries to reproduce it. The automated check is cheap; the failure mode
is total.

---

## 5. Project

| ID | Risk | Imp | Lik | Sev | Mitigation | Fallback |
|---|---|---|---|---|---|---|
| **R-29** ⭐ | Insufficient time for app polish | M | H | 🟠 | Parallel tracks; mock pipeline | Documented cut order ([`21-…`](./21-implementation-plan.md) §5) |
| **R-30** | Scope creep | M | H | 🟠 | Explicit non-goals in the charter; every addition needs an ADR | Charter §4 is the answer |
| **R-31** | Single point of failure (solo maintainer) | H | M | 🟠 | Documentation-first; ADRs; runbooks; no undocumented tribal knowledge | This doc set is the mitigation |
| **R-32** | Contract drift breaks the parallel-track plan | H | M | 🟠 | Manifest schema frozen at Phase 1; blocking CI contract check | Detected in CI, not at integration |
| **R-33** | GPU hardware failure | H | L | 🟡 | Cloud GPU fallback; checkpoints in object storage | Rent capacity; lose days, not weeks |

---

## 6. Top risks

| Rank | ID | Risk | Why it ranks here |
|---|---|---|---|
| 1 | R-17 | ffmpeg RCE | Highest impact; entirely preventable; blocks launch |
| 2 | R-18 | IDOR | Highest impact; entirely preventable; blocks launch |
| 3 | R-01 | Multi-speaker quality | Core product capability; only partly controllable |
| 4 | R-02 | Domain transfer failure | Would make good benchmark numbers meaningless |
| 5 | R-26 | VVX data loss | Irreplaceable; cheap to prevent; catastrophic if it happens |

Risks 1, 2 and 5 are fully within our control and have known, affordable mitigations. Risks 3 and 4
are research risk — managed by honest measurement and documented fallbacks rather than eliminated.

---

## 7. Review

| When | Action |
|---|---|
| Phase boundary | Full review; update likelihood from observed evidence |
| Weekly | Scan actively-mitigating risks |
| On trigger | Execute the fallback; record what happened and what it cost |
| Post-mortem | Add any risk that materialised and was not on this register |

The last row matters most: a register that never gains entries is not being used. Every surprise is
a gap in the register, and recording it is how the register earns its keep on the next project.
