# Data Protection Impact Assessment — VisioVox

> Template. Complete before beta ([`../16-privacy-compliance.md`](../16-privacy-compliance.md) §5).
> Required under GDPR Art. 35 — this is large-scale processing of biometric data.
> Review annually and on any change to biometric handling.

| | |
|---|---|
| Version / date | |
| Author | |
| Reviewer / DPO | |
| Next review | |

---

## 1. Describe the processing

**What:** Users upload audio/video recordings. The system performs voice activity detection, speaker
diarization, face detection and tracking, active speaker detection, speaker embedding extraction,
target speaker extraction, and automatic speech recognition. Outputs are per-speaker isolated audio
tracks and per-speaker transcripts.

**Why:** To let a user listen to one speaker at a time from a recording where several people spoke
simultaneously.

**Data categories:**

| Category | Special category? | Retention |
|---|---|---|
| Source A/V recording | Potentially, by content | User-configured, default 30 d |
| Speaker embeddings (voiceprints) | **Yes — biometric** | **Job duration only** |
| Face crops / thumbnails | **Yes — biometric** | **Job duration only** (thumbnails: with project) |
| Derived per-speaker audio | Personal data | With project |
| Transcripts | Personal data; possibly special by content | With project |
| Account data | Personal data | Life of account + 30 d |
| Audit events (hashed IP) | Personal data | 1 year |

**Data subjects:**
1. **Account holders** — our users, with a contractual relationship
2. **⭐ Persons recorded in uploaded media** — third parties with **no relationship to us**, who did
   not choose to be processed

Category 2 is the defining feature of this DPIA and drives most of the analysis below.

**Volume / scale:** [estimate jobs/month, distinct persons/month]

---

## 2. Necessity and proportionality

**Lawful basis:**
- Account holders: **contract** (Art. 6(1)(b))
- Recorded third parties: **legitimate interests** (Art. 6(1)(f)), supported by the uploader's
  attestation that they have the right to process the recording

**Art. 9 (special category):** Biometric data is processed *transiently* for the technical purpose of
separation, and is **not retained**. No biometric identifier persists that could later identify a
person. This materially reduces Art. 9 exposure. Where any persistence occurs it is by explicit
opt-in (Art. 9(2)(a)).

**Is it necessary?** Separating speakers requires distinguishing them, which requires characterising
their voices. There is no less-intrusive technical route to the stated purpose. Face processing is
necessary for the audio-visual model and for the speaker-selection interface.

**Is it proportionate?** Processing is limited to what the task requires:
- Biometric derivatives deleted at job completion
- Work artifacts deleted after 7 days
- Default retention 30 days, user-reducible to 24 hours
- No cross-video identification
- No training on customer media without separate opt-in
- Region-pinned storage and processing

---

## 3. Consultation

| Party | Consulted | Outcome |
|---|---|---|
| DPO | | |
| Ethics board (for VVX corpus) | | |
| Legal counsel | | |
| Prospective users | | |

---

## 4. Risk assessment

| # | Risk to data subjects | Likelihood | Severity | Overall | Mitigation | Residual |
|---|---|---|---|---|---|---|
| 1 | A persistent biometric database is created and later breached or misused | Low | **High** | Medium | ⭐ Biometric derivatives are ephemeral by design ([ADR-0008](../adr/0008-ephemeral-biometrics.md)); nullable column; 15-min sweeper; nightly assertion test | **Low** |
| 2 | Sensitive conversation content is exposed | Low | High | Medium | Encryption at rest and in transit; ownership checks; signed URLs ≤ 15 min; short retention; no public buckets | Low |
| 3 | A person is recorded and processed without their knowledge or consent | **Medium** | High | **High** | Uploader attestation with audit record; terms of service; takedown process; in-product transparency | **Medium** |
| 4 | Speech is misattributed to the wrong person | Medium | High | **High** | Cross-stream leakage audit; contested spans kept in both transcripts and marked; calibrated per-segment trust scores surfaced in the UI | Medium |
| 5 | Generated audio puts words in a real person's mouth | Low | **High** | Medium | Faithful track is default and is what gets transcribed; fidelity gate refuses restoration on severely degraded input; "AI-restored" labelling in UI, filename and metadata; hallucination rate measured and gated in CI | Low |
| 6 | Third-party data subjects cannot exercise their rights (they don't know we hold data) | Medium | Medium | Medium | Short retention; ephemeral biometrics limit what exists; takedown process; uploader is the accountable party | Medium |
| 7 | Cross-border transfer without adequate protection | Low | Medium | Low | Region pinning (EU/US/IN) at workspace level, enforced by region-pinned buckets and workers | Low |

Risks 3 and 6 are the ones that cannot be fully solved by technical means — they are inherent to
processing recordings of people who are not our users. They are reduced, not eliminated, and that
should be stated honestly rather than engineered around on paper.

Risks 4 and 5 are notable for being **product-quality risks that are simultaneously data-accuracy
obligations** under Art. 5(1)(d). Two of the system's five research contributions exist partly to
address them.

---

## 5. Measures

| Measure | Addresses | Status |
|---|---|---|
| Ephemeral biometric derivatives, opt-in persistence | 1 | |
| Encryption in transit (TLS 1.3) and at rest (AES-256) | 2 | |
| Server-side ownership checks on every artifact access | 2 | |
| Short-lived, scoped signed URLs | 2 | |
| Uploader rights attestation, logged | 3, 6 | |
| Default 30-day retention, user-reducible | 1, 2, 6 | |
| Verifiable deletion with receipts | 6 | |
| Leakage audit, contested-span marking, trust scores | 4 | |
| Faithful-by-default + generation labelling + hallucination gate | 5 | |
| Region pinning | 7 | |
| Sandboxed, credential-free media processing | 2 | |
| Audit logging | all | |
| Takedown process and contact | 3, 6 | |

---

## 6. Outcome

**Residual risk:** Low / **Medium** / High

**Highest residual risk:** A user uploading a recording they had no right to make (risk 3). Reduced
by attestation, terms and takedown, but not eliminable by technical means.

**Conclusion:** Processing may proceed / may proceed with conditions / requires prior consultation
with the supervisory authority.

**Conditions (if any):**

<br>

**Approved by:** ________________  **Role:** ____________  **Date:** ____________
