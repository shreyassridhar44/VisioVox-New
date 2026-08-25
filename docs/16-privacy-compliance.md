# 16 — Privacy & Compliance

> ⚠️ This document describes design decisions and their reasoning. It is not legal advice. Obtain
> qualified counsel before commercial launch, and before recording the VVX corpus if the project is
> institutional.

---

## 1. Why this needs unusual care

Most web applications process data about **their users**. VisioVox processes data about **people who
are not its users** — everyone recorded in an uploaded video, who never agreed to anything.

Worse, the data is **biometric**:

| Data | Category | Regime |
|---|---|---|
| Speaker embeddings (voiceprints) | Biometric identifier | GDPR Art. 9, BIPA, CUBI, DPDP |
| Face crops and thumbnails | Biometric when used to identify | Same |
| Voice recordings | Personal data; often sensitive by content | GDPR Art. 4 |
| Transcripts | Personal data; may reveal special categories | GDPR Art. 9 by content |

A speaker embedding is not incidental output — it is a **compact, durable, matchable identifier**
derived from a person's body. Storing them across videos would build a covert voice-identification
database. That is the thing to design away from, not to manage with a policy.

---

## 2. Design decisions

### D1 — Biometric derivatives are ephemeral by default ⭐

Speaker embeddings and face crops exist **only for the duration of the job** and are deleted at S9.
The database column is nullable and normally null
([`10-data-model.md`](./10-data-model.md) §2, invariant I4).

*Why:* it eliminates the entire risk category rather than managing it. There is no biometric
database to breach, subpoena, or misuse. The product works exactly as well without persistence —
the embeddings are needed during processing, not after.

*Cost:* cross-video speaker identification is impossible. That feature (proposed in the archived
roadmap §10) is deliberately not built. [ADR-0008](./adr/0008-ephemeral-biometrics.md).

### D2 — Persistence is opt-in, per-user, off by default
If cross-video identification is ever added, `users.persist_voiceprints` gates it, defaults FALSE,
requires explicit consent with a plain-language explanation, and carries its own independent expiry.

### D3 — Uploader attests to rights and consent
Before the first byte is uploaded (FR-UPL-08), the uploader affirms:

> I confirm I have the right to upload this recording and to process the voices in it, and that
> everyone recorded has consented or that I have another lawful basis.

Recorded with timestamp and hashed IP. This does not transfer legal responsibility to the user, but
it establishes the lawful basis, creates an audit record, and — most importantly — puts the question
in front of them at the right moment.

### D4 — No training on customer media without separate opt-in
`users.allow_training_use` defaults FALSE (NFR-PRIV-05). Separate from the terms of service,
separately revocable, with clear language. Media from users who have not opted in never enters a
training set, and the model card records which data each checkpoint saw.

### D5 — Short default retention
30 days by default, user-configurable down to 24 hours. Work artifacts deleted after 7 days.
Expiry warning 3 days ahead (FR-PRJ-05).

*Why:* data that no longer exists cannot leak. Long retention is a liability the product does not
need — this is not an archival service.

### D6 — Verifiable deletion
Deletion produces a **receipt** enumerating removed object keys and row counts, verified by a
follow-up job that confirms the keys no longer resolve (NFR-PRIV-04,
[`10-data-model.md`](./10-data-model.md) §2). "We deleted it" is a claim; a receipt is evidence.

### D7 — Data residency at workspace level
EU / US / IN, selected at signup, enforced by region-pinned storage buckets and workers. Data does
not cross regions, including for processing.

---

## 3. GDPR mapping

| Article | Requirement | Implementation |
|---|---|---|
| Art. 5 | Minimisation, storage limitation | Ephemeral biometrics (D1); 30-day default (D5); work artifacts 7 days |
| Art. 6 | Lawful basis | Contract for users; legitimate interest + uploader attestation for recorded third parties |
| **Art. 9** | **Special category (biometric)** | **Avoided by D1** — no persistent biometric identifier is stored. Where processing is transient and non-identifying, Art. 9 exposure is materially reduced. |
| Art. 12–14 | Transparency | Privacy policy in plain language; in-product notice of what is processed |
| Art. 15 | Access | `POST /me/export` — full export |
| Art. 16 | Rectification | Editable profile, speaker labels, transcripts |
| **Art. 17** | **Erasure** | Project and account deletion with receipts (D6) |
| Art. 20 | Portability | JSON + media export |
| Art. 25 | Privacy by design/default | ⭐ The safe setting is the default and requires no user action |
| Art. 28 | Processors | DPAs with all sub-processors; list published |
| Art. 30 | Records of processing | Maintained (§7) |
| Art. 32 | Security | [`15-security.md`](./15-security.md) |
| Art. 33/34 | Breach notification | 72 h; procedure in security §11 |
| **Art. 35** | **DPIA** | ⭐ Required — biometric processing at scale. §5 below. |

---

## 4. Other regimes

| Regime | Relevance | Response |
|---|---|---|
| **BIPA (Illinois)** | Voiceprints and face geometry; **private right of action, statutory damages per violation** | D1 substantially reduces exposure. If persistence is enabled, BIPA requires written consent, a published retention schedule, and destruction within 3 years — all of which must be built before that feature ships. |
| **CUBI (Texas)** | Similar; AG enforcement | Same posture |
| **CCPA/CPRA** | Biometric = sensitive personal information | Do-not-sell/share (we do neither), disclosure, deletion |
| **DPDP Act 2023 (India)** | Consent, purpose limitation, data-principal rights | Consent notice, grievance officer, breach reporting |
| **EU AI Act** | ⭐ See below | §6 |
| **ePrivacy** | Cookies | Only strictly necessary cookies; no advertising or analytics cookies → no consent banner needed |

### On the EU AI Act
Remote biometric *identification* systems are heavily restricted. VisioVox performs biometric
**processing for separation**, not identification: it distinguishes speakers within a single
recording without determining who they are, and (under D1) retains nothing that could identify them
later.

This distinction is legally meaningful, and D1 is what preserves it. **Enabling cross-video speaker
identification would change the classification** and pull the product toward a far more heavily
regulated category. That is an additional and substantial reason not to build it — beyond privacy
hygiene, it is a regulatory boundary worth staying on the safe side of.

Transparency obligations for AI-generated or AI-manipulated content apply to the "Natural"
restoration track: it is generative output and is labelled as such in the UI, in the filename and in
file metadata ([`04-novelty.md`](./04-novelty.md) §5).

---

## 5. DPIA summary

Full template: `docs/templates/dpia.md`.

| Section | Content |
|---|---|
| **Processing** | Upload of A/V recordings; automated diarization, biometric-derived separation, transcription; storage of derived audio and text |
| **Necessity** | Separation is impossible without distinguishing speakers; distinguishing speakers requires voice characterisation. Minimum viable processing. |
| **Data subjects** | Account holders (users) **and all persons recorded** (non-users) |
| **Risks** | (1) Biometric database creation · (2) Sensitive conversation exposure · (3) Recording without consent · (4) Misattribution of speech · (5) Hallucinated content attributed to a real person |
| **Mitigations** | (1) D1 ephemeral · (2) Encryption, short retention, access control · (3) D3 attestation + notice · (4) Confidence disclosure + contested marking (Novelty 5) · (5) Faithful default + labelling + hallucination gate (Novelty 4) |
| **Residual risk** | Low–medium. Highest remaining: a user uploading a recording they had no right to make. Mitigated by attestation, terms, and takedown, but not eliminable by technical means. |
| **Review** | Annually, and on any change to biometric handling |

Risks 4 and 5 are notable: they are **product-quality risks that are also privacy risks**. Putting
words in someone's mouth is a data-accuracy failure under GDPR Art. 5(1)(d), not merely a bug. Two
of the five novelty contributions exist partly to address them — which is a useful thing to be able
to say in the DPIA.

---

## 6. Transparency in-product

| Moment | Disclosure |
|---|---|
| Signup | What is processed, retention, region choice |
| Before upload | Rights attestation (D3) with plain-language explanation |
| During processing | Which stages run; that faces and voices are analysed |
| On results | Confidence indicators; contested spans marked |
| Natural track | Labelled "AI-restored audio" wherever it plays, downloads or exports |
| Settings | Retention, region, training opt-in, voiceprint persistence — all visible in one place |
| Deletion | Receipt with what was removed |

---

## 7. Records of processing (Art. 30 extract)

| Purpose | Categories | Subjects | Recipients | Retention | Transfers |
|---|---|---|---|---|---|
| Speaker separation | A/V recordings, derived audio, transient voiceprints | Users, recorded persons | Cloud storage, GPU compute | 30 d default; biometrics job-scoped | Region-pinned |
| Transcription | Audio, transcripts | Same | Compute | 30 d | Region-pinned |
| Account | Email, name, auth | Users | Email provider | Life of account + 30 d | Region-pinned |
| Security audit | Events, hashed IPs | Users | — | 1 y | Region-pinned |

Sub-processors: cloud provider (compute, storage), email delivery, error tracking, observability.
Published list with DPAs; users notified 30 days before any addition.

---

## 8. Research-specific obligations (VVX corpus)

Recording the VVX dataset is human-subjects research involving biometric data.

- [ ] Ethics board / IRB approval **before** any recording
- [ ] Written informed consent per participant, covering: research use, model training, retention
      period, and (separately) publication or public demo use
- [ ] Right to withdraw, with recordings deleted and the model retrained without them
- [ ] Participants briefed on conversation topics; no sensitive personal content
- [ ] Raw recordings encrypted at rest, access limited to the project
- [ ] Consent records retained separately from the recordings
- [ ] Demo clips require an additional, explicit public-use consent

Template: `docs/templates/vvx-consent-form.md`.

---

## 9. Compliance checklist

**Before beta**
- [ ] Privacy policy, terms, AUP published and accurate to the implementation
- [ ] DPIA completed and signed off
- [ ] Deletion verified end-to-end with receipts
- [ ] Ephemeral biometric handling verified — a nightly test asserts zero persisted voiceprints for
      non-opted-in users
- [ ] Data export tested
- [ ] Sub-processor list published with DPAs in place
- [ ] Cookie audit — strictly necessary only

**Before general availability**
- [ ] Legal review of policy documents
- [ ] BIPA assessment if serving Illinois users
- [ ] Region pinning verified by test, not by configuration inspection
- [ ] Breach response rehearsed (tabletop exercise)
- [ ] Records of processing complete
- [ ] AI Act transparency labelling verified on every surface the Natural track reaches
