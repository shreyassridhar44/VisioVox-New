# 04 — Novelty & Contributions

> **SEAVE** — *Self-Enrolled Audio-Visual Extraction* — is the name of the core model and method.
> This document states each contribution as a falsifiable claim with a designed experiment.
> A claim without an ablation is marketing, not a contribution.

---

## 1. Why the original claim was not novel

The archived roadmap positioned the contribution as *"fine-tuning an open-source speech separation
model."* Fine-tuning a published checkpoint on the dataset it was designed for reproduces a known
result. The roadmap's own risk table anticipated the outcome — *"fine-tuning shows little
improvement over pretrained on Libri2Mix"* — which is exactly right, and is the tell that the framing
was wrong.

The contributions below instead target the **deployment gap** identified in
[`03-research-landscape.md`](./03-research-landscape.md) §6: problems that block real systems and
that benchmark-chasing does not address.

---

## 2. Contribution 1 — Self-Enrolment from Diarization (SEAVE-SE)

### Claim
> Target-speaker extraction can be applied to arbitrary uploaded video **with no user-supplied
> enrolment**, by mining enrolment cues from diarization-identified single-talker regions, with a
> purity-weighted aggregation that outperforms naive longest-segment selection.

### The gap
Every TSE system in §3.3 of the research landscape assumes an externally supplied clean enrolment
recording. Real users have none. This is not a small usability wrinkle — it is the reason TSE has
essentially no consumer products despite a decade of strong research results.

### The method

```
diarization (overlap-aware)
        │
        ▼
  candidate single-talker regions for speaker k
        │
        ▼  score each candidate region:
        │
        │   purity(r) = w1·(1 − overlap_prob(r))          # is anyone else talking?
        │             + w2·snr_norm(r)                     # is it clean?
        │             + w3·speech_density(r)               # is it actually speech?
        │             + w4·embedding_agreement(r, centroid_k)  # is it consistent?
        │             − w5·reverb_penalty(r)
        │
        ▼
  select top-M regions (target ≥ 8 s aggregate, ≥ 1.5 s each)
        │
        ├─► AUDIO CUE:  purity-weighted mean of L2-normalised ReDimNet embeddings,
        │               renormalised; plus a scalar audio_cue_confidence ∈ [0,1]
        │
        └─► VISUAL CUE: mouth-ROI sequence from the ASD-bound face track over the
                        same regions; plus visual_cue_confidence from face size,
                        frontality (yaw/pitch), sharpness and continuity
```

Two details that matter and are not obvious:

- **Purity weighting, not longest-segment.** The intuitive choice is "use the longest clean stretch."
  It is worse: a single long region captures one prosodic context (one sentence, one emotional
  register) and generalises poorly. Weighted aggregation over M diverse regions gives an embedding
  closer to the speaker's true centroid.
- **The confidences are the interface to Contribution 2.** They are not diagnostics; they are the
  routing signal.

### Failure handling
| Condition | Behaviour |
|---|---|
| < 1.5 s of pure speech for speaker k | audio cue unreliable → visual-only conditioning |
| no bound face track | visual cue absent → audio-only conditioning |
| neither available | speaker is merged/dropped from the registry with a user-visible note |

### Ablation
| Variant | Expected |
|---|---|
| Oracle enrolment (ground-truth clean recording) | upper bound |
| **SEAVE-SE purity-weighted (ours)** | within ~1 dB of oracle |
| Longest-single-region enrolment | −1 to −2 dB |
| Random 3 s region | −3 dB+ |
| Naive mean over all attributed audio (overlap included) | worst — contamination |

**Falsified if:** purity-weighted aggregation does not beat longest-region selection by ≥ 0.5 dB
SI-SDRi on AMI-Eval.

---

## 3. Contribution 2 — Modality-Adaptive Conditioning (SEAVE-MAC)

### Claim
> A single extraction model conditioned through a **reliability-gated fusion** of audio and visual
> cues degrades gracefully across the full range of real-world modality availability, and
> outperforms both (a) a fixed AV model given degraded video and (b) a hard modality switch.

### The gap
AV separation literature evaluates on datasets where the face is always visible, frontal and
well-lit (LRS2/LRS3 are curated for exactly this). Real uploads have head turns, occlusions, camera
cuts, speakers walking off-frame, and speakers who are never on camera at all. A fixed AV model fed
a bad visual stream doesn't just lose the visual benefit — it is actively *misled* by it.

### The method

Conditioning vector per frame, computed from continuous reliability scores rather than a discrete
mode switch:

```
  α_t = σ( g_a( audio_cue_conf ) )                # audio cue reliability, per-utterance
  β_t = σ( g_v( visual_conf_t ) )                 # visual reliability, PER FRAME

  c_t = α_t · (W_a · e_spk)  +  β_t · (W_v · v_t)  +  b
```

with FiLM modulation of the separator's intermediate features by `c_t`, and cross-attention from
audio frames to the visual sequence weighted by `β_t`.

**Training regime — this is the part that makes it work:** modality dropout. During training we
randomly (a) zero the visual stream, (b) zero the audio cue, (c) corrupt the visual stream
(occlusion masks, blur, frame drops, profile-view simulation via 3D warp), and (d) corrupt the
audio cue (mix in an interferer). The reliability gates are trained to *recognise* corruption, not
merely to be told about it.

This means the same weights serve AV, audio-only and visual-only inference. No model zoo, no
dispatch logic, no separate maintenance burden.

### Why per-frame β matters
A speaker who turns away for 2 seconds should lose visual conditioning **for those 2 seconds only**,
not for the whole utterance. Per-utterance gating (which is what a hard switch gives you) throws
away good visual evidence on either side of the occlusion.

### Ablation
| Condition | Fixed-AV | Hard switch | **SEAVE-MAC** |
|---|---|---|---|
| Clean frontal video | best | best | ≈ best |
| 30% frames occluded | degrades | degrades (switches whole utterance) | **holds** |
| Profile view throughout | degrades sharply | falls back to audio-only | **holds** |
| No video at all | fails / needs separate model | audio-only | **audio-only, same weights** |
| Contaminated audio enrolment | degrades | — | **leans visual** |

**Falsified if:** SEAVE-MAC does not beat the hard-switch baseline under partial occlusion by
≥ 0.7 dB SI-SDRi.

---

## 4. Contribution 3 — Suppression-First Training Objective (SEAVE-SFO)

### Claim
> Augmenting SI-SDR with an explicit **interferer-suppression** term and a **speaker-consistency**
> term materially reduces audible leakage (measured by SIR and cross-stream leakage word rate) at
> negligible SI-SDR cost — and SI-SDR alone does not achieve this because it cannot distinguish
> leakage from other error.

### The gap — and why it's the most product-relevant contribution

SI-SDR treats *all* error identically. Decompose the error:

```
  ŝ = s_target_scaled + e_interference + e_noise + e_artifact
```

SI-SDR sums them. But these are **not perceptually equivalent**:

- `e_artifact` (metallic/watery texture) → sounds like bad audio quality. Annoying.
- `e_interference` (another person's voice) → sounds like **another person talking**. Disqualifying.

For this product's stated requirement — *"the other speaker's voice must almost be zero"* — a system
that trades 1 dB of artifact for 4 dB of leakage suppression is a large win that SI-SDR would score
as a small loss. So we must put it in the objective explicitly.

### The loss

```
L = λ₁ · L_sisdr(ŝ, s)                      # fidelity to the target
  + λ₂ · L_suppress(ŝ, {s_j}_{j≠k})          # interferer energy in the output
  + λ₃ · L_consistency(ŝ, e_spk)             # output must sound like the enrolled speaker
  + λ₄ · L_mrstft(ŝ, s)                      # multi-resolution STFT — perceptual texture
  + λ₅ · L_silence(ŝ, s)                     # near-zero output where target is silent
```

Term by term:

| Term | Definition | Purpose |
|---|---|---|
| `L_sisdr` | −SI-SDR(ŝ, s) | Standard fidelity |
| `L_suppress` | `Σ_{j≠k} max(0, τ + SI-SDR(ŝ, s_j))` — hinged | Directly penalises the output correlating with any interferer. Hinged so it stops pushing once suppression is sufficient, avoiding over-suppression that eats the target. |
| `L_consistency` | `1 − cos(E(ŝ), e_spk)` | Output embedding must match the enrolment. Attacks leakage at the *identity* level, not the waveform level. Also directly reinforces Contribution 1. |
| `L_mrstft` | multi-res spectral convergence + log-magnitude | Improves perceived texture; SI-SDR is phase-sensitive and texture-blind |
| `L_silence` | energy of ŝ in ground-truth-silent target regions | **The one that fixes the actual user complaint.** During periods when the selected speaker is silent, the track must be *silent* — not a quiet murmur of the other speaker. Naive SI-SDR barely penalises this because there's no target signal to reference. |

`L_silence` deserves emphasis: in a real conversation, each speaker is silent 50–70% of the time.
That is the majority of the listening experience, and it's where leakage is most audible because
nothing masks it. It is also where standard training objectives are weakest.

### Ablation
| Objective | SI-SDRi | SIR | Leak-WER | DNSMOS |
|---|---|---|---|---|
| SI-SDR only | baseline | baseline | baseline | baseline |
| + `L_suppress` | ≈ −0.2 dB | **+3–5 dB** | ↓ | ≈ |
| + `L_consistency` | ≈ | +1–2 dB | **↓↓** | ≈ |
| + `L_mrstft` | −0.1 dB | ≈ | ≈ | **↑** |
| + `L_silence` | ≈ | +1 dB | ↓ | **↑↑ (perceptual)** |
| **All (SEAVE-SFO)** | ≥ baseline − 0.5 dB | **≥ +5 dB** | **≥ 40% ↓** | **↑** |

**Falsified if:** the full objective does not improve SIR by ≥ 3 dB over SI-SDR-only at a cost of
more than 0.5 dB SI-SDRi.

---

## 5. Contribution 4 — Gated Generative Restoration with Dual Delivery (SEAVE-GGR)

### Claim
> Coupling discriminative extraction with a **selectively applied** generative restoration stage,
> gated by a fidelity estimator and delivered as a labelled dual track, achieves higher perceptual
> quality than either alone **without** the transcription-fidelity regression that unguarded
> generative enhancement causes.

### The gap
Discriminative separation is faithful but artifact-y. Generative restoration is clean but
hallucinates — it resynthesises audio and can produce plausible words that were never spoken. The
literature treats these as competing approaches. Products must not choose blindly, because the
correct choice **depends on the segment and on the user's purpose**.

### The method

```
  discriminative output ŝ
        │
        ▼
  fidelity estimator  φ(ŝ, mixture, enrolment) → [0,1]
        │   features: DNSMOS-proxy head, residual energy, spectral holes,
        │             embedding drift, extraction confidence from the separator
        │
        ├── φ high (clean already) ──────────► pass through, restoration adds nothing
        │
        ├── φ mid  ──────────────────────────► restore, blend: ŝ_out = γ·ŝ_gen + (1−γ)·ŝ
        │
        └── φ very low ──────────────────────► DO NOT restore.
                                               Input too degraded → hallucination risk high.
                                               Mark segment low-confidence in the UI instead.
```

The third branch is the important one and is the opposite of what a quality-maximising system would
do. **Where restoration would help the numbers most, it is most dangerous.** Refusing to restore
there — and disclosing it — is the design position.

### Dual delivery
Both tracks are packaged and shipped:

- **Faithful** (default): discriminative output only. What the model actually recovered.
- **Natural**: gated restoration applied. Pleasant listening.

The UI labels which is active. Downloads and exports state it in the filename and metadata.

### Verification of the safety claim
Run ASR on both tracks and compute **hallucination rate**: words present in the Natural transcript
but absent from both the Faithful transcript and the ground-truth reference, aligned by timestamp.
The gate is tuned so that hallucination rate stays under a hard threshold; if it cannot, the gate
becomes more conservative rather than the threshold moving.

### Ablation
| Variant | DNSMOS | WER | Hallucination rate |
|---|---|---|---|
| Discriminative only | baseline | baseline | 0 by construction |
| Ungated generative | **↑↑** | ↑ (worse) | **high** |
| **Gated (ours)** | **↑** | ≈ baseline | **≤ 0.5%** |

**Falsified if:** gating does not reduce hallucination rate by ≥ 60% relative to ungated while
retaining ≥ 60% of its DNSMOS gain.

---

## 6. Contribution 5 — Cross-Stream Transcript-Consistency Leakage Repair (SEAVE-XL)

### Claim
> Residual leakage can be detected **post hoc** by cross-stream ASR agreement, attributed to its true
> speaker using speaker-embedding and ASD evidence, and suppressed — improving both audio SIR and
> caption attribution accuracy, and yielding a calibrated per-segment trust score.

### The gap
Extraction quality is currently assessed only with reference-based metrics that require ground
truth. At inference time on real user data there *is* no ground truth, so no deployed system can
tell the user where it is unreliable. Everything is presented with uniform, unearned confidence.

### The insight
If the phrase *"the quarterly numbers"* appears at `00:04:12.3–00:04:13.9` in **both** Speaker A's
and Speaker B's transcript, one of them is wrong. It was one utterance. This is a **ground-truth-free
leakage detector**, and it is available for free because we already run ASR per stream.

### The method

```
1. DETECT
   Align word sequences across all K streams by timestamp (IoU ≥ 0.5 on word spans).
   Flag matching word n-grams (n ≥ 2, normalised text, fuzzy match) as leakage candidates.

2. ATTRIBUTE — evidence fusion over the disputed span:
     • speaker-embedding similarity of the span's audio to each enrolment
     • ASD evidence: whose mouth was moving?  (strongest single cue when available)
     • diarization posterior over the span
     • per-stream extraction confidence
   → posterior over "true owner"; if max posterior < threshold, mark UNRESOLVED (do not guess)

3. REPAIR
   For each losing stream: estimate a suppression mask over the disputed T-F region
   (target-absent → residual is interference) and apply with smooth boundaries.
   Remove the leaked words from that stream's caption track.

4. SCORE
   trust(segment, speaker) = f(leakage density, attribution margin, extraction confidence,
                               modality availability, φ from Contribution 4)
   Calibrated on AMI-Eval so that stated confidence matches observed accuracy.
```

The `UNRESOLVED` state is deliberate. When evidence is genuinely ambiguous, the system keeps the
words in **both** transcripts and marks the region as contested, rather than silently deleting a
real utterance. Deleting a word that was actually said is a worse error than showing it twice with a
warning.

### Why this is a real contribution
It is a **closed loop**: ASR (a downstream consumer of separation) feeds a correction signal *back*
into separation. Standard pipelines are strictly feed-forward — separation → ASR, and errors flow
one way. And it produces the calibrated confidence the product needs, which no reference-based
metric can supply at inference time.

### Ablation
| Variant | SIR | Caption attribution F1 | Trust calibration (ECE) |
|---|---|---|---|
| No audit | baseline | baseline | n/a |
| Detect + caption-only repair | ≈ | **↑** | — |
| **Full detect + attribute + audio repair (ours)** | **+1–2 dB** | **↑↑** | **ECE ≤ 0.05** |

**Falsified if:** attribution accuracy on injected synthetic leakage is below 85%, or repair
degrades SI-SDR.

---

## 7. Contribution 6 (systems) — Switchable Multi-Rendition Speaker Audio Delivery

### Claim
> Per-speaker isolated audio can be delivered to a browser as instantly switchable, drift-free
> renditions synchronised to video, without seeking or re-buffering.

Not an ML contribution, but a genuine systems one: no shipping product delivers switchable
per-speaker isolated audio tracks. The design (shared-`AudioContext` equal-power crossfade for short
media, HLS `EXT-X-MEDIA` audio renditions for long) is documented in
[`12-media-pipeline-and-sync.md`](./12-media-pipeline-and-sync.md) and measured against
NFR-PERF-03 (≤ 120 ms switch) and FR-PLAY-05 (≤ 40 ms drift over 10 min).

---

## 8. Contribution matrix

| # | Contribution | Targets gap | Primary metric | Falsification threshold |
|---|---|---|---|---|
| 1 | Self-enrolment | Enrolment unavailable in the wild | SI-SDRi vs oracle | < 0.5 dB over longest-region |
| 2 | Modality-adaptive conditioning | Video not always usable | SI-SDRi under occlusion | < 0.7 dB over hard switch |
| 3 | Suppression-first objective | SI-SDR ≠ audible leakage | SIR, Leak-WER | < 3 dB SIR gain |
| 4 | Gated generative restoration | Quality vs faithfulness | DNSMOS, hallucination rate | < 60% hallucination reduction |
| 5 | Cross-stream leakage repair | No inference-time confidence | Attribution F1, ECE | attribution < 85% |
| 6 | Multi-rendition delivery | No delivery pattern exists | switch latency, drift | > 120 ms / > 40 ms |

---

## 9. Honest positioning

What we are **not** claiming:

- Not a new separator architecture. TF-GridNet is prior work; we adapt it.
- Not state-of-the-art on WSJ0-2mix. We do not optimise for it and will not report a competitive
  number there.
- Not the first audio-visual TSE. That lineage is well established (§3.2 of the research landscape).
- Not solving 4+ speaker separation. The degradation curve will be reported, not hidden.

What we **are** claiming: a coherent, evaluated method for making TSE work on **real uploaded video
with no enrolment, unreliable modalities, and no ground truth at inference time**, delivered as a
system a person can actually use — with each component contribution independently ablated.

That framing is defensible precisely because it is narrow.

---

## 10. Publication framing (optional)

If pursued as a paper:

> **SEAVE: Self-Enrolled Audio-Visual Target Speaker Extraction for In-the-Wild Video**
>
> Sections: (1) enrolment-free TSE via diarization-mined purity-weighted cues; (2) reliability-gated
> modality-adaptive conditioning trained with modality dropout; (3) a suppression-first objective
> targeting audible leakage rather than aggregate distortion; (4) gated generative restoration with
> a hallucination-safety analysis; (5) ground-truth-free leakage detection via cross-stream ASR
> agreement, with calibrated confidence.
>
> Venues: Interspeech, ICASSP, WASPAA. Systems angle (Contribution 6) fits ACM MM or a demo track.
