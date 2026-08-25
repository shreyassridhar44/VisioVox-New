# 05 — ML Architecture

The processing pipeline in full. Each stage lists its inputs, outputs, model, failure mode and
fallback. Stages are **idempotent** and independently retryable (FR-PIPE-14).

---

## 1. Pipeline overview

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ S0  INGEST            demux, decode, normalise                     CPU      │
 │ S1  ENHANCE           dereverb → denoise                           GPU      │
 │ S2A ANALYSE·AUDIO     VAD → diarize → overlap detect               GPU      │
 │ S2B ANALYSE·VIDEO     detect → track → landmark → ASD              GPU  ║ ∥ │
 │ S3  FUSE              bind voice clusters ↔ face tracks            CPU      │
 │ S4  SELF-ENROL        mine cues, score purity          [NOVEL 1]   GPU      │
 │ S5  EXTRACT           route + AV-TSE, per speaker    [NOVEL 2,3]   GPU      │
 │ S6  RESTORE           fidelity gate + generative        [NOVEL 4]   GPU     │
 │ S7  TRANSCRIBE        ASR + word alignment, per stream             GPU      │
 │ S8  AUDIT             cross-stream leakage repair       [NOVEL 5]   GPU     │
 │ S9  PACKAGE           loudness, encode, HLS, captions              CPU      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

`S2A` and `S2B` run in parallel. Everything else is sequential. Total GPU stages: 7.

---

## 2. S0 — Ingest

**In:** uploaded media object · **Out:** normalised media set

| Artifact | Spec | Purpose |
|---|---|---|
| `video.mp4` | H.264, original resolution, capped 1080p | Playback source |
| `analysis.wav` | 16 kHz, mono, PCM s16 | All ML stages |
| `reference.wav` | 48 kHz, stereo (or original) | Final packaging quality |
| `frames/` | 25 fps JPEG or decoded on demand | Vision stages |
| `probe.json` | ffprobe output | Validation record |

```bash
# analysis audio — resample with a high-quality filter, do not use the default
ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 -af "aresample=resampler=soxr:precision=28" \
       -c:a pcm_s16le analysis.wav

# reference audio — preserve original fidelity
ffmpeg -i in.mp4 -vn -ar 48000 -c:a pcm_s24le reference.wav
```

**Critical:** all timing downstream is expressed in **samples at 16 kHz** internally and converted to
seconds only at the API boundary. Float-seconds arithmetic accumulates error over a 60-minute
timeline and is the classic source of caption drift.

**Sandboxing:** every ffmpeg/ffprobe invocation runs under the constraints in
[`15-security.md`](./15-security.md) §4. Non-negotiable.

**Failure modes:** no audio stream → reject (FR-UPL-05). Variable frame rate → force CFR at 25 fps.
Corrupt container → attempt `-err_detect ignore_err` remux once, then reject.

---

## 3. S1 — Enhancement front-end

**In:** `analysis.wav` · **Out:** `enhanced.wav` + `enhancement_stats.json`

| Step | Model | Notes |
|---|---|---|
| Dereverberation | WPE (`nara_wpe`), 3 iterations, taps 10, delay 3 | Unsupervised — no training data required, no domain risk |
| Denoising | DeepFilterNet3 | Fast; preserves speech structure well |

**Why before extraction, not after:** extraction models are trained predominantly on anechoic
mixtures. Reverb is *distribution shift*, and shifting the input toward the training distribution is
worth more than cleaning up the output afterwards. Ablated in
[`08-evaluation-protocol.md`](./08-evaluation-protocol.md) §5.

**Guard:** if the denoiser's estimated SNR improvement is < 1 dB, skip it — needless processing of
already-clean audio costs quality. Recorded in `enhancement_stats.json`.

**Note:** the *enhanced* signal feeds analysis and extraction. The **original** `reference.wav` is
retained so that packaging can optionally blend back some natural room tone; fully dry isolated
speech sounds unnatural against the video.

---

## 4. S2A — Audio analysis

**In:** `enhanced.wav` · **Out:** `diarization.rttm`, `overlap.json`, `vad.json`, `speaker_clusters.json`

| Step | Model | Output |
|---|---|---|
| VAD | Silero VAD (or pyannote segmentation) | speech/non-speech spans |
| Diarization | pyannote.audio 3.x | `(start, end, speaker_k)` spans |
| Overlap detection | pyannote overlap-aware segmentation | per-frame overlap probability |
| Embeddings | ReDimNet-B2 | per-segment 192-d vectors + cluster centroids |
| Count estimation | clustering + constraints | K ∈ [1, 4] with confidence |

**Speaker-count estimation** is more delicate than it looks. Diarization clustering is sensitive to
the threshold and will over-segment on channel changes or under-segment on similar voices. We use:

```python
K = argmax_k [ silhouette(k) · prior(k) ]     # k ∈ 1..4
# priors from the expected use case: 2 and 3 speakers most likely
# then cross-check against the face-track count from S2B in the FUSE stage.
# disagreement → prefer the audio estimate, lower confidence, surface to the UI
```

If K > 4, cap at 4, keep the 4 largest by speaking time, and flag `speaker_count_capped` for a UI
warning (charter non-goal).

**Overlap statistics** computed here drive S5's router and are reported in the UI as a
"difficulty" indicator — an honest signal of how hard this recording was.

---

## 5. S2B — Video analysis

**In:** `frames/` · **Out:** `face_tracks.json`, `asd_scores.json`, `thumbnails/`

| Step | Model | Notes |
|---|---|---|
| Detection | SCRFD-10G | Every frame at 25 fps; robust to profile and small faces |
| Tracking | ByteTrack | IoU + appearance; handles brief occlusion |
| Landmarks | 68-point (or 5-point + affine) | Needed for stable mouth-ROI crops |
| Mouth ROI | 96×96 grayscale, landmark-aligned | Standard AVSR frontend input |
| ASD | LoCoNet | Per-face-track per-frame speaking probability |
| Quality | sharpness, size, yaw/pitch, occlusion | Feeds `visual_conf_t` (Novelty 2) |

**Mouth-ROI extraction** must be alignment-stable. Crops that jitter frame-to-frame inject motion
the visual encoder reads as articulation. Stabilise with a smoothed similarity transform from the
landmarks (EMA over ±5 frames) before cropping.

**Scene-cut detection** runs alongside: a cut breaks track continuity, so tracks are split at cuts
and re-linked in S3 by face embedding rather than by IoU.

**Failure modes:**
| Condition | Behaviour |
|---|---|
| No faces detected | `visual_available = false`; whole pipeline goes audio-only |
| Faces detected, ASD low-confidence throughout | face tracks kept for thumbnails, not used for conditioning |
| More face tracks than speakers | non-speaking faces (audience, photos on a wall) filtered by ASD |
| Fewer face tracks than speakers | off-screen speakers → audio-only conditioning for those speakers |

---

## 6. S3 — Cross-modal fusion

**In:** `diarization.rttm`, `face_tracks.json`, `asd_scores.json` · **Out:** `speaker_registry.json`

Binds voice clusters to face tracks. Formulated as a bipartite assignment:

```
cost(voice_k, face_track_f) = − Σ_t  1[diar(t) = k] · asd_score(f, t) · w(t)

  where w(t) down-weights overlapped frames — during overlap, diarization
  attribution is least reliable, so it should contribute least evidence.

solve with Hungarian algorithm; accept assignments above a confidence floor.
```

**Output — the Speaker Registry**, the canonical object every later stage and the whole UI depends on:

```jsonc
{
  "speakers": [{
    "id": "spk_1",
    "label": "Speaker 1",              // user-renameable
    "color_token": "spk-1",
    "face_track_ids": ["ft_3"],        // empty ⇒ audio-only
    "thumbnail_key": "…/spk_1.webp",   // best frontal, sharp, speaking frame
    "speaking_seconds": 214.6,
    "speaking_ratio": 0.41,
    "binding_confidence": 0.93,
    "modality": "audiovisual"          // audiovisual | audio_only | visual_only
  }],
  "count_confidence": 0.88,
  "overlap_ratio": 0.17,
  "difficulty": "moderate"
}
```

**Conflict resolution:** audio and video disagree on count more often than either fails alone.
Rule — trust the audio count (diarization is the more reliable counter), attach faces where
binding confidence permits, leave the rest audio-only, and lower `count_confidence`. Never invent a
speaker to match a face.

---

## 7. S4 — Self-enrolment · **Novelty 1**

**In:** `enhanced.wav`, `speaker_registry.json`, `frames/` · **Out:** `enrolments/spk_k.npz`

Method specified in [`04-novelty.md`](./04-novelty.md) §2. Implementation notes:

```python
def mine_enrolment(spk, diar, overlap, audio, faces, asd):
    cands = [r for r in diar.regions(spk)
             if r.duration >= 1.5
             and overlap.mean_prob(r) < 0.15
             and vad.speech_ratio(r) > 0.7]

    for r in cands:
        r.purity = (0.40 * (1 - overlap.mean_prob(r))
                  + 0.25 * snr_norm(audio, r)
                  + 0.15 * vad.speech_ratio(r)
                  + 0.20 * cos(embed(audio, r), spk.centroid)
                  - 0.10 * reverb_estimate(audio, r))

    top = topk_until(cands, key='purity', min_total_seconds=8.0, max_regions=6)
    if total_seconds(top) < 1.5:
        return Enrolment(audio_cue=None, audio_conf=0.0, ...)   # → visual-only

    embs = [embed(audio, r) for r in top]
    w    = softmax([r.purity for r in top])
    e    = l2norm(sum(wi * ei for wi, ei in zip(w, embs)))

    audio_conf = calibrate(total_seconds(top), mean_purity(top), emb_dispersion(embs))

    visual = extract_mouth_rois(faces, spk.face_track_ids, top) if spk.face_track_ids else None
    visual_conf = visual_quality(visual) if visual else 0.0

    return Enrolment(e, audio_conf, visual, visual_conf)
```

`audio_conf` and `visual_conf` are **calibrated** against held-out data so that they predict actual
extraction quality — they are consumed as gate inputs in S5, not shown as decoration.

**Determinism:** enrolment mining must be deterministic given the same inputs (fixed seeds, stable
sort by `(purity, start_sample)`) so that job re-runs reproduce and caching is sound.

---

## 8. S5 — Extraction · **Novelty 2 & 3**

**In:** `enhanced.wav`, `enrolments/`, mouth ROIs, `overlap.json`
**Out:** `isolated/spk_k.faithful.wav` (full length), `confidence/spk_k.json`

Runs **once per speaker**.

### 8.1 Router — single-talker passthrough (F11)

```
for each region r in the timeline:
    if overlap_prob(r) < 0.10 and diar(r) == k:
        out[r] = enhanced[r]                  # PASSTHROUGH — do not process clean audio
    elif diar(r) does not include k:
        out[r] = silence                      # target not speaking
    else:
        out[r] = AV_TSE(enhanced[r], enrolment_k, rois_k)
crossfade 30 ms equal-power at every region boundary
```

Three reasons this matters more than it appears:
1. **Quality** — running a separator on already-clean single-speaker audio *removes* quality.
2. **Cost** — real conversations are 5–20% overlapped, so 80–95% of the timeline skips the GPU.
   This is the largest single lever on NFR-PERF-01.
3. **Faithfulness** — the majority of what the user hears is untouched original audio.

Regions are dilated by 200 ms around overlap boundaries so the extractor has context and the
crossfade lands in processed territory.

### 8.2 Model — SEAVE

```
 mixture (16 kHz)                 enrolment e_spk (192-d)      mouth ROIs (96×96 @25fps)
      │                                    │                             │
      ▼                                    ▼                             ▼
 ┌──────────┐                     ┌────────────────┐          ┌────────────────────┐
 │ STFT     │                     │ Linear → 256-d │          │ Visual frontend    │
 │ 512/128  │                     └────────┬───────┘          │ 3D-conv + ResNet18 │
 └────┬─────┘                              │                  │ (pretrained, frozen│
      │                                    │                  │  early training)   │
      │                                    │                  └─────────┬──────────┘
      │                                    │                            │ 25 fps
      │                                    │                            ▼
      │                                    │                  ┌────────────────────┐
      │                                    │                  │ upsample → 125 Hz  │
      │                                    │                  │ (match STFT frames)│
      │                                    │                  └─────────┬──────────┘
      │                                    │                            │
      │                              ┌─────▼────────────────────────────▼──────┐
      │                              │  RELIABILITY-GATED FUSION  [Novelty 2]  │
      │                              │  c_t = α·W_a·e + β_t·W_v·v_t + b        │
      │                              └─────────────────┬───────────────────────┘
      ▼                                                │
 ┌────────────────────────────────────────────────────▼──────────────────────────┐
 │  TF-GRIDNET BACKBONE — 6 blocks                                               │
 │   per block:  intra-frame (freq) BLSTM  →  inter-frame (time) BLSTM           │
 │            →  full-band self-attention                                        │
 │            →  FiLM(c_t):  h ← γ(c_t) ⊙ h + δ(c_t)      ← conditioning injected│
 │            →  cross-attention(audio → visual), scaled by β_t                  │
 └────────────────────────────────────────────────────┬──────────────────────────┘
                                                      ▼
                                         ┌────────────────────────┐
                                         │ complex mask → iSTFT   │
                                         └───────────┬────────────┘
                                                     │
                                    ┌────────────────┴───────────────┐
                                    ▼                                ▼
                              ŝ (waveform)              confidence head → conf_t
```

**Reference configuration** (tuned for a 12 GB VRAM budget):

| Param | Value |
|---|---|
| Sample rate | 16 kHz |
| STFT | 512 window / 128 hop, Hann |
| Blocks | 6 |
| Hidden dim `D` | 96 |
| LSTM hidden | 192 |
| Attention heads | 4 |
| Chunk length (train) | 4 s |
| Chunk length (infer) | 10 s, 2 s overlap, cross-faded |
| Params | ≈ 14 M (+ 11 M frozen visual frontend) |
| Precision | bf16 autocast, fp32 master weights |

**Inference chunking has no permutation problem** — every chunk is conditioned on the same
enrolment, so all chunks emit the same speaker. Overlap-add with a 2 s cross-faded region removes
boundary discontinuities. This is the concrete payoff of choosing TSE (see
[`03-research-landscape.md`](./03-research-landscape.md) §1.1).

### 8.3 Loss

Per [`04-novelty.md`](./04-novelty.md) §4:

```python
L = (1.0 * si_sdr_loss(s_hat, s)
   + 0.3 * suppression_loss(s_hat, interferers, tau=-10.0)   # hinged
   + 0.2 * (1 - cos(spk_encoder(s_hat), e_spk))
   + 0.5 * mrstft_loss(s_hat, s)                             # [512,1024,2048] windows
   + 0.2 * silence_loss(s_hat, s, target_silence_mask))
```

Weights are the starting point; tuned by the sweep in
[`07-training-playbook.md`](./07-training-playbook.md) §6.

### 8.4 Confidence head

A small head over the final block predicts per-frame extraction confidence, trained to regress the
true per-frame SI-SDR on training data. At inference this gives a ground-truth-free quality
estimate — consumed by S6's gate, S8's attribution, and the UI (FR-PIPE-09).

### 8.5 Fallbacks
| Condition | Behaviour |
|---|---|
| `audio_conf < 0.3` and visual available | visual-only conditioning (α → 0) |
| `visual_conf_t < 0.3` for a frame | audio-only conditioning for that frame (β_t → 0) |
| Both low | passthrough diarization-masked audio; mark segment low-confidence |
| OOM on a chunk | halve chunk length and retry once, then fall back to the audio-only path |

---

## 9. S6 — Gated restoration · **Novelty 4**

**In:** `isolated/spk_k.faithful.wav`, mixture, enrolment
**Out:** `isolated/spk_k.natural.wav`, `restoration_report.json`

Per [`04-novelty.md`](./04-novelty.md) §5.

| φ (fidelity) | Action |
|---|---|
| ≥ 0.75 | passthrough (already clean; restoration can only hurt) |
| 0.35 – 0.75 | restore, blend with γ = f(φ) rising as φ falls, capped at 0.7 |
| < 0.35 | **do not restore** — hallucination risk; mark segment low-confidence |

**Model:** speaker-conditioned generative restoration — cleaned mel + speaker embedding → BigVGAN or
Vocos vocoder. Implemented as a **pluggable stage**; if quality or timeline doesn't permit, it is
disabled by a feature flag and the product ships Faithful-only. Nothing else depends on it.

**Safety verification:** hallucination-rate check per [`04-novelty.md`](./04-novelty.md) §5, run in
CI against a fixed eval subset. Regression here blocks release.

---

## 10. S7 — Transcription

**In:** each isolated track · **Out:** `captions/spk_k.json`

| Step | Model | Notes |
|---|---|---|
| ASR | Whisper large-v3 (faster-whisper / CTranslate2, int8_float16) | Robust to separation artifacts |
| Alignment | WhisperX wav2vec2 forced alignment | Word-level timestamps |
| VAD gating | reuse S2A VAD | Suppress hallucinated text on silence |

**Important:** Whisper hallucinates on silence and low-energy audio — a well-known failure that will
be triggered constantly here, because isolated tracks are mostly silence. Mitigations, all required:

1. Transcribe **only** VAD-positive spans of the isolated track, not the whole file
2. `condition_on_previous_text=False` (prevents runaway repetition loops)
3. Drop segments with `no_speech_prob > 0.6` or `avg_logprob < -1.0`
4. Drop segments whose span has near-zero energy in the isolated track
5. Cross-check against S2A diarization: text at a timestamp where the speaker wasn't speaking is dropped

Transcribe the **Faithful** track, always. Captions must reflect what was actually recovered, not
what a generative model produced — otherwise Novelty 4's safety argument is undermined at the source.

```jsonc
{
  "speaker_id": "spk_1",
  "language": "en",
  "segments": [{
    "start_ms": 12340, "end_ms": 15780,
    "text": "the quarterly numbers came in higher",
    "confidence": 0.94,
    "trust": 0.88,                       // from S8
    "words": [{ "w": "the", "s": 12340, "e": 12480, "c": 0.99 }]
  }]
}
```

---

## 11. S8 — Leakage audit and repair · **Novelty 5**

**In:** all `isolated/*.wav`, all `captions/*.json`, `asd_scores.json`, enrolments
**Out:** repaired audio + captions, `trust_scores.json`

Method in [`04-novelty.md`](./04-novelty.md) §6.

Implementation notes:
- Normalise text before matching (lowercase, strip punctuation, expand numerals) — otherwise
  *"quarterly"* vs *"Quarterly,"* fails to match.
- Require n ≥ 2 word matches. Single common words (*"the"*, *"yeah"*, *"okay"*) genuinely do occur
  simultaneously in real conversation and are not leakage.
- Attribution posterior below 0.6 → `UNRESOLVED`: keep in both transcripts, mark contested, do not
  modify audio. **Never delete a real utterance to make the output look cleaner.**
- Audio repair is a gentle spectral suppression over the disputed T-F region with smooth boundaries;
  aggressive gating sounds worse than the leakage it removes.

**Skip condition:** if K = 1 or measured overlap ratio < 2%, skip the stage entirely.

---

## 12. S9 — Packaging

**In:** all tracks and captions · **Out:** the delivery bundle

| Step | Detail |
|---|---|
| Loudness | `ffmpeg -af loudnorm=I=-16:TP=-1.0:LRA=11` two-pass, per track (F13) |
| Length assertion | every track sample-count == video duration × sr, **hard assert** |
| Encode | AAC-LC 128 kbps mono per speaker track |
| Package | CMAF/fMP4, HLS multivariant with per-speaker `EXT-X-MEDIA:TYPE=AUDIO` renditions |
| Captions | WebVTT per speaker + word-timed JSON |
| Extras | waveform peaks (per speaker), thumbnails, speaker registry, manifest |

The **length assertion** is a hard gate. A single-sample length mismatch between tracks becomes
accumulating A/V drift in the player, and it is very easy to introduce via resampling rounding.
Fail the job rather than ship a drifting bundle.

Full delivery spec in [`12-media-pipeline-and-sync.md`](./12-media-pipeline-and-sync.md).

---

## 13. Resource budget

Reference: 10-minute video, 2 speakers, 15% overlap, RTX 4070-class GPU.

| Stage | Time | VRAM | Notes |
|---|---|---|---|
| S0 Ingest | 25 s | — | CPU, I/O bound |
| S1 Enhance | 40 s | 2 GB | |
| S2A Audio analysis | 55 s | 3 GB | ∥ with S2B |
| S2B Video analysis | 150 s | 4 GB | Dominated by per-frame face detect |
| S3 Fuse | 2 s | — | |
| S4 Self-enrol | 15 s | 2 GB | |
| S5 Extract | 90 s | 6 GB | ×2 speakers; passthrough saves ~80% |
| S6 Restore | 45 s | 4 GB | Only on gated segments |
| S7 Transcribe | 70 s | 5 GB | ×2 speakers, VAD-gated |
| S8 Audit | 20 s | 2 GB | |
| S9 Package | 45 s | — | CPU |
| **Total** | **≈ 9 min** | peak 6 GB | RTF ≈ 0.9× |

Meets NFR-PERF-02 (≤ 8 min) only marginally. Optimisations if needed, in priority order:
1. Face detection at 12.5 fps with interpolation (halves the largest cost)
2. Batch multi-speaker extraction in one forward pass
3. Distil the extractor
4. Parallelise S7 across speakers on separate GPUs

---

## 14. Model registry

| Stage | Model | Source | Trained by us? | Licence check |
|---|---|---|---|---|
| Dereverb | WPE | `nara_wpe` | No | MIT ✅ |
| Denoise | DeepFilterNet3 | HF | No | MIT/Apache ✅ |
| VAD | Silero VAD | HF | No | MIT ✅ |
| Diarization | pyannote 3.1 | HF (gated) | No | MIT, gated access ⚠️ |
| Speaker emb. | ReDimNet-B2 | HF | Fine-tuned | MIT ✅ |
| Face detect | SCRFD-10G | InsightFace | No | Apache ⚠️ non-commercial variants exist — verify |
| Tracking | ByteTrack | — | No | MIT ✅ |
| ASD | LoCoNet | GitHub | Fine-tuned | check ⚠️ |
| Visual frontend | AVSR ResNet-3D stem | Auto-AVSR | Fine-tuned | check ⚠️ |
| **Extractor** | **SEAVE (TF-GridNet)** | **ours** | **Yes** | ours |
| Restoration | BigVGAN / Vocos | HF | Fine-tuned | check ⚠️ |
| ASR | Whisper large-v3 | OpenAI | No | MIT ✅ |
| Alignment | wav2vec2 aligner | HF | No | ✅ |

⚠️ = licence must be verified before any commercial deployment. Tracked as a release-gate item in
[`19-testing-strategy.md`](./19-testing-strategy.md) §9.

---

## 15. Artifact manifest (the contract)

Written at the end of S9. This is the **interface between the ML pipeline and the application** —
the frontend and API depend on this schema, not on pipeline internals. Versioned; changes require a
version bump and a migration note.

```jsonc
{
  "manifest_version": "1.0",
  "job_id": "job_01HX…",
  "duration_ms": 612480,
  "speakers": [ /* speaker registry */ ],
  "tracks": {
    "mixed":  { "hls": "…/mixed.m3u8", "wav": "…/mixed.wav" },
    "spk_1": {
      "faithful": { "hls": "…/spk_1_f.m3u8", "wav": "…/spk_1_f.wav", "peaks": "…/spk_1.peaks.json" },
      "natural":  { "hls": "…/spk_1_n.m3u8", "wav": "…/spk_1_n.wav" }
    }
  },
  "captions": { "spk_1": { "vtt": "…/spk_1.vtt", "json": "…/spk_1.json" } },
  "master_playlist": "…/master.m3u8",
  "metrics": {
    "overlap_ratio": 0.17,
    "mean_confidence": { "spk_1": 0.86, "spk_2": 0.81 },
    "leakage_repairs": 12,
    "unresolved_spans": 3
  },
  "warnings": ["speaker_2_no_face_track"],
  "pipeline_version": "seave-1.0.3",
  "model_versions": { "extractor": "seave-tfgridnet-v1.0.3", "asr": "whisper-large-v3" }
}
```
