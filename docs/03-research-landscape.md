# 03 — Research Landscape

> Why the architecture is what it is. Read this before [`05-ml-architecture.md`](./05-ml-architecture.md).
> Reported numbers are from the cited papers on their own benchmarks; they are **not** comparable
> across different datasets, and none of them predict real-world performance. See §7.

---

## 1. Problem taxonomy

Three distinct problems are frequently conflated. Getting this distinction right is the single most
consequential decision in the project.

| | **Blind Source Separation (BSS)** | **Target Speaker Extraction (TSE)** | **Diarization** |
|---|---|---|---|
| Input | mixture | mixture + *cue for one speaker* | mixture |
| Output | K separated streams | 1 stream (the cued speaker) | timestamps + labels |
| Output count | fixed at train time | 1, invoke K times | variable |
| Identity binding | **none — permutation-invariant** | **bound to the cue** | cluster labels |
| Training | Permutation-Invariant Training | direct, no PIT needed | clustering / EEND |
| Long-form stitching | **hard** (channel order drifts) | **trivial** | n/a |
| Fits "click a speaker" UX | poorly | **exactly** | provides the cue |

**The decisive point.** The product's interaction model is *"the user names a speaker, the system
returns that speaker's audio."* That is the literal definition of TSE. BSS answers a different
question — "give me all the sources in unknown order" — and then requires an error-prone
post-hoc identity-assignment step to answer the question actually being asked.

### 1.1 The permutation problem, concretely

PIT trains a BSS model by computing the loss under all K! output orderings and back-propagating the
best one. This makes the model excellent at *separating* and completely indifferent to *ordering*.

For a 6-minute video processed in 5-second windows (72 windows), the model emits:

```
window 1:  [ch0 = Alice, ch1 = Bob]
window 2:  [ch0 = Bob,   ch1 = Alice]     ← arbitrary flip
window 3:  [ch0 = Bob,   ch1 = Alice]
...
```

To build a coherent Alice track you must solve a 72-step assignment problem. The usual heuristics:

- **Overlap-region cross-correlation** — process windows with overlap and match by correlation in
  the shared region. Fails when the shared region is silent for one speaker.
- **Speaker-embedding matching** — embed each chunk output and match to a running centroid. This
  is more robust, but note what it implies: *you now need speaker embeddings and a matching step
  anyway.* At which point you have reconstructed a worse version of TSE, with an extra failure mode.

Errors here are **catastrophic rather than graceful**: a single mis-assignment doesn't degrade
quality slightly, it puts the wrong person's voice in the track, which is exactly the failure the
product exists to prevent (see the journalist persona in [`00-charter.md`](./00-charter.md) §5).

TSE has no such step. The cue *is* the identity.

---

## 2. Blind separation — state of the art

| Model | Year | WSJ0-2mix SI-SDRi | Notes |
|---|---|---|---|
| Conv-TasNet | 2019 | ~15.3 dB | First to beat ideal masks; small and fast; still a fine cheap baseline |
| DPRNN | 2020 | ~18.8 dB | Dual-path: intra/inter-chunk RNN — introduced the pattern everything since uses |
| SepFormer | 2021 | ~22.3 dB | Dual-path transformer; SpeechBrain recipe; the original roadmap's choice |
| TF-GridNet | 2023 | ~23.5 dB | Time-frequency domain, dual-path + full-band attention; strong on reverberant data |
| MossFormer2 | 2023–24 | ~24.1 dB | Joint attention + RNN-free recurrence |
| SPMamba / Mamba-TasNet | 2024–25 | competitive | State-space backbones; better long-context scaling cost |

**Key observations:**

1. Progress on WSJ0-2mix has largely saturated — the last several dB came at large complexity cost.
2. **The benchmark is anechoic, clean, fully overlapped, exactly 2 speakers.** WHAMR! (noisy +
   reverberant) numbers drop by roughly a third to a half. Real conversational recordings are worse
   still.
3. TF-GridNet's advantage is largest precisely where our data lives: **reverberant and noisy**
   conditions. Its time-frequency formulation handles reverb tails better than time-domain masking.

> **Design consequence:** we adopt the **TF-GridNet backbone** but not the BSS framing.
> See [ADR-0002](./adr/0002-tfgridnet-backbone.md).

---

## 3. Target speaker extraction

### 3.1 Audio-only TSE

| Model | Cue | Idea |
|---|---|---|
| VoiceFilter (2019) | d-vector | Concatenate speaker embedding into a spectrogram-masking network |
| SpeakerBeam / TD-SpeakerBeam (2019–20) | enrolment utterance | Adaptation layer multiplicatively modulated by the speaker embedding |
| SpEx / SpEx+ (2020) | enrolment | Multi-scale time-domain encoder, jointly trained speaker encoder |
| WeSep (2024) | enrolment | Modern open toolkit; strong baselines and recipes |

**Fundamental limitation of audio-only TSE:** it fails hardest when the target and interferer have
similar voices — same gender, similar pitch, similar accent. That's a large fraction of real
meetings, and it is the *worst* failure mode for our product because the leakage is not just
audible, it's **plausible** — a listener cannot tell it's leakage.

### 3.2 Audio-visual TSE — the important branch

The insight: **lip motion is unaffected by acoustic overlap.** When two voices are perfectly mixed
in the audio, the video still shows unambiguously which mouth produced which phoneme. It is an
*orthogonal* information channel, not a redundant one.

| Work | Contribution |
|---|---|
| Looking to Listen (Ephrat et al., 2018) | Established AV separation; face-embedding conditioned masking |
| AV-ConvTasNet (2019) | Time-domain AV separation |
| VisualVoice (2021) | Joint face-appearance + lip-motion cues; cross-modal consistency loss |
| MuSE (2021) | Multi-modal speaker extraction with visual + self-enrolled audio cues |
| AV-SepFormer (2023) | Transformer AV extraction; cross-attention fusion of visual and audio streams |
| IIANet (2024) | Intra/inter-attention AV separation, strong efficiency/quality trade-off |

**Consistently reported findings across this literature:**

- AV conditioning yields large gains over audio-only in **same-gender** and **high-overlap** conditions
- AV models are far more **robust to unknown speaker count** — you extract who you can see
- Visual cues degrade gracefully with occlusion, profile views and low resolution, but do degrade
- **Both cues together beat either alone**, and the gap widens as SNR worsens

> **Design consequence:** AV-TSE is the primary architecture. But visual availability is not
> guaranteed in real uploads, which motivates our **modality-adaptive gate**
> ([`04-novelty.md`](./04-novelty.md) §3).

### 3.3 The enrolment problem — and our opening

Every TSE system needs a cue. In the literature this is almost always a **clean enrolment
recording** of the target speaker, supplied externally.

**This is a product blocker.** A journalist uploading an interview has no clean enrolment for the
interviewee. Asking users to supply one destroys the "upload and it works" experience, which is the
entire value proposition.

Some works (notably MuSE) use self-enrolment ideas within an utterance. What is not well explored is
a **full pipeline that mines enrolments from long-form conversational video** using diarization to
find single-talker regions, scores their purity, and adapts the conditioning regime per speaker
based on what it found.

> **Design consequence:** this is novelty axis #1. See [`04-novelty.md`](./04-novelty.md) §2.

---

## 4. Supporting stages

### 4.1 Diarization

| System | Notes |
|---|---|
| pyannote.audio 3.x | Current practical default; end-to-end segmentation + embedding clustering; handles overlap; gated HF models |
| NeMo `diar_msdd` | Multi-scale diarization decoder; strong; heavier stack |
| DiariZen / Sortformer | 2024–25 end-to-end approaches |
| EEND-VC family | End-to-end with vector clustering; good overlap handling |

We use **pyannote 3.x** for its overlap-aware segmentation, which we need for two purposes beyond
labelling: (a) finding the pure single-talker regions for self-enrolment, and (b) driving the
single-talker passthrough router.

### 4.2 Speaker embeddings

| Model | Notes |
|---|---|
| ECAPA-TDNN | Robust workhorse; widely available (SpeechBrain, WeSpeaker) |
| ECAPA2 | Hybrid; stronger |
| ReDimNet | Current top-tier on VoxCeleb benchmarks; several size variants |

Used for (a) diarization clustering, (b) TSE conditioning, (c) the speaker-consistency loss term,
and (d) leakage attribution. Chosen at the size that fits the VRAM budget; ReDimNet-B2/B3 class.

### 4.3 Face detection, tracking and active speaker detection

| Stage | Choice | Alternatives |
|---|---|---|
| Detect | SCRFD | RetinaFace, YOLOv8-face, MediaPipe (fastest, weakest on profile) |
| Track | ByteTrack | OC-SORT, DeepSORT |
| ASD | LoCoNet | TalkNet-ASD, Light-ASD (fastest), ASDNet |

ASD is trained/evaluated on AVA-ActiveSpeaker. **LoCoNet** models long-term intra-speaker and
short-term inter-speaker context, which matters for multi-face conversational scenes; Light-ASD is
the fast fallback for CPU-bound or high-throughput paths.

ASD serves three purposes here — only the first is in the original roadmap:
1. Face↔voice binding for the UI (thumbnails on the speaker selector)
2. **Selecting which face track supplies the visual conditioning stream to the extractor**
3. Evidence for leakage attribution in the audit stage

### 4.4 Visual speech representation

For lip-motion conditioning we need an encoder over mouth-ROI crops:

| Option | Notes |
|---|---|
| AV-HuBERT | Self-supervised AV speech representations; strong lip features; the quality choice |
| Auto-AVSR / RAVEn frontends | ResNet-18 3D-conv stem + transformer; well-documented recipes |
| Lightweight 3D-CNN stem | Trained from scratch; cheapest, weakest |

Plan: start with a **frozen pretrained visual frontend** (ResNet-3D stem from an AVSR recipe), then
optionally unfreeze the top blocks in late fine-tuning if VRAM allows.

### 4.5 Speech enhancement front-end

| Stage | Choice |
|---|---|
| Dereverberation | WPE (`nara_wpe`) — unsupervised, no training data needed, robust |
| Denoising | DeepFilterNet3 (fast, real-time class) or a Demucs-based denoiser (heavier, cleaner) |

Placed **before** extraction. Rationale: separation models trained largely on anechoic data are
brittle to reverb; removing reverb moves the input closer to the training distribution. This is a
distribution-shift mitigation, not merely a cosmetic cleanup. Ablated in
[`08-evaluation-protocol.md`](./08-evaluation-protocol.md).

### 4.6 ASR and alignment

| Component | Choice | Notes |
|---|---|---|
| ASR | Whisper large-v3 via faster-whisper (CTranslate2) | Robust to the residual artifacts extraction leaves behind — a genuinely important property here |
| Word alignment | WhisperX / forced aligner | Whisper's native segment timestamps are too coarse for click-to-seek transcripts |
| Fast alternative | NVIDIA Parakeet-TDT | English-only, much faster, native word timings |

**Why Whisper specifically:** it was trained on 680k hours of extremely heterogeneous web audio,
which makes it unusually tolerant of the "watery"/"metallic" artifacts characteristic of
discriminative separation output. Cleaner-sounding models trained on clean speech often do worse
here.

### 4.7 Generative speech restoration

Discriminative extraction produces faithful-but-artifacted audio. Generative restoration produces
clean-but-potentially-hallucinated audio. Relevant work:

| Work | Approach |
|---|---|
| Miipher (2023) | Parametric restoration conditioned on SSL features + speaker embedding + text |
| Universal speech enhancement (diffusion/flow) | Resynthesis-based restoration |
| Vocos / BigVGAN | High-fidelity neural vocoders for resynthesis |
| Genhancer and similar (2024–25) | Generative enhancement of discriminative outputs |

**The trade-off nobody resolves cleanly:** generative models can and do invent plausible words when
the input is severely degraded. For a journalist quoting a source, a hallucinated word is a
catastrophic failure — worse than audible artifacts.

> **Design consequence:** we do not choose. We ship **both tracks** with a fidelity gate deciding
> which is *default*, and always label which one is playing. Novelty axis #4;
> [ADR-0005](./adr/0005-gated-generative-restoration.md).

---

## 5. Competitive landscape (product)

| Product | What it does | What it doesn't |
|---|---|---|
| Otter / Fireflies / Zoom AI | Speaker-labelled transcripts | Audio stays mixed |
| Adobe Podcast Enhance | Denoise + de-reverb a mix | No per-speaker decomposition |
| Auphonic | Levelling, loudness, noise | No separation |
| LALAL.AI / Moises | Music stems; a "voice/noise" split | Not speaker-vs-speaker |
| Descript | Studio Sound + transcript editing; multitrack **if recorded multitrack** | Cannot un-mix a single mic |
| Krisp / NVIDIA Broadcast | Real-time noise + voice focus (own mic) | Not post-hoc, not multi-speaker decomposition |
| Research demos (AV-SepFormer, VisualVoice) | Strong separation on clips | No long-form identity stability, no product, no captions, no UI |

**Nobody occupies the intersection of:** long-form video · per-speaker isolated audio · interactive
in-player speaker switching · per-speaker captions · confidence disclosure. That intersection is
the product.

---

## 6. Where the research gap actually is

The academic frontier is *"higher SI-SDR on WSJ0-2mix."* The **deployment** frontier — which is
where this project lives — is a different and less-crowded set of problems:

1. **Enrolment acquisition in the wild** — TSE assumes a cue that real users cannot provide
2. **Long-form identity stability** — benchmarks are 4–10 s utterances; products are 60-minute recordings
3. **Modality reliability** — literature assumes the face is always visible; real video has occlusion, cuts, off-screen speakers
4. **Objective/perception mismatch** — SI-SDR is optimised; *listenability* is what users judge
5. **Leakage disclosure** — no system tells the user *where* it is unreliable
6. **Delivery** — no established pattern for shipping N switchable audio renditions to a browser in sync with video

Every one of our five novelty axes targets one of these. That is not a coincidence — it is how they
were chosen. See [`04-novelty.md`](./04-novelty.md).

---

## 7. Reading the numbers honestly

Three cautions that must appear in the final report:

1. **Benchmark numbers do not transfer.** A 23 dB SI-SDRi on WSJ0-2mix does not mean 23 dB on your
   phone recording of a dinner conversation. Expect a large drop. Our targets in
   [`01-requirements.md`](./01-requirements.md) §2.1 are set against **VVX-Eval (real recordings)**,
   deliberately, and are much lower than published benchmark figures. They are not less ambitious —
   they are measured on harder data.
2. **SI-SDR is a mixture of failure modes.** Distortion, residual noise and interferer leakage all
   feed one number. Two systems with identical SI-SDR can sound entirely different. We therefore
   report **SIR separately** — it is the metric that actually corresponds to "the other speaker is
   almost zero."
3. **Comparisons must be same-data, same-split, same-metric-implementation.** SI-SDR implementations
   differ subtly (zero-mean handling, per-utterance vs global). We pin one implementation
   (`torchmetrics.audio`) and evaluate every baseline through our own harness rather than quoting
   paper numbers. See [`08-evaluation-protocol.md`](./08-evaluation-protocol.md) §2.
