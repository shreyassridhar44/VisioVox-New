# 24 — Glossary

---

## Core concepts

**Blind Source Separation (BSS)** — Separating a mixture into all its constituent sources without
knowing who they are. Output order is arbitrary (see *Permutation problem*). Rejected as this
project's primary architecture; retained as an evaluation baseline.

**Target Speaker Extraction (TSE)** — Extracting *one specific* speaker from a mixture, given a cue
identifying them. Output identity is bound to the cue, so there is no permutation problem. **The
primary architecture here.**

**Permutation problem** — In BSS, the model's output channels have no fixed relationship to
speakers, and that relationship changes between inference windows. Stitching windows into a
full-length track requires solving an assignment problem where errors are catastrophic (wrong voice)
rather than gradual. See [`03-research-landscape.md`](./03-research-landscape.md) §1.1.

**PIT (Permutation-Invariant Training)** — Training BSS by computing loss under all output orderings
and back-propagating the best. Excellent at separating; indifferent to identity. The root cause of
the permutation problem.

**Diarization** — "Who spoke when." Produces time-labelled speaker turns without separating audio.
Here it also supplies the single-talker regions used for self-enrolment and passthrough routing.

**Active Speaker Detection (ASD)** — Determining which visible face is speaking at each moment.
Binds voices to faces.

**Enrolment** — The cue identifying the target speaker to a TSE model. Conventionally a clean
recording supplied externally. **Self-enrolment** (this project's Novelty 1) mines it automatically
from the video itself.

**Modality-adaptive conditioning** — Continuously weighting audio and visual cues by their measured
reliability, per frame, so the model degrades gracefully when the face is occluded or the voice cue
is contaminated. Novelty 2.

**Leakage** — Another speaker's voice audible in a target speaker's isolated track. The failure mode
this product must not have. Not directly measured by SI-SDR; measured by SIR.

**Single-talker passthrough** — Routing regions where only one person speaks around the separation
model, since separating already-clean audio degrades it. Improves quality *and* cuts cost.

**Faithful vs Natural** — Two delivered variants per speaker. *Faithful* is the discriminative
model's output: accurate, possibly artifact-y, the default. *Natural* has generative restoration
applied: cleaner, with a small hallucination risk, explicitly labelled.

---

## Metrics

**SI-SDR / SI-SNR** — Scale-Invariant Signal-to-Distortion (or Noise) Ratio, in dB. Measures how
close an estimate is to the reference, ignoring overall gain. The two terms are used
interchangeably in this literature (Le Roux et al., 2019); this project standardises on **SI-SDR**.

**SI-SDRi** — SI-SDR *improvement*: the estimate's SI-SDR minus the unprocessed mixture's. The
standard comparable figure. Isolates the model's contribution from how easy the input was.

**SIR (Signal-to-Interference Ratio)** — Target energy over interfering-speaker energy. ⭐ **The
metric that corresponds to "the other speaker is almost inaudible."** Co-primary with SI-SDRi here,
because SI-SDR cannot distinguish leakage from other error.

**SAR** — Signal-to-Artifact Ratio. Isolates processing artifacts from leakage.

**Silence-region leakage** — Residual energy where the target speaker is silent. Matters
disproportionately because speakers are silent 50–70% of the time and nothing masks the leakage
there.

**Cross-stream leakage word rate** — Metric defined by this project: the fraction of words in one
speaker's transcript that were actually spoken by someone else. Measures misattribution directly.

**WER (Word Error Rate)** — Transcription errors as a fraction of reference words.
**Target-speaker WER** measures it against only the target's reference.

**DER (Diarization Error Rate)** — Missed speech + false alarm + speaker confusion, over total
speech time.

**PESQ / eSTOI** — Intrusive (reference-requiring) perceptual quality and intelligibility measures.

**DNSMOS P.835 / UTMOS** — Non-intrusive learned MOS predictors. Work without a reference, so they
can score real user output at inference time. DNSMOS P.835 gives SIG (speech), BAK (background) and
OVRL (overall).

**ECE (Expected Calibration Error)** — How well stated confidence matches observed accuracy. A
confidence score with poor calibration is worse than no score.

**RTF (Real-Time Factor)** — Processing time ÷ media duration. RTF 0.5 means a 10-minute video takes
5 minutes.

**LUFS / dBTP** — Loudness Units relative to Full Scale (integrated perceived loudness, EBU R128)
and decibels True Peak. Tracks are normalised to −16 LUFS / −1 dBTP so switching speakers doesn't
jump in volume.

---

## Models & techniques

**TF-GridNet** — Time-frequency separation architecture combining intra-frame, inter-frame and
full-band modelling. Strong on reverberant data. The SEAVE backbone.

**SepFormer / Conv-TasNet / DPRNN / MossFormer2** — Blind separation architectures. SepFormer and
Conv-TasNet serve as baselines here.

**SpeakerBeam / SpEx+ / VoiceFilter / WeSep** — Audio-only TSE systems.

**AV-SepFormer / VisualVoice / MuSE / IIANet** — Audio-visual separation and extraction systems.

**pyannote.audio** — Diarization toolkit; overlap-aware segmentation plus embedding clustering.

**ECAPA-TDNN / ReDimNet** — Speaker embedding (voiceprint) models. ReDimNet is the current
top-tier choice.

**LoCoNet / TalkNet-ASD / Light-ASD** — Active speaker detection models.

**SCRFD / RetinaFace / ByteTrack** — Face detection and multi-object tracking.

**AV-HuBERT** — Self-supervised audio-visual speech representation model; source of strong lip
features.

**Whisper / faster-whisper / WhisperX** — ASR, its CTranslate2 acceleration, and word-level forced
alignment. Whisper's tolerance of separation artifacts is why it is used here.

**WPE** — Weighted Prediction Error dereverberation. Unsupervised, so no training data or domain
risk.

**DeepFilterNet** — Real-time-class speech denoiser.

**BigVGAN / Vocos** — Neural vocoders used for generative restoration.

**FiLM (Feature-wise Linear Modulation)** — Conditioning mechanism that scales and shifts a network's
intermediate features based on a conditioning vector. How speaker and visual cues are injected into
the separator.

**Modality dropout** — Training-time random removal or corruption of a modality, so the model learns
to detect and compensate for unreliable inputs rather than assuming they are always present.

---

## Datasets

**LibriMix / Libri2Mix / Libri3Mix** — Synthetic speech mixtures from LibriSpeech + WHAM! noise.
Standard benchmark. Fully overlapped, anechoic — hence the domain-gap problem.

**WSJ0-2mix** — The classic separation benchmark. Largely saturated.

**WHAMR!** — Noisy and reverberant variant.

**VoxCeleb2** — Large-scale audio-visual speaker dataset from YouTube interviews.

**LRS2 / LRS3** — High-quality audio-visual speech corpora. ⚠️ Research-only licensing.

**AVSpeech** — Large-scale AV corpus of clean single-speaker segments.

**AMI Meeting Corpus** — Real meetings with **per-speaker headset microphones**, giving near-ground-
truth references for genuinely overlapping speech. Under-used and valuable here.

**AVA-ActiveSpeaker** — ASD benchmark.

**AMI-Train / AMI-Val / AMI-Eval** — Splits of the AMI Meeting Corpus, using each participant's
own headset microphone as a per-speaker reference and their Closeup camera as the face track.
**All headline results are reported on AMI-Eval**, and are meeting-domain results rather than
general ones. Replaced the planned self-recorded VVX corpus; see ADR-0015.

---

## System terms

**Artifact manifest** — The versioned JSON contract describing a completed job's outputs. The
boundary between the ML pipeline and the application; frozen early so both tracks can progress
independently.

**Speaker registry** — Canonical per-project speaker list: IDs, labels, colours, thumbnails,
modality, confidence.

**Mock pipeline** — A worker that returns fixture manifests with realistic timing, allowing the
entire application to be built and tested without a GPU.

**Correlation ID** — Identifier minted in the browser and carried through every log, span and error
for one user action. The primary debugging handle.

**Trust score** — Per-segment, per-speaker calibrated confidence produced by the leakage audit
(Novelty 5). Surfaced in the UI so users know where to be sceptical.

**Contested span** — A region where leakage attribution was ambiguous. Kept in *both* transcripts
and marked, rather than deleted — deleting a real utterance is worse than showing it twice.

**Equal-power crossfade** — Fading between two audio sources using cos/sin curves rather than linear
ones, so total power stays constant. A linear crossfade dips ~3 dB at the midpoint, which is audible.

**HLS / CMAF / EXT-X-MEDIA** — Streaming format, its fragmented-MP4 container, and the playlist tag
that declares alternative audio renditions — the mechanism for switchable per-speaker tracks in long
content.

**Scale-to-zero** — Running zero GPU workers when the queue is empty. The primary cost control.

**Expand → migrate → contract** — Schema-change discipline where old and new code can both run
against the database at all times, making deploys and rollbacks safe.

---

## Abbreviations

| | |
|---|---|
| ADR | Architecture Decision Record |
| ASR | Automatic Speech Recognition |
| ASD | Active Speaker Detection |
| ASVS | Application Security Verification Standard (OWASP) |
| BFF | Backend For Frontend |
| BIPA | Biometric Information Privacy Act (Illinois) |
| DPIA | Data Protection Impact Assessment |
| IDOR | Insecure Direct Object Reference |
| MSE | Media Source Extensions |
| RED / USE | Rate-Errors-Duration / Utilisation-Saturation-Errors metric patterns |
| RIR | Room Impulse Response |
| ROI | Region of Interest (here: mouth crops) |
| RPO / RTO | Recovery Point / Time Objective |
| SBOM | Software Bill of Materials |
| SLO | Service Level Objective |
| SSE | Server-Sent Events |
| VAD | Voice Activity Detection |
