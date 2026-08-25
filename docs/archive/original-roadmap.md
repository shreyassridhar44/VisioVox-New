# Speaker-Isolated Audio & Captioning — Project Roadmap

## 0. Note on scope vs. earlier plan

An earlier version of this plan targeted **speaker-labeled captions only** (a "cocktail party" caption generator that stays clean during overlapping speech). Your actual end goal is broader and more interactive:

> Upload a video with 2–3 overlapping speakers → system detects speaker count (voice + face) → user plays the video and **selects a speaker** → **that speaker's isolated audio plays** in place of the mixed audio → **that speaker's captions** are shown, synced.

This is a superset of the earlier plan. It needs everything from before (diarization, separation, ASR) **plus**:
- Persisting **separated per-speaker audio tracks** (not just using separation internally to clean up transcripts)
- A **frontend audio-track switcher** tied to speaker selection
- Per-speaker caption tracks that swap when the user picks a different speaker

This document supersedes the earlier one and reflects the real target.

---

## 1. Project statement

**System**: A web app. User uploads a video with multiple simultaneous speakers and background noise. The system:
1. Detects number of speakers (audio + face cues)
2. Separates the mixed audio into per-speaker isolated tracks
3. Transcribes each speaker's track into speaker-specific captions
4. Lets the user click/select a speaker (e.g., a face box, or a name/label list) during playback
5. Swaps the played audio to that speaker's isolated track and shows that speaker's captions, in sync with the video

**Core research contribution**: fine-tuning an open-source **speech separation** model (and optionally an **active speaker detection** model) on top of pretrained checkpoints, so isolation quality is good enough for this to actually sound listenable — not just transcribable.

---

## 2. Feasibility report

### 2.1 Is this feasible for a capstone/individual project? **Yes, with scope discipline.**

| Sub-problem | Feasibility | Why |
|---|---|---|
| Speaker diarization ("who spoke when") | High — solved problem | pyannote.audio pretrained models are strong out of the box |
| Speaker counting from audio | High | Byproduct of diarization |
| Face detection / face counting | High | Mature, fast models (RetinaFace, MediaPipe) |
| Active speaker detection (which face is talking) | Medium | Works well on frontal, reasonably-lit video (e.g. Zoom-call style); degrades on side angles, poor lighting, off-screen speakers |
| **Speech separation of 2 overlapping speakers** | Medium-High | Well-studied (SepFormer, Conv-TasNet on Libri2Mix). Fine-tuning pretrained checkpoints is realistic on a single GPU in weeks, not months |
| **Speech separation of 3–4 overlapping speakers** | Medium-Low | Quality drops sharply as speaker count increases. This is genuinely research-grade difficulty, not a solved productized problem |
| Isolated-audio quality being pleasant to listen to (not just "transcribable") | Medium | Separation models are optimized for SI-SNR / WER, not perceptual audio quality. Expect some robotic/watery artifacts, especially at 3+ speakers |
| ASR per isolated stream | High | Whisper is robust even on imperfect separated audio |
| Full interactive web app (upload → process → play → switch speaker) | High | Standard full-stack engineering; the hard part is the ML, not the UI |

### 2.2 The honest risk

The **weakest link is 3–4 speaker separation quality**, not the web app or ASR. Plan your demo and grading expectations around **2–3 speakers with real but not extreme overlap** (a few seconds of cross-talk, not a chaotic group argument for the whole clip). Treat 4-speaker separation as a stretch goal you report on honestly rather than promise as a guaranteed deliverable.

### 2.3 Hardware feasibility (assumed: single workstation, NVIDIA GPU, WSL2)

Fine-tuning SepFormer/Conv-TasNet-scale models on Libri2Mix/Libri3Mix ("min" 16kHz subsets) is realistic on a single consumer/prosumer GPU with 12GB+ VRAM in the timeframe of a multi-week project, provided you don't try to train on the *full* LibriMix at 8kHz+16kHz+both mix types simultaneously — pick one config and stick to it. Whisper and pyannote are used pretrained/frozen (no fine-tuning needed unless your domain is very unusual), which keeps compute demand down.

---

## 3. Why WSL2 (not native Windows, not bare Linux)

If your machine is Windows with an NVIDIA GPU, WSL2 is the right interface because:
- **CUDA passthrough works properly in WSL2** (via NVIDIA's WSL driver) — you get near-native GPU performance for PyTorch, unlike trying to juggle Windows-native CUDA toolchains which are more fragile for ML tooling.
- The ML ecosystem (SpeechBrain, Asteroid, pyannote.audio, ffmpeg builds, Linux-first pip packages) behaves the way its documentation assumes — most tutorials and error-fixes online assume Linux.
- You keep Windows for daily use (browser, video review, note-taking) while running the actual training/inference in a Linux environment.

**Setup outline:**
1. Install/update the NVIDIA GPU driver **on Windows** (not inside WSL — WSL2 uses the Windows host driver via passthrough).
2. `wsl --install -d Ubuntu-22.04` (or your preferred Ubuntu LTS) from an elevated PowerShell.
3. Inside WSL2 Ubuntu: verify GPU visibility with `nvidia-smi` — if this works, passthrough is correctly configured.
4. Install Miniconda or `venv` inside WSL2, then install PyTorch with CUDA support matching your driver (check `nvidia-smi` for the max supported CUDA version, install a PyTorch build ≤ that version).
5. Do **all** ML work (dataset storage, training, inference, the Python backend) inside the WSL2 filesystem (e.g. `~/projects/...`), not on the Windows `/mnt/c/...` mount — I/O across the WSL/Windows boundary is noticeably slower and will bottleneck data loading during training.
6. Use VS Code with the "WSL" extension to edit code that lives inside WSL2 from a normal-feeling Windows editor window.

---

## 4. System architecture

```
[Upload video]
       │
       ▼
[1. Ingest] ── ffmpeg: extract audio (16kHz mono), keep video separately
       │
       ▼
[2. Speaker count estimation]
   ├─ Audio path: speaker embeddings (ECAPA-TDNN) + clustering
   └─ Video path: face detection (RetinaFace/MediaPipe) → face tracks
       │
       ▼
[3. Diarization] ── pyannote.audio: "who spoke when", flags overlapping regions
       │
       ▼
[4. Active speaker detection] ── TalkNet / Light-ASD: maps face tracks → voice segments
       │  (used to build the "click a face to select a speaker" UI feature)
       ▼
[5. Source separation] ── SepFormer / Conv-TasNet (fine-tuned by you)
       │  Runs on overlapping segments (and optionally the whole track) to
       │  produce one isolated audio stream PER detected speaker, full-length,
       │  time-aligned with the original video.
       ▼
[6. Per-speaker ASR] ── Whisper, run once per isolated audio stream
       │  Produces one caption track (SRT/VTT-like JSON) PER speaker.
       ▼
[7. Storage/packaging]
   Output per video:
     - original video (unchanged)
     - N isolated audio tracks (one per speaker), time-aligned to the video
     - N caption tracks (one per speaker), time-aligned to the video
     - speaker metadata (id, representative face thumbnail if available, label)
       ▼
[8. Frontend player]
   - Video plays with the ORIGINAL mixed audio by default (or muted, your choice)
   - Speaker selector UI (face thumbnails or "Speaker 1/2/3" buttons)
   - On selection: swap the active <audio> track to that speaker's isolated
     track, keep it time-synced to the <video> element (mute video's own
     audio, play the isolated track instead), and swap the caption overlay
     to that speaker's caption track
   - Switching speaker mid-playback re-syncs to the current timestamp
```

### 4.1 Key engineering detail: audio/video sync on speaker switch

This is the part that's easy to get subtly wrong. Implementation approach:
- Keep the `<video>` element muted at all times; it's the visual/timing clock.
- Maintain N `<audio>` elements (or one audio element whose `src` you swap), one per speaker, all pre-generated to the **same length as the original video** (silence-padded where that speaker isn't talking).
- On speaker selection, set the chosen `<audio>` element's `currentTime = video.currentTime` and `play()`, pause/mute the others.
- On every `video.play()`/`seek` event, re-sync the active audio's `currentTime` to the video's.
- Captions: store each speaker's captions as timestamped JSON (not just SRT files) so the frontend can filter/display "current caption for currently selected speaker" reactively as playback advances, rather than parsing SRT client-side.

---

## 5. Datasets

| Dataset | Use | Notes |
|---|---|---|
| Libri2Mix (min, 16kHz) | Fine-tune + benchmark 2-speaker separation | Standard benchmark, comparable to published numbers |
| Libri3Mix (min, 16kHz) | 3-speaker separation (stretch) | Needed only if you attempt 3+ speaker separation |
| Your own recordings | Realistic demo + qualitative eval | Strongly recommended: record yourself + friends talking over each other on camera. Makes the demo and the "before/after fine-tuning" story much more convincing than benchmark numbers alone, and is the only way to test the face/active-speaker-detection path realistically |
| VoxCeleb (optional) | Only if pretrained diarization embeddings underperform on your data | Large — avoid unless needed |

Storage discipline: stick to "min" versions, one sample rate (16kHz), avoid regenerating LibriMix in multiple configs. Budget dataset storage before downloading — LibriMix generation scripts can silently produce far more than you need if you generate all splits/modes.

---

## 6. Phased roadmap

### Phase 0 — Environment setup (Week 1)
- WSL2 + Ubuntu + NVIDIA driver passthrough verified (`nvidia-smi` inside WSL2)
- Python env (conda/venv) with PyTorch (CUDA build matching driver), SpeechBrain, Asteroid, pyannote.audio, openai-whisper, ffmpeg, face detection lib (MediaPipe or RetinaFace)
- Smoke test: run each pretrained model on one sample clip end-to-end (no fine-tuning) to confirm plumbing works

### Phase 1 — Baseline pipeline, pretrained only (Weeks 2–3)
- Wire stages 1–7 with off-the-shelf pretrained weights, no fine-tuning yet
- Test on: (a) a clean turn-taking video, (b) a video with real overlapping speech
- Document exactly where it breaks — garbled isolated audio, wrong speaker count, misattributed captions during overlap. This is your "before" baseline for the final report

### Phase 2 — Dataset preparation (Weeks 3–4)
- Generate Libri2Mix (min, 16kHz); Libri3Mix if attempting the stretch goal
- Record your own overlapping-speech test clips (with video, so face detection has real data to work with) — do this early so Phase 4/6 aren't scrambling for eval data
- Proper train/val/test split; no leakage between fine-tuning data and your qualitative eval clips

### Phase 3 — Fine-tune the separation model (Weeks 4–6)
- Start from a pretrained SepFormer or Conv-TasNet 2-speaker checkpoint
- Fine-tune on Libri2Mix first — validates the training loop and gives a literature-comparable number
- Track **SI-SNRi** before/after fine-tuning — this is your primary quantitative result
- If time allows, extend to Libri3Mix for a 3-speaker model — report this as a stretch result with an honest quality drop-off documented

### Phase 4 — Active speaker detection (Weeks 5–6, can overlap Phase 3)
- Integrate TalkNet or Light-ASD (pretrained) to map detected faces to active speech segments
- Test on your own recorded clips — this is the path most likely to need real-world debugging (lighting, angle, partial faces)
- Explicitly build the audio-only fallback: if no usable video/faces, speaker selection UI falls back to "Speaker 1 / Speaker 2" labels instead of face thumbnails

### Phase 5 — Full-length isolated audio track generation (Week 7)
- Extend separation output from "just the overlapping segments" to a **full-length, silence-padded, per-speaker audio track** matching the video's duration (needed for the play/select/swap feature — see §4.1)
- Re-run Phase 1's test videos through the fine-tuned pipeline; compare isolated audio quality and WER against baseline

### Phase 6 — Web application (Weeks 8–10)
- **Backend**: FastAPI wrapping the pipeline; upload endpoint → background job (Celery or simple async task queue, since processing is slow) → polling/status endpoint → result endpoint serving video + N audio tracks + N caption JSONs
- **Frontend**: upload UI, video player with muted video + speaker-switchable audio (per §4.1), speaker selector (face thumbnails if available, else labeled buttons), live caption overlay for the selected speaker
- Single-user, local-first is fine for a project of this scope — don't over-invest in multi-user infra, auth, or cloud deployment unless explicitly required

### Phase 7 — Evaluation, writeup, polish (Weeks 10–12)
- Held-out test set metrics: **SI-SNRi** (separation), **WER** (transcription accuracy per isolated stream), **DER** (diarization error rate)
- Ablation: with vs. without fine-tuning; 2- vs 3-speaker degradation curve
- Demo reel: your own recorded overlapping-speech video, showing speaker switch → audio/caption swap live
- Written report: honest limitations section (this matters as much as the demo for grading/credibility)

---

## 7. Evaluation metrics

| Metric | Measures | Used for |
|---|---|---|
| SI-SNRi (dB) | Separation quality improvement over the mixed input | Core quantitative result for the separation model |
| WER (Word Error Rate) | Transcription accuracy | Per-speaker caption quality, before/after separation |
| DER (Diarization Error Rate) | Speaker turn/count accuracy | Diarization stage quality |
| MOS-style listening test (optional, informal) | Perceptual quality of isolated audio | Worth doing informally — SI-SNRi doesn't always correlate with "sounds pleasant to a human" |
| Face-to-voice mapping accuracy (qualitative) | Whether the selected face actually corresponds to the audio that plays | Active speaker detection sanity check |

---

## 8. Tech stack summary

- **Interface/OS**: WSL2 (Ubuntu LTS) on Windows, for proper CUDA passthrough
- **ML/audio**: PyTorch, SpeechBrain (SepFormer recipes), Asteroid (Conv-TasNet/DPRNN), pyannote.audio (diarization), OpenAI Whisper (ASR)
- **Vision**: MediaPipe or RetinaFace (face detection), TalkNet or Light-ASD (active speaker detection)
- **Backend**: FastAPI, background task queue (Celery or FastAPI `BackgroundTasks`/async workers for a simpler setup)
- **Frontend**: React (or plain HTML/JS for speed) — custom video/audio-sync player, speaker selector, caption overlay
- **Storage**: local filesystem (project scope doesn't need cloud storage)
- **Tooling**: ffmpeg for all audio/video extraction and muxing

---

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| 3–4 speaker separation quality is poor | Report as a documented degradation curve; cap the live demo at 2–3 speakers with moderate overlap |
| Active speaker detection unreliable on your test videos (lighting/angle) | Treat as an enhancement layer, not a dependency — fall back to audio-only diarization and generic "Speaker 1/2/3" selection labels |
| Isolated audio sounds artifact-y even when SI-SNRi looks decent | Be explicit in your report that separation is optimized for a numeric metric, not perceptual quality; mention this as a known limitation and possible future work (e.g. adding a perceptual/PESQ-aware loss term) |
| Fine-tuning shows little improvement over pretrained on Libri2Mix | Fine-tune on a domain-shifted set instead (your own recordings, noisier conditions) — more likely to show a real, defensible improvement than same-distribution fine-tuning |
| Audio/video sync drifts on speaker switch | Re-sync `currentTime` on every switch and periodically during playback (see §4.1); test explicitly with seeking/scrubbing, not just play-through |
| Running out of time for full web app polish | Prioritize pipeline + evaluation for grading; a functional-but-plain UI that correctly demonstrates the speaker-switch feature is enough |

---

## 10. Things you could add (beyond the core goal)

- **Background noise suppression** as a pre-processing step (e.g. a pretrained speech enhancement model) before separation — directly addresses your mention of "disturbance in the background" and is a relatively cheap addition since good pretrained denoisers exist
- **Speaker naming/identification** across multiple uploaded videos (voice-print matching) — "this is the same Speaker 2 as in last week's video" — a nice extension once basic separation works, using the same ECAPA-TDNN embeddings you already compute for diarization
- **Confidence indicator** shown in the UI when separation quality for a segment is low, so users know when to trust the isolated audio less
- **Downloadable outputs**: let the user download an isolated-audio-only file or a captioned clip for a single speaker
- **Live highlight of the speaking face** on the video overlay, synced to the active caption, using the active speaker detection output — a strong visual demo feature with a small marginal cost given Phase 4 already produces this mapping

---

## 11. Definition of done

1. End-to-end pipeline: upload video → speaker count + isolated per-speaker audio + per-speaker captions, all time-aligned to the source video
2. Working frontend feature: selecting a speaker during playback swaps the played audio and captions to that speaker, correctly synced, including after seeking
3. A fine-tuned separation model with a documented, honest before/after comparison (SI-SNRi, WER)
4. At least one own-recorded demo video showing real overlapping speech handled end-to-end, including the speaker-switch interaction
5. A written report covering methodology, quantitative results, an honest limitations section (especially around 3–4 speaker separation and perceptual audio quality), and possible future work
