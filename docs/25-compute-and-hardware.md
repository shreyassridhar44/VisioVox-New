# 25 — Compute Strategy & Hardware

Answers two questions: **which machine does what**, and **what actually needs training**.

---

## 1. Available hardware

| | Laptop | College workstation |
|---|---|---|
| RAM | 16 GB | **128 GB** |
| GPU | integrated / entry-level | **NVIDIA RTX A5000, 24 GB GDDR6** |
| Role | Application development | **All ML work** |
| Access | Always | Shared / scheduled |

The A5000 is a genuinely good card for this project — Ampere generation, so native **bf16 tensor
cores** and TF32 matmuls, and 24 GB is double what the original plan assumed. That changes the
training configuration materially (§5).

---

## 1b. Pre-flight checklist for the workstation ⭐

Run this **before** committing to the workstation as the primary machine. Each item has bitten
real projects; items 3, 4 and 8 are the ones that silently waste weeks.

```bash
# 1. OS and kernel — determines the whole setup path
uname -a || systeminfo | head -5

# 2. Admin rights — needed for Docker, CUDA toolkit, driver updates
sudo -v && echo "sudo OK"          # Linux
net session 2>&1 | head -1          # Windows: "Access is denied" = no admin

# 3. GPU present, idle, and yours
nvidia-smi
#    → Confirm: A5000 listed, 24GB, and NO other processes in the compute list.
#      A second user's process on the same card is the #1 cause of OOM at hour 40.

# 4. Is the home directory network-mounted?  ⭐ biggest silent killer
df -h ~ && mount | grep -E "$(df ~ --output=source | tail -1)"
#    → nfs / cifs / smb in the output means DO NOT put datasets here.
#      Network storage bottlenecks the dataloader before the GPU saturates.
#      Find local NVMe instead:  df -h -t ext4 -t xfs -t btrfs

# 5. Free space — need ≥ 510 GB on a LOCAL disk
df -h

# 6. Can it reach what we need?  College firewalls commonly block these.
curl -sI https://huggingface.co            | head -1
curl -sI https://www.openslr.org           | head -1
curl -sI https://github.com                | head -1
#    → Any failure means datasets and model weights cannot be fetched directly.

# 7. Python / CUDA toolchain
python3 --version && nvcc --version 2>/dev/null || echo "no nvcc (fine — PyTorch ships its own)"

# 8. Will long jobs survive?  ⭐
#    Ask the lab admin three questions:
#      a) Is the machine reimaged or wiped on a schedule?
#      b) Are processes killed after N hours, or on logout?
#      c) Is there a job scheduler (Slurm/PBS) I must submit through?
#    Then verify persistence yourself:
which tmux screen || echo "install tmux — runs must survive disconnection"

# 9. Remote access — lets you drive it from the laptop
systemctl is-active ssh sshd 2>/dev/null || echo "no SSH — you must work at the machine"
```

### Interpreting the results

| Finding | Consequence |
|---|---|
| No admin rights | Docker likely unavailable → run services natively or use rootless Podman. Training still works (PyTorch needs no admin). |
| Home is network-mounted | Put `~/data` on local NVMe and symlink. Non-negotiable. |
| GPU shared with other users | Coordinate scheduling; always `nvidia-smi` before launching; keep per-epoch checkpoints |
| Machine gets reimaged | Treat it as ephemeral: code from git, checkpoints synced off after every stage, **VVX backed up elsewhere** |
| HuggingFace blocked | Download weights on the laptop, transfer manually; set `HF_HUB_OFFLINE=1` |
| Slurm/PBS scheduler | Wrap training in a job script; the checkpoint-resume design already handles preemption |
| < 510 GB free | Use the minimum dataset set in §6; cut in the order given there |

---

## 2. Verdict: split the work

**Do not try to do everything on one machine.** They are good at different things.

| Work | Machine | Why |
|---|---|---|
| Frontend, API, workers, infra | **Laptop** | No GPU needed — the mock pipeline returns fixture results |
| Documentation, design, CI setup | **Laptop** | — |
| Dataset acquisition and generation | **College PC** | ~500 GB–1.1 TB; needs the fast local disk |
| **Model training** | **College PC** | Impossible without the A5000 |
| Evaluation, ablations | **College PC** | GPU-bound |
| Real pipeline inference | **College PC** | GPU-bound |
| Final integration testing | **College PC** | Needs real pipeline + app together |

### Can the laptop run the real pipeline at all?

Technically yes, practically no. Rough CPU-only estimates for a **1-minute** clip on 16 GB:

| Stage | CPU time |
|---|---|
| Face detection @ 25 fps | 8–12 min |
| Diarization | 2–3 min |
| Whisper large-v3 (int8) | 3–5 min |
| Extraction | 5–8 min |
| **Total** | **~20–30 min for 60 seconds of video** |

That is fine for a one-off sanity check, useless for iteration. Use `PIPELINE_MODE=mock` on the
laptop.

**This is exactly why the mock pipeline exists** ([`17-infrastructure-deployment.md`](./17-infrastructure-deployment.md) §2).
The entire application — auth, upload, player, speaker switching, captions, landing page — is
buildable and testable on the laptop with zero GPU involvement. The parallel-track plan in
[`21-implementation-plan.md`](./21-implementation-plan.md) was designed for exactly this split
before the hardware was known; it happens to fit perfectly.

### Should the workstation be the *only* machine?

Reasonable, if the pre-flight checks pass cleanly. It depends on one thing:

| Your access | Recommendation |
|---|---|
| **Unrestricted** — admin rights, use it anytime, persistent, SSH available | ⭐ **Make it primary.** Do everything there; keep the laptop as a git-synced client for working elsewhere. One environment is genuinely simpler. |
| **Shared or scheduled** — booked slots, other users, or reimaged periodically | **Keep the split.** Don't spend a GPU time slot on Next.js hot-reload. App work on the laptop, GPU work in your slots. |
| No SSH, must be physically present | **Keep the split**, and lean on the laptop for anything not GPU-bound |

Either way the architecture is unchanged — the mock pipeline still matters, because CI runs without a
GPU and because it is what keeps the application off the ML critical path.

### Working across two machines

```
Laptop  ──git push──►  GitHub  ──git pull──►  College PC
                                                  │
   code only. Datasets, weights and checkpoints    │
   never enter git (see .gitignore).               ▼
                                          ~/data/      (datasets)
                                          ~/models/    (weights, checksummed)
                                          ~/runs/      (checkpoints, logs)
```

Practical notes for a shared workstation:

- Run training inside **`tmux`** or **`screen`** so a disconnect doesn't kill a 40-hour run
- Check for other users before starting: `nvidia-smi` — a second process on the same card will OOM you
- Checkpoint every epoch (already in the training config) so an interruption costs one epoch
- Keep datasets on that machine's **local NVMe**, never a network home directory — network storage
  will bottleneck the dataloader before the GPU is saturated
- Sync checkpoints you care about off the machine; shared machines get wiped

---

## 3. What needs training — the precise answer

**Exactly one model.** Everything else is pretrained or is a deterministic algorithm.

| Stage | Component | Pretrained? | Training needed |
|---|---|---|---|
| S0 | ffmpeg ingest | n/a | ❌ None |
| S1 | **WPE dereverberation** | No weights at all — it is an unsupervised signal-processing algorithm | ❌ None |
| S1 | DeepFilterNet3 denoiser | ✅ MIT | ❌ None |
| S2A | Silero VAD | ✅ MIT | ❌ None |
| S2A | pyannote 3.1 diarization | ✅ (gated HF, free) | ❌ None |
| S2A | ReDimNet speaker embeddings | ✅ | ⚪ Optional fine-tune |
| S2B | SCRFD face detection | ✅ | ❌ None |
| S2B | ByteTrack | Algorithm | ❌ None |
| S2B | LoCoNet / Light-ASD | ✅ | ⚪ Optional fine-tune |
| S3 | Cross-modal fusion | Our algorithm (Hungarian assignment) | ❌ None |
| S4 | **Self-enrolment** | Our algorithm — scoring logic, no weights | ❌ None |
| **S5** | **SEAVE extractor** | ⚠️ **No suitable checkpoint exists** | ✅ **YES — this is the one** |
| S6 | BigVGAN / Vocos restoration | ✅ | ⚪ Optional fine-tune |
| S7 | Whisper large-v3 | ✅ MIT | ❌ None |
| S7 | wav2vec2 forced aligner | ✅ | ❌ None |
| S8 | Leakage audit | Our algorithm | ❌ None |
| S9 | Packaging | n/a | ❌ None |

Two of the five research contributions — **self-enrolment (Novelty 1)** and **cross-stream leakage
repair (Novelty 5)** — are algorithms, not models. They need no training at all. That is worth
knowing: a meaningful part of the novelty is implementable on the laptop.

### Why the extractor can't just be downloaded

Three independent reasons, in increasing order of importance:

1. **No usable pretrained AV-TSE checkpoint exists.** The published audio-visual extraction models
   are tied to LRS2/LRS3 preprocessing pipelines and evaluated on curated frontal-face data. They do
   not transfer to arbitrary uploaded video.

2. **Pretrained audio-only TSE exists but is trained on the wrong distribution.** WeSep and
   SpeakerBeam checkpoints are trained on Libri2Mix-style mixtures: fully overlapped, anechoic, clean,
   level-balanced. Real conversational video is sparsely overlapped, reverberant, noisy and
   unbalanced. Quality collapses.

3. ⭐ **Your primary requirement is a loss-function property.** "The other speaker must be as low as
   possible" is about *interferer suppression*. Every pretrained separation and extraction model is
   trained on SI-SDR, which **cannot distinguish leakage from artifact** — it sums all error into one
   number. A model can score well on SI-SDR while another person is clearly audible underneath.

   You cannot obtain leakage suppression from a checkpoint that was never trained for it. The
   suppression-first objective ([`04-novelty.md`](./04-novelty.md) §4) is the direct mechanism for
   your stated goal, and it only exists if we train.

**But nothing is trained from scratch.** The extractor is initialised from a pretrained separation
checkpoint and adapted. That is fine-tuning with a modified objective, not building a model from
noise.

---

## 4. Staged training plan

Four tiers. Each produces a working system; each is a valid stopping point if time runs out.

### Tier 0 — Zero training (Weeks 2–3)
Assemble everything pretrained, including off-the-shelf blind separation for S5.

| | |
|---|---|
| GPU cost | ~0 (inference only) |
| Expected on real video | SI-SDRi ≈ 6–9 dB · SIR ≈ 10–13 dB |
| Sounds like | Works, but the other speaker is clearly audible |
| Purpose | Working end-to-end system + the honest "before" baseline |

This is Phase 1 of the implementation plan. **It de-risks everything** — you have a complete
pipeline and a demonstrable product in three weeks, before any training happens.

### Tier 1 — Fine-tune audio-only TSE with the suppression objective ⭐
Initialise from a pretrained TSE/separation checkpoint. Add `L_suppress`, `L_consistency`,
`L_silence`. Train on realistic simulation + VVX.

| | |
|---|---|
| GPU cost | **~130 hours** |
| Expected | SI-SDRi ≈ 11–13 dB · **SIR ≈ 17–20 dB** |
| Sounds like | Other speaker mostly gone; noticeable only in dense overlap |
| Purpose | ⭐ **This is the tier that delivers your primary objective** |

**If you only do one training tier, do this one.** It targets leakage directly and gets most of the
perceptual benefit.

### Tier 2 — Add visual conditioning (SEAVE proper)
Visual frontend, reliability-gated fusion, modality dropout.

| | |
|---|---|
| GPU cost | **~170 hours** |
| Expected | SI-SDRi ≈ 13–15 dB · **SIR ≈ 20–24 dB** |
| Fixes | Same-gender pairs, dense overlap, similar voices — the hard cases |
| Purpose | Quality ceiling + Novelties 2 and 3 fully realised |

### Tier 3 — Baselines and ablations (for the report)
| | |
|---|---|
| GPU cost | **~150 hours** |
| Purpose | Makes the results interpretable and the contributions defensible |
| Parallelisable | Perfectly — could be rented for ~$40 if the A5000 is contended |

### Revised budget

| Tier | GPU-hours |
|---|---|
| Tier 0 | ~5 |
| Tier 1 | ~130 |
| Tier 2 | ~170 |
| Tier 3 | ~150 |
| Hyperparameter search | ~40 |
| **Total** | **≈ 495 GPU-hours** |

Down from the ~690 estimated for a 12 GB card. Three reasons: no gradient checkpointing needed
(~35% faster), larger batches (better utilisation), and initialising from pretrained rather than
training from scratch.

**Wall clock:** ~21 days of continuous GPU time. At 8 h/day of exclusive access that is ~9 weeks; with
overnight runs (16 h/day) it is ~4–5 weeks. The plan's Phase 4–7 window (weeks 6–19) accommodates
either.

---

## 5. RTX A5000 configuration

24 GB changes the training config meaningfully from the 12 GB baseline in
[`07-training-playbook.md`](./07-training-playbook.md).

```yaml
# configs/seave_a5000.yaml — overrides for 24GB + 128GB system RAM
data:
  batch_size: 8              # was 4
  grad_accum: 2              # was 4  → same effective batch of 16, half the optimiser steps
  chunk_seconds: 4.0
  num_workers: 12            # was 8  — 128GB RAM affords this comfortably
  prefetch_factor: 6
  persistent_workers: true
  pin_memory: true

model:
  emb_dim: 128               # was 96 — the VRAM headroom buys real capacity
  n_blocks: 6

train:
  precision: bf16            # native on Ampere
  grad_checkpointing: false  # ⭐ was true — ~35% speedup, and we no longer need it
  tf32_matmul: true          # Ampere default; keep it on
  compile: true              # torch.compile, +10–20% after first-run warmup
```

**The 128 GB of system RAM is underrated here.** It lets the OS page cache hold a large fraction of
the dataset, and supports 12+ dataloader workers with deep prefetch. The dataloader bottleneck that
usually caps GPU utilisation at 40–60% on consumer machines largely disappears. Target **> 90% GPU
utilisation**; if you see less, the loader is still the problem.

### VRAM headroom check

| Configuration | Est. peak VRAM |
|---|---|
| Baseline (batch 4, checkpointing on) | ~7 GB |
| A5000 config (batch 8, checkpointing off, emb 128) | ~17 GB |
| Headroom | ~7 GB |

Comfortable. If you want to push further: `chunk_seconds: 6.0` for longer context, or unfreeze the
top blocks of the visual frontend in late Tier 2. Log peak VRAM every epoch either way.

---

## 6. Storage plan

Full corpus is ~1.1 TB ([`06-datasets.md`](./06-datasets.md) §7). On a shared machine that may not be
available. Minimum viable set:

| Dataset | Full | **Minimum** | Notes |
|---|---|---|---|
| LibriSpeech + WHAM! | 90 GB | 90 GB | Required |
| Libri2Mix (16k, min, both) | 150 GB | 150 GB | Required |
| Libri3Mix | 100 GB | — | Skip; simulate 3-speaker mixtures on the fly instead |
| WHAMR! | 80 GB | — | Skip; RIR convolution in the simulator covers reverb |
| VoxCeleb2 | 300 GB | 120 GB | Subset of ~2000 speakers is enough for Tier 2 |
| AVSpeech | 200 GB | — | Optional |
| AMI | 100 GB | 60 GB | Headset + video only; skip far-field array |
| VVX (ours) | 90 GB | 90 GB | ⭐ Never skip — irreplaceable |
| **Total** | **1.1 TB** | **~510 GB** | |

Mixtures are simulated **on the fly** ([`07-training-playbook.md`](./07-training-playbook.md) §3), so
we never store a pre-generated mixture set. That is what keeps this at 510 GB instead of several
terabytes.

If even 510 GB is unavailable, cut in this order: AVSpeech → WHAMR! → Libri3Mix → VoxCeleb2 down to
1000 speakers. **Never cut VVX** — it cannot be re-downloaded.

---

## 7. If the college machine becomes unavailable

| Situation | Response |
|---|---|
| Contended for a few days | Keep working on the app track — it is fully independent |
| Contended for weeks | Rent cloud GPU for the ablation sweep (Tier 3 parallelises perfectly; ~$40 for the full suite on 8× spot instances) |
| Lost entirely | Tier 0 still gives a working demonstrable product; Tier 1 on rented GPU is ~$60–90 |
| Data lost on the shared machine | Everything except VVX is re-downloadable. **VVX must be backed up off that machine, encrypted.** (R-26) |

Checkpoints go to object storage after every training stage, not just at the end. A shared machine
being wiped should cost you a resume, not a restart.

---

## 8. Recommended sequence

```
NOW ──────────────────────────────────────────────────────────────► 
 Laptop:   scaffold repo · app skeleton · mock pipeline · player · landing page
 College:  (idle at first) → datasets → Tier 0 baseline → Tier 1 → Tier 2 → ablations
                                          ▲
                                          └── start college work in Week 2,
                                              in parallel with app development
```

Concretely, for the next two weeks:

1. **Laptop** — Phase 0 scaffolding and Phase 2 app skeleton. No GPU needed.
2. **College PC** — Phase 0 environment verification (`nvidia-smi`, PyTorch CUDA check), then start
   the LibriMix download early: it is slow and unattended.
3. **Both** — Phase 1 baseline runs on the college PC and produces the frozen artifact manifest that
   the laptop's app work builds against.

The single most important early action on the college machine is confirming
`torch.cuda.is_available()` and running the Phase 0 model smoke test. Everything downstream assumes
it.
