# 07 — Training Playbook

Operational guide for training SEAVE. Target hardware: **NVIDIA RTX A5000 24 GB + 128 GB RAM**
([`25-compute-and-hardware.md`](./25-compute-and-hardware.md)). Environment setup in
[`23-runbook.md`](./23-runbook.md) §2.

> **Only one model is trained in this project** — the extractor (S5). Every other stage uses a
> pretrained checkpoint or a deterministic algorithm. And the extractor itself is **fine-tuned from a
> pretrained separation checkpoint**, never trained from scratch. See
> [`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §3.

---

## 1. Curriculum

Five stages. Each has an exit gate; do not advance until it is met. Skipping a gate wastes far more
time than the gate costs.

| Stage | Data | Init | Epochs | Exit gate |
|---|---|---|---|---|
| **C0 Smoke** | 100 samples | random | 200 | Overfits to < 0.1 loss. *Proves the loop, nothing else.* |
| **C1 Audio-only TSE** | Libri2Mix | ⭐ **pretrained separation ckpt** | ~60 | ≥ 13 dB SI-SDRi on Libri2Mix test |
| **C2 Add visual** | VoxCeleb2-Mix | C1 weights | ~80 | ≥ +1.5 dB over C1 on same-gender pairs |
| **C3 Realistic sim** | Realistic simulation (§5 of datasets) | C2 | ~60 | ≥ 10 dB SI-SDRi on realistic sim |
| **C4 In-domain** | AMI-Train | C3 | ~30 | Meets NFR-ML targets on AMI-Val |

**C1 initialises from a pretrained checkpoint**, not from random weights. A SepFormer or Conv-TasNet
encoder/separator trained on Libri2Mix already knows how to separate; what it does not know is how to
be *conditioned* and how to *suppress* rather than merely reconstruct. Adapting it is roughly 60
epochs where training from scratch is ~150 — the single largest saving in the compute budget.

**C0 is not optional.** A model that cannot overfit 100 samples has a bug — in the loss, the data
loader, the masking, or the gradient flow. Finding that after 40 hours of C1 training is the single
most common and most expensive mistake in this kind of project.

### Why this order
- Audio-only first: fewer moving parts, faster iteration, isolates separator bugs from fusion bugs.
- Visual second, initialised from C1: the visual pathway learns to *add* to a working audio model
  rather than the two co-adapting from noise (which converges slowly and unstably).
- Realism third: the model needs separation competence before it can learn robustness.
- In-domain last, briefly: 30 epochs on 18 sessions overfits fast. Low LR, early stopping, heavy
  augmentation.

---

## 2. Configuration

```yaml
# configs/seave_base.yaml
model:
  backbone: tfgridnet
  init_from: pretrained/sepformer-libri2mix   # ⭐ not random
  n_blocks: 6
  emb_dim: 128                                # 24GB headroom buys real capacity
  lstm_hidden: 192
  attn_heads: 4
  n_fft: 512
  hop: 128
  cond:
    speaker_emb_dim: 192
    visual_dim: 512
    fusion: film_crossattn
    reliability_gate: true          # Novelty 2
  confidence_head: true

data:
  sample_rate: 16000
  chunk_seconds: 4.0
  batch_size: 8                      # A5000 24GB
  grad_accum: 2                      # effective batch 16, half the optimiser steps
  num_workers: 12                    # 128GB system RAM affords this
  prefetch_factor: 6
  pin_memory: true
  persistent_workers: true

loss:
  si_sdr: 1.0
  suppression: 0.3
  suppression_tau: -10.0
  consistency: 0.2
  mrstft: 0.5
  silence: 0.2

optim:
  name: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-2
  betas: [0.9, 0.98]
  scheduler: cosine
  warmup_steps: 4000
  grad_clip: 5.0

train:
  precision: bf16                    # native on Ampere
  grad_checkpointing: false          # ⭐ 24GB — not needed; ~35% faster without it
  tf32_matmul: true                  # Ampere default; keep on
  compile: true                      # torch.compile, +10–20% after warmup
  ema_decay: 0.999
  seed: 1337
  deterministic: false               # cudnn benchmark on; log the tradeoff
  max_epochs: 150
  early_stop_patience: 15
  val_every_epoch: 1

augment:
  modality_dropout: { visual: 0.2, audio_cue: 0.15, both_min_one: true }
  visual_corrupt:   { occlusion: 0.15, blur: 0.2, framedrop: 0.1, profile: 0.25 }
  audio_aug:        { speed: [0.95, 1.05], codec: 0.3, level_jitter_db: 6 }
```

### VRAM budget (24 GB target)

| Configuration | Est. peak VRAM |
|---|---|
| A5000 config above (batch 8, no checkpointing, emb 128) | ~17 GB |
| Headroom | ~7 GB |

Comfortable. Spend the headroom, in this order, if you want more quality:
1. `chunk_seconds: 6.0` — longer context, usually the best return
2. Unfreeze the top blocks of the visual frontend in late C2
3. `n_blocks: 8`

**If you hit OOM anyway** (a shared machine with another user's process is the usual cause), apply
in order: `grad_checkpointing: true` → `batch_size: 4, grad_accum: 4` → `chunk_seconds: 3.0` →
`emb_dim: 96`.

⚠️ **On a shared workstation, always `nvidia-smi` before starting.** A second process on the same
card is the most common cause of a run dying at epoch 40. Log peak VRAM every epoch either way.

---

## 3. Data loading

The dataloader is the most likely bottleneck, not the GPU.

```python
class MixtureDataset(Dataset):
    """Simulates mixtures on the fly. Never pre-generate the full mixture set —
    on-the-fly simulation gives effectively infinite augmentation diversity
    and avoids hundreds of GB of disk."""

    def __getitem__(self, idx):
        # deterministic per-(epoch, idx) RNG so runs are reproducible AND
        # each epoch sees different mixtures
        rng = np.random.default_rng(hash((self.epoch, idx)) % 2**32)

        sources = self.sample_sources(rng, n=rng.integers(2, 4))
        mix, refs, vids = simulate(sources, rng)      # docs/06 §5

        target_idx = rng.integers(len(sources))
        enrol = self.mine_or_load_enrolment(sources[target_idx], rng)

        return dict(mixture=mix, target=refs[target_idx],
                    interferers=[r for i, r in enumerate(refs) if i != target_idx],
                    enrolment=enrol, visual=vids[target_idx],
                    silence_mask=(np.abs(refs[target_idx]) < 1e-4))
```

Checklist — verify all of these before a long run:

- [ ] `num_workers` ≥ 8, `persistent_workers=True`, `prefetch_factor=4`
- [ ] Data on WSL2 local NVMe, **not** `/mnt/c`
- [ ] RIRs pre-generated and memory-mapped (generating RIRs per sample is far too slow)
- [ ] Mouth ROIs pre-extracted to a packed format (LMDB/webdataset), not decoded from video per sample
- [ ] GPU utilisation ≥ 85% during training — if it is 40%, the bottleneck is the loader, not the model
- [ ] Enrolments cached per source utterance

---

## 4. Training loop essentials

```python
for batch in loader:
    with autocast(dtype=torch.bfloat16):
        out  = model(batch.mixture, batch.enrolment, batch.visual)
        loss = criterion(out, batch)

    loss.backward()                     # no GradScaler needed with bf16
    if step % grad_accum == 0:
        clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        ema.update(model)

    if step % 100 == 0:
        log(loss=..., si_sdr=..., sir=..., lr=..., vram=torch.cuda.max_memory_allocated(),
            grad_norm=..., alpha_mean=out.alpha.mean(), beta_mean=out.beta.mean())
```

Logging `alpha_mean` / `beta_mean` (the reliability gates) is how you verify Novelty 2 is actually
learning. If `beta` sits at a constant regardless of visual corruption, the gate is not working and
the contribution is not real — catch that in week 1 of C2, not at evaluation.

**Checkpointing:** every epoch, keep best-3 by val SI-SDR + last. Save model, EMA, optimiser,
scheduler, RNG state, epoch — so a crashed run resumes exactly rather than restarting.

---

## 5. Monitoring

| Signal | Healthy | Action if not |
|---|---|---|
| Train loss | monotone down, small noise | spikes → lower LR or raise grad clip |
| Val SI-SDR | rising, plateau late | falling while train falls → overfitting; more augmentation |
| Train–val gap | < 2 dB | > 3 dB → overfitting |
| Grad norm | 0.5–5 | > 50 → instability; check for NaN in the loss terms |
| `beta_mean` under clean video | > 0.7 | low → visual pathway is being ignored |
| `beta_mean` under corrupted video | < 0.4 | high → gate isn't learning; check dropout is applied |
| GPU util | > 85% | low → dataloader bound |
| VRAM | stable | growing → leak, usually a retained graph in logging |

Track with Weights & Biases (or TensorBoard + MLflow if offline is required). Log **audio samples**
every 5 epochs — spectrograms plus listenable clips. Numbers hide artifacts that are obvious in two
seconds of listening.

---

## 6. Hyperparameter search

Do not grid search — the compute budget doesn't allow it. Search in this order, one variable at a
time, on a **10% subset** with short runs:

1. **Learning rate** (highest impact): `{3e-4, 5e-4, 1e-3, 2e-3}`
2. **Loss weights** (Novelty 3 depends on this): `suppression ∈ {0.1, 0.3, 0.6}`,
   `consistency ∈ {0.1, 0.2, 0.4}` — evaluate on **SIR**, not SI-SDR, or the search optimises the
   wrong thing
3. **Suppression hinge** `tau ∈ {-5, -10, -15} dB` — controls how hard suppression is pushed before
   it stops
4. **Chunk length** `{2, 4, 6} s` — trades context against batch size
5. **Model size** — only if 1–4 have converged and VRAM is free

Record every run in the experiment tracker with the exact config hash and git SHA. An unreproducible
good result is worth nothing.

---

## 7. Baselines to train

Required for [`08-evaluation-protocol.md`](./08-evaluation-protocol.md). Budget time for these —
they are not optional, they are what makes the results interpretable.

| Baseline | Purpose |
|---|---|
| B1: Pretrained SepFormer (Libri2Mix), off-the-shelf | "What you get for free" — the original roadmap's plan |
| B2: SepFormer fine-tuned on AMI-Train | The original roadmap's *full* proposal, honestly executed |
| B3: Audio-only TSE (our C1 checkpoint) | Isolates the value of visual conditioning |
| B4: SEAVE without suppression loss | Ablates Novelty 3 |
| B5: SEAVE with oracle enrolment | Upper bound for Novelty 1 |
| B6: Oracle mask (IRM) | Theoretical ceiling for masking approaches |

B1 and B2 matter beyond the ablation table: they are the evidence for the architecture decision in
[`02-approach-review.md`](./02-approach-review.md) §F1. If B2 beats SEAVE, the review was wrong and
must be revised. Design the experiment so it *could* say that.

---

## 8. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN at step ~0 | SI-SDR with a zero-energy target | Add ε; skip batches where target energy ≈ 0 |
| Output is the input, unchanged | Model learned identity — target and mixture too correlated | Check overlap ratio in simulation; verify targets aren't the mixture |
| Output is silence | Suppression/silence weights too high | Lower λ₂/λ₅; check hinge τ |
| Great SI-SDR, audible other speaker | Exactly the SI-SDR blind spot | Confirms Novelty 3's premise; raise λ₂ |
| Visual makes no difference | ROIs misaligned or time-misaligned with audio | Verify ROI/audio alignment on a visualised sample **frame by frame** |
| Val >> train performance | Val set easier, or a leak | Check split disjointness |
| Works on sim, fails on AMI | Domain gap | More aggressive simulation realism; more C4 |
| Same-gender pairs much worse | Known audio-only weakness | Should improve at C2 — if not, visual pathway is broken |

---

## 9. Reproducibility

Every reported result must be reproducible from:
```
configs/<name>.yaml  +  git SHA  +  dataset manifest hash  +  seed
```

- Pin all seeds (python, numpy, torch, cuda)
- Log the dataset manifest hash (file list + sizes) with every run
- `pip freeze` / `uv.lock` captured per run
- Checkpoints tagged with the run ID
- `make reproduce RUN=<id>` re-runs evaluation from a checkpoint

`cudnn.benchmark=True` (non-deterministic) is enabled for speed. Document that exact bitwise
reproduction is not guaranteed; statistical reproduction (±0.2 dB across seeds) is. Report mean ± std
over **3 seeds** for headline numbers.

---

## 10. Compute budget

| Stage | GPU-hours (RTX A5000) |
|---|---|
| C0 Smoke | 0.5 |
| C1 Audio-only (from pretrained init) | 65 |
| C2 Add visual | 75 |
| C3 Realistic sim | 50 |
| C4 In-domain | 20 |
| HP search | 40 |
| Baselines (B1–B6) | 90 |
| Ablations (7 suites × 3 seeds) | 150 |
| **Total** | **≈ 490 GPU-hours ≈ 21 days continuous** |

Down from ~690 on a 12 GB card. Three reasons: gradient checkpointing is off (~35% faster), larger
batches raise utilisation, and C1 starts from a pretrained checkpoint rather than random weights.

**Wall clock:** ~9 weeks at 8 h/day of exclusive access; ~4–5 weeks with overnight runs. This is the
critical path and drives [`21-implementation-plan.md`](./21-implementation-plan.md).

**If over budget:** reduce ablation seeds to 2 · run ablations on an AMI-Eval subset · rent cloud GPUs
for the ablation sweep — it parallelises perfectly, so 150 h across 8 spot instances is under a day
for roughly $40.

**If you must stop early**, the tiers in
[`25-compute-and-hardware.md`](./25-compute-and-hardware.md) §4 are each a valid stopping point.
C1 + C3 + C4 alone (~135 h) delivers the primary objective; C2 raises the ceiling.
