# ADR-0002 — TF-GridNet as the separator backbone

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** ADR-0001, [`05-ml-architecture.md`](../05-ml-architecture.md) §8

## Context

Having chosen TSE (ADR-0001), we need a separator architecture to adapt for conditioning. Constraints:
a single 12 GB consumer GPU; real recordings are **reverberant** (the dominant real-world
degradation); the model must accept per-frame conditioning from two modalities.

## Options considered

| Option | WSJ0-2mix SI-SDRi | Notes |
|---|---|---|
| **Conv-TasNet** | ~15.3 dB | Small, fast, easy. Weakest quality. Time-domain masking handles reverb tails poorly. |
| **SepFormer** | ~22.3 dB | Dual-path transformer, mature SpeechBrain recipe. Heavy attention cost at long context. |
| **TF-GridNet** ✅ | ~23.5 dB | Time-frequency domain; intra-frame + inter-frame + full-band attention. Strongest on reverberant data. |
| **MossFormer2** | ~24.1 dB | Marginally better; less mature tooling; heavier. |
| **SPMamba** | competitive | Promising long-context scaling; immature ecosystem. |

## Decision

**TF-GridNet**, 6 blocks, `D=96`, LSTM hidden 192, 4 attention heads, STFT 512/128. ≈14 M parameters.

## Rationale

Two properties decide it beyond raw benchmark numbers:

1. **Reverberation robustness.** TF-GridNet's time-frequency formulation models reverb tails better
   than time-domain masking. Our data is reverberant rooms, not anechoic chambers — this is where
   the real quality lives, and where the benchmark ranking and the deployment ranking most nearly
   agree.
2. **Clean conditioning insertion points.** The per-block structure gives natural places to apply
   FiLM modulation and cross-attention to the visual stream, without redesigning the backbone.

MossFormer2's ~0.6 dB edge does not survive the domain shift to real data and costs tooling maturity
we cannot afford on a solo project. Conv-TasNet remains available as a fast ablation baseline.

## Consequences

**Positive** — Strong quality ceiling; good reverb behaviour; conditioning fits naturally.

**Negative** — Heavier than Conv-TasNet: requires bf16, gradient checkpointing and gradient
accumulation to fit 12 GB. Slower training. The VRAM tuning ladder in
[`07-training-playbook.md`](../07-training-playbook.md) §2 exists because of this choice.

**Neutral** — Reference config may shrink if VRAM proves tight; the architecture choice is
independent of the size.

## Revisit when

- The VRAM ladder reaches step 5 (real capacity reduction) — then a smaller architecture may beat a
  crippled TF-GridNet.
- A state-space backbone shows a clear win on reverberant conversational data specifically.
- Inference RTF becomes the binding constraint → distil or swap to a lighter backbone.
