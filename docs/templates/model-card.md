# Model Card — `<model-id>`

> Required for every trained checkpoint. Deployment tooling refuses checkpoints whose
> **Commercial use** field is not `yes` ([ADR-0013](../adr/0013-dataset-licensing.md)).

## Identity

| | |
|---|---|
| Model ID | `seave-tfgridnet-v1.0.0` |
| Track | research \| **production** |
| **Commercial use** | **yes \| no** |
| Date | |
| Git SHA | |
| Config | `ml/training/configs/….yaml` (hash: …) |
| Checkpoint SHA-256 | |

## Intended use

**Intended:** Extracting a single target speaker from conversational recordings with 2–4 speakers,
offline, English-dominant, conversational speech in rooms.

**Out of scope:** Real-time use · > 4 speakers · music/singing · speaker identification · telephony
or broadcast without re-evaluation · languages absent from training data.

## Architecture

| | |
|---|---|
| Backbone | TF-GridNet, N blocks, D=…, LSTM=… |
| Conditioning | speaker embedding + lip ROI, reliability-gated FiLM + cross-attention |
| Parameters | … M (+ … M frozen visual frontend) |
| Sample rate | 16 kHz |
| Inference chunk | 10 s, 2 s overlap |

## Training data

| Dataset | Split | Hours | Licence | Commercial |
|---|---|---|---|---|
| | | | | |

**Dataset manifest hash:** `…`
**Customer media used:** none \| opted-in only (count: …)

## Training

| | |
|---|---|
| Curriculum stage | C0 / C1 / C2 / C3 / C4 |
| Initialised from | |
| Epochs / steps | |
| Loss weights | si_sdr / suppression / consistency / mrstft / silence |
| Hardware, wall clock | |
| Seeds | |

## Evaluation — AMI-Eval

Mean ± 95% CI, 3 seeds. Report by slice, not only aggregate.

| Metric | 2 spk | 3 spk | 4 spk |
|---|---|---|---|
| SI-SDRi (dB) | | | |
| SIR (dB) | | | |
| Silence leakage (dB) | | | |
| DNSMOS OVRL | | | |
| Target-speaker WER | | | |
| Leakage word rate | | | |

**Sliced results:** same-gender vs mixed · overlap ratio bins · RT60 bins · visual quality bins.
**Benchmark (comparability only):** Libri2Mix SI-SDRi …

## Limitations

State plainly, with numbers:
- Quality at 4 speakers
- Behaviour with no visible face
- Same-gender pair performance
- Reverberation ceiling
- Language and accent coverage
- Headset-reference bleed floor bounding measured SI-SDR

## Ethical considerations

- Biometric derivatives are job-scoped and deleted ([ADR-0008](../adr/0008-ephemeral-biometrics.md))
- Known demographic gaps in training data:
- Misattribution risk and how it is disclosed to users:

## Version history

| Version | Date | Change | Δ SI-SDRi | Δ SIR |
|---|---|---|---|---|
| | | | | |
