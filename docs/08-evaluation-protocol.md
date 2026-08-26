# 08 — Evaluation Protocol

The rule: **every headline number is measured on AMI-Eval by our own harness**, and is reported
as a meeting-recording result rather than a general one ([ADR-0015](./adr/0015-ami-replaces-vvx.md)). Benchmark numbers on
public sets are reported for comparability only and labelled as such.

---

## 1. Metric set

### 1.1 Separation quality

| Metric | Definition | Why | Target |
|---|---|---|---|
| **SI-SDRi** | SI-SDR(ŝ,s) − SI-SDR(mix,s) | Standard, comparable to literature | ≥ 14 dB (2spk) / ≥ 11 dB (3spk) |
| **SIR** | Target energy ÷ interferer energy in ŝ | ⭐ **The metric that matches the requirement.** "Other speakers almost zero." | ≥ 20 dB (2spk) |
| SAR | Target ÷ artifact energy | Separates artifact from leakage | report |
| **Silence-region leakage** | RMS of ŝ where target is silent, dB rel. mix | ⭐ Where leakage is most audible — 50–70% of the timeline | ≤ −30 dB |
| SDRi | Classic SDR improvement | Legacy comparability | report |

**Why SIR is elevated to co-primary.** SI-SDR lumps distortion, noise and leakage into one number.
Two systems at 14 dB SI-SDRi can differ by 10 dB in SIR — one sounds a bit processed, the other has
a second person audibly talking underneath. Only the second is a product failure. Report both; when
they disagree, the SIR number is the one that describes the user's experience.

### 1.2 Perceptual quality

| Metric | Type | Notes |
|---|---|---|
| **DNSMOS P.835** (SIG/BAK/OVRL) | non-intrusive | No reference needed → works on real user data at inference time |
| **UTMOS** | non-intrusive | Second opinion; the two disagree usefully |
| PESQ (wideband) | intrusive | Standard but weakly correlated with separation quality — report, don't optimise |
| eSTOI | intrusive | Intelligibility |
| **MOS panel** | human | ⭐ Ground truth. §4 below. |

### 1.3 Downstream (captions)

| Metric | Definition | Target |
|---|---|---|
| **Target-speaker WER** | WER of ŝ's transcript vs target reference | ≤ 15% |
| ΔWER | WER(mixed) − WER(isolated) | The practical gain from isolation |
| **Cross-stream leakage word rate** | ⭐ Words in speaker A's transcript actually spoken by B | ≤ 3% |
| Caption attribution F1 | Per-word speaker attribution | ≥ 0.92 |
| Word timing error | Median abs. error vs forced alignment | ≤ 120 ms |

**Cross-stream leakage word rate is a metric we define** (see [`04-novelty.md`](./04-novelty.md) §6).
It measures the failure the product must not have: putting words in the wrong person's mouth. It is
not measured by anything in the standard separation literature.

### 1.4 Upstream stages

| Stage | Metric | Target |
|---|---|---|
| Diarization | DER, JER | ≤ 12% / ≤ 20% |
| Speaker counting | Exact-match accuracy over K ∈ 2..4 | ≥ 92% |
| Overlap detection | F1 on overlapped frames | ≥ 0.75 |
| Face detection | mAP@0.5 | ≥ 0.90 |
| ASD | mAP on AVA-ActiveSpeaker | ≥ 0.90 |
| Face↔voice binding | Accuracy on AMI-Eval | ≥ 0.90 |

### 1.5 Calibration

| Metric | Purpose | Target |
|---|---|---|
| **ECE** of the trust score | Does stated confidence match observed accuracy? | ≤ 0.05 |
| Reliability diagram | Visual calibration check | monotone |

An uncalibrated confidence score is worse than none — it tells the user to trust output that is
wrong. This is a release gate, not a nice-to-have.

### 1.6 System

| Metric | Target |
|---|---|
| RTF (pipeline / media duration) | ≤ 2.0× per speaker |
| Peak VRAM | ≤ 10 GB |
| Speaker-switch latency (p95) | ≤ 120 ms |
| A/V drift over 10 min | ≤ 40 ms |
| Job success rate | ≥ 98% |

---

## 2. Harness

One implementation, pinned, used for **every** system including all baselines. We do not quote
numbers from papers.

```python
# eval/harness.py
METRICS = {
  'si_sdr':  torchmetrics.audio.ScaleInvariantSignalDistortionRatio(zero_mean=True),
  'sir':     bss_eval_sir,                   # mir_eval / fast_bss_eval
  'pesq':    torchmetrics.audio.PerceptualEvaluationSpeechQuality(16000, 'wb'),
  'stoi':    torchmetrics.audio.ShortTimeObjectiveIntelligibility(16000, extended=True),
  'dnsmos':  DNSMOSP835(),
  'utmos':   UTMOSScore(),
}

def evaluate(system, dataset, out_dir):
    rows = []
    for item in dataset:                          # deterministic order
        est = system(item.mixture, item.enrolment, item.visual)
        for k, spk in enumerate(item.speakers):
            rows.append({
              'item': item.id, 'speaker': spk.id,
              'n_speakers': item.n_speakers,
              'overlap_bin': bin_overlap(item.overlap_ratio),
              'same_gender': item.same_gender,
              'rt60_bin': bin_rt60(item.rt60),
              'visual_quality_bin': bin_visual(item.visual_quality),
              **{m: fn(est[k], item.refs[k]) for m, fn in METRICS.items()},
              **downstream_metrics(est[k], item.refs[k], item.transcripts[k]),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / 'per_item.parquet')   # raw rows always retained
    return summarise(df)
```

**Rules:**
- Per-item rows are always saved. Aggregates without raw rows cannot be re-sliced or audited.
- Report **mean ± 95% CI** (bootstrap, 1000 resamples), never bare means.
- Statistical significance via **paired bootstrap** on per-item deltas — systems are evaluated on
  identical items, so paired tests are both valid and much more sensitive.
- Fixed seed; deterministic dataset ordering.

---

## 3. Evaluation matrix

Aggregate numbers hide everything interesting. Every result is sliced by:

| Dimension | Bins |
|---|---|
| Speaker count | 2 / 3 / 4 |
| Overlap ratio | 0–10% / 10–25% / 25–50% / >50% |
| Gender pairing | same / mixed |
| Reverberation | RT60 < 0.3 / 0.3–0.6 / > 0.6 s |
| Visual quality | good / degraded / absent |
| SNR | > 20 / 10–20 / < 10 dB |

**Required plots** (these become the report's figures):
1. **Degradation curve:** SI-SDRi and SIR vs speaker count — the honest answer to "does 3-speaker work?"
2. SI-SDRi vs overlap ratio
3. Same-gender vs mixed-gender bar chart, ours vs audio-only baseline — the clearest demonstration of the visual contribution
4. Visual quality ablation: good / degraded / absent — demonstrates Novelty 2
5. SIR vs SI-SDR scatter — demonstrates that Novelty 3 moves SIR at negligible SI-SDR cost
6. Reliability diagram for the trust score

---

## 4. Listening test

Numbers do not establish listenability. This is required, not optional (G3).

| Aspect | Spec |
|---|---|
| Design | MUSHRA-inspired, simplified |
| Participants | ≥ 15, non-expert, normal hearing (self-reported) |
| Stimuli | 20 clips × 6 systems (B1, B2, B3, SEAVE-faithful, SEAVE-natural, hidden reference) |
| Anchor | 3.5 kHz low-pass anchor to validate rating spread |
| Scales | (a) overall quality 1–5, (b) **"how audible is the other speaker?"** 1–5, (c) naturalness 1–5 |
| Presentation | Randomised order, blind, headphones required |
| Screening | Discard participants who rate the hidden reference below 4 |
| Analysis | Mean ± CI per system; paired t-test / Wilcoxon vs baseline |

Scale (b) is the one that matters most and is absent from standard MUSHRA. It measures the actual
user requirement directly.

**Also validate:** correlation between DNSMOS and human MOS on this data. If it is weak, say so and
downgrade DNSMOS to a secondary metric in the report. Honest reporting of metric–perception
mismatch is itself a finding worth stating.

---

## 5. Ablation suite

Every ablation: 3 seeds, AMI-Eval, paired bootstrap significance.

### A1 — Architecture (validates F1 of the review)
| System | Question |
|---|---|
| B1 pretrained SepFormer | What does off-the-shelf blind separation give? |
| B2 SepFormer fine-tuned on VVX | Does the original roadmap's plan work? |
| B3 audio-only TSE | Is TSE better than BSS, holding modality fixed? |
| SEAVE (AV-TSE) | Does visual conditioning add on top? |

Plus, critically: **permutation-error rate on full-length output** for B1/B2 vs SEAVE. This
quantifies F1.1 — the argument that BSS cannot maintain identity over long recordings. If B1/B2
show a low permutation-error rate, the review's central claim is weakened and must be revised.

### A2 — Self-enrolment (Novelty 1)
oracle enrolment · purity-weighted (ours) · longest-region · random 3 s · contaminated mean

### A3 — Modality-adaptive conditioning (Novelty 2)
fixed-AV · hard switch · **reliability-gated (ours)** — each evaluated under clean / occluded /
profile / absent video

### A4 — Suppression-first objective (Novelty 3)
SI-SDR only · +suppression · +consistency · +mrstft · +silence · full — reported on **SI-SDRi, SIR,
silence leakage, DNSMOS** together, since the whole claim is about the trade between them

### A5 — Gated restoration (Novelty 4)
none · ungated generative · **gated (ours)** — on DNSMOS, WER, **hallucination rate**

### A6 — Leakage repair (Novelty 5)
none · caption-only · full — on SIR, attribution F1, ECE

### A7 — Pipeline stages
| Ablation | Expected finding |
|---|---|
| − dereverberation | quality drop, larger at high RT60 |
| − denoising | drop at low SNR |
| − single-talker passthrough | quality drop **and** large RTF increase (validates F11) |
| − loudness normalisation | switch-time level jumps (measured as inter-track LUFS spread) |

---

## 6. Continuous evaluation

Runs in CI on every model change:

```
CI-EVAL-QUICK   30 items, ~5 min   → every PR touching ml/
CI-EVAL-FULL    AMI-Eval, ~2 h     → nightly + pre-release
```

**Regression gates (blocking):**

| Gate | Threshold |
|---|---|
| SI-SDRi regression | > 0.5 dB → block |
| SIR regression | > 1.0 dB → block |
| WER regression | > 1.0 point → block |
| Hallucination rate | > 0.5% absolute → block |
| Trust-score ECE | > 0.05 → block |
| RTF regression | > 20% → block |

Results are appended to `eval/history.jsonl` and plotted on a dashboard so quality over time is
visible rather than rediscovered at release.

---

## 7. Report structure

1. **Introduction** — problem, why it matters, the gap
2. **Related work** — from [`03-research-landscape.md`](./03-research-landscape.md)
3. **Method** — SEAVE, five contributions
4. **Experimental setup** — datasets, splits, disjointness, metrics, harness, baselines
5. **Results** — main table, degradation curves, sliced analyses
6. **Ablations** — A1–A7
7. **Listening test**
8. **System** — architecture, latency, delivery (Contribution 6)
9. **Limitations** ⭐
10. **Ethics and privacy**
11. **Conclusion and future work**

### §9 Limitations — required content

State plainly, without hedging:

- **Speaker-count ceiling.** Quality at 4 speakers, with the numbers. If it is unusable, say it is
  unusable.
- **Off-screen speakers** fall back to audio-only with measured quality loss.
- **Generative restoration can hallucinate.** Report the measured rate, not zero.
- **AMI-Eval is small** (6 sessions, 12 speakers) and English-only. Results may not generalise
  across languages, accents or recording conditions.
- **Headset reference bleed** bounds measurable SI-SDR. Report the measured bleed floor.
- **Domain scope** — the model was trained and evaluated on conversational speech in rooms. No claim
  is made about telephony, broadcast, singing or heavily accented speech outside the training set.
- **Compute** — training used a single consumer GPU; larger-scale training would likely improve results.

The limitations section is a release gate. A results section without it is not shippable — and for
a project whose central design principle is honest disclosure of uncertainty
([`00-charter.md`](./00-charter.md) §7), an overclaimed report would contradict the product itself.
