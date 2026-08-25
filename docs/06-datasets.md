# 06 — Datasets

Four tiers, each with a distinct job. Tier 3 and Tier 4 are the ones that decide whether the product
actually works; Tiers 1–2 exist to get the model to a useful starting point cheaply.

---

## 1. Tier strategy

| Tier | Purpose | Data | Why |
|---|---|---|---|
| **T1 Synthetic audio** | Bootstrap the separator; validate the training loop | LibriMix, WSJ0-2mix-style | Cheap, unlimited, perfect ground truth |
| **T2 Synthetic AV** | Teach visual conditioning | VoxCeleb2-Mix, LRS3-Mix | Ground truth + real faces |
| **T3 Realistic AV** | Close the domain gap | AVSpeech-Mix, EasyCom, AMI, MISP | Real rooms, real conversation |
| **T4 In-domain** | Fine-tune + honest evaluation | **VVX-Train / VVX-Eval (self-recorded)** | Matches actual product input |

**The rule that matters:** every headline number in the report is reported on **VVX-Eval (T4)**.
Benchmark numbers on T1/T2 are reported only for comparability with literature and are labelled as
such. See [`03-research-landscape.md`](./03-research-landscape.md) §7.

---

## 2. Tier 1 — Synthetic audio-only

### LibriMix
| | |
|---|---|
| Source | `github.com/JorisCos/LibriMix` (generated from LibriSpeech + WHAM! noise) |
| Config | **Libri2Mix and Libri3Mix, `min` mode, 16 kHz, `mix_both` (noisy)** |
| Size | ~250 GB for the above; do **not** generate all sample-rate/mode combinations |
| Licence | CC BY 4.0 (LibriSpeech) ✅ |
| Use | Pretrain the separator backbone; establish literature-comparable numbers |

**Storage discipline** (carried from the original roadmap, which was right about this): the
generation script will produce ~500 GB+ if you let it generate 8 kHz + 16 kHz × `min` + `max` ×
`clean` + `both`. Pick one config. Delete intermediates after generation.

```bash
# ONE config only
python scripts/create_librimix_from_metadata.py \
  --librispeech_dir ./LibriSpeech --wham_dir ./wham_noise \
  --metadata_dir ./LibriMix/metadata/Libri2Mix \
  --librimix_outdir ./data/Libri2Mix \
  --n_src 2 --freqs 16k --modes min --types mix_both
```

### WHAMR!
Reverberant + noisy variant. Used specifically to test the **dereverberation ablation** (S1) and to
train reverb robustness. ~80 GB.

---

## 3. Tier 2 — Synthetic audio-visual

This is where the AV model actually learns lip-conditioning. It requires video, which LibriMix has
none of.

### VoxCeleb2-Mix (primary AV training corpus)
| | |
|---|---|
| Source | VoxCeleb2 (`robots.ox.ac.uk/~vgg/data/voxceleb`) |
| Content | ~1M utterances, 6k speakers, YouTube interviews, face crops available |
| Mixing | **We generate the mixtures ourselves** (see §5) |
| Licence | CC BY 4.0 for annotations; underlying video is YouTube-sourced — **research use** ⚠️ |
| Caveat | Link rot; older download scripts break. Budget time for acquisition. |

### LRS3-TED
| | |
|---|---|
| Content | TED talks, high-quality frontal faces, word-aligned transcripts |
| Quality | The cleanest AV speech data available |
| Licence | **Requires a signed agreement with BBC/Oxford; research-only, non-commercial** ⚠️⚠️ |
| Use | AV pretraining if the agreement is obtainable |

> **Licensing decision.** LRS2/LRS3 are non-commercial research-only. If VisioVox is ever
> commercialised, models trained on them cannot ship. **Mitigation: train the production model on
> VoxCeleb2 + AVSpeech + VVX-Train only, and use LRS3 solely for the research ablations reported in
> the paper.** Track which checkpoint saw which data in the model card. Decided in
> [ADR-0013](./adr/0013-dataset-licensing.md).

---

## 4. Tier 3 — Realistic audio-visual

| Dataset | Content | Value | Licence |
|---|---|---|---|
| **AVSpeech** | 4700 h, 290k YouTube speakers, single-speaker clean segments | Large-scale AV mixing source, more diverse than VoxCeleb2 | YouTube-derived ⚠️ |
| **EasyCom** | Meta egocentric conversations, real overlapping speech, multi-mic, with annotations | **Real overlap with ground truth** — extremely rare and valuable | CC BY-NC 4.0 ⚠️ |
| **AMI Meeting Corpus** | 100 h meetings, headset (per-speaker!) + far-field mics, video | **Headset channels give near-ground-truth per-speaker audio for real overlap** | CC BY 4.0 ✅ |
| **MISP** | Chinese home-TV multi-modal conversation, far-field | AV diarization + separation | research ⚠️ |
| **VoxConverse** | Diarization benchmark, in-the-wild | DER evaluation | CC BY ✅ |
| **AVA-ActiveSpeaker** | ASD benchmark | Validating the ASD stage | CC BY ✅ |

**AMI is the single most useful Tier-3 dataset for this project** and is under-used in the
separation literature. Its close-talking headset microphones provide a per-speaker reference for
*genuinely overlapping real conversational speech* — which synthetic mixing cannot reproduce.
The headset signal is not a perfect reference (it carries bleed), but it is far closer to the truth
than anything synthetic. Use it for evaluation with that caveat stated explicitly.

---

## 5. Mixture simulation

The realism of simulated mixtures determines how far T1/T2 gains transfer. Naive `a + b` mixing is
the main reason benchmark-trained models collapse on real data.

### Naive (what LibriMix does, what most papers do)
```
mix = a + b          # both fully overlapped, anechoic, level-balanced
```

### VVX realistic simulation (ours)
```python
def simulate(sources, video_tracks):
    # 1. Realistic overlap — NOT 100%. Real conversation: 5-20% overlap.
    #    Sample a turn-taking schedule from AMI's measured turn/gap/overlap distributions.
    timeline = sample_turn_schedule(n_speakers=len(sources),
                                    overlap_ratio=uniform(0.05, 0.35),
                                    dist='ami_empirical')

    # 2. Room simulation — convolve each source with its OWN RIR from its OWN position
    room = sample_room(dim=uniform([3,3,2.4],[10,8,3.5]), rt60=loguniform(0.15, 0.8))
    for src in sources:
        src.audio = convolve(src.audio, rir(room, mic_pos, sample_speaker_pos(room)))

    # 3. Level imbalance — one speaker is closer to the mic. Real recordings are not balanced.
    for src in sources:
        src.audio *= db_to_lin(normal(0, 6))       # ±6 dB spread

    # 4. Additive noise at realistic SNR
    mix = sum(s.audio for s in sources) + noise * snr_scale(uniform(5, 25))

    # 5. Codec/device degradation — real uploads are compressed
    if coin(0.5):
        mix = codec_roundtrip(mix, choice(['aac_64k','opus_32k','mp3_96k','amr_nb']))

    # 6. Video degradation matched to the audio conditions
    for v in video_tracks:
        v = apply(v, [random_occlusion(p=0.15), motion_blur(p=0.2),
                      compression(crf=uniform(23,35)), profile_warp(p=0.25),
                      frame_drop(p=0.1), resolution_drop(p=0.2)])
    return mix, timeline, sources, video_tracks
```

Steps **1**, **3** and **6** are the ones usually skipped and the ones that matter most:

- **Step 1** — training only on 100%-overlapped mixtures produces a model that has never seen a
  single-talker region and behaves badly on one. Since 80–95% of real timelines are single-talker,
  this is the dominant real-world condition.
- **Step 3** — level-balanced training makes the model rely on relative level as an implicit cue,
  which then fails when one speaker is twice as far from the mic.
- **Step 6** — corrupting video during training is what makes the Novelty-2 reliability gate learn
  to *detect* unreliable video rather than being told about it.

---

## 6. Tier 4 — VVX: our own corpus

**The most important dataset in the project.** It is the only data that matches actual product input
and the only basis for credible claims.

### Recording protocol

| Aspect | Spec |
|---|---|
| Sessions | ≥ 30 |
| Duration | 3–10 min each |
| Speakers | 10 sessions × 2 spk, 15 × 3 spk, 5 × 4 spk |
| **Reference capture** | **Each speaker wears a lavalier/headset mic on a separate channel** |
| Mixture capture | A single camera mic at 1–3 m — this is the actual system input |
| Sync | Clap slate at start; align channels by cross-correlation |
| Rooms | 6+: small office, large meeting room, echoey hall, outdoor, home, car |
| Overlap | Scripted: 0%, 10%, 25%, 50% target overlap segments per session |
| Visual | Frontal, profile, occluded (hand/mask), off-screen, poor lighting, backlit |
| Speakers | ≥ 12 people, gender-balanced, **including same-gender pairs** (the hard case) |
| Content | Natural conversation + scripted overlap drills |
| Consent | Written, explicit, covering research use, retention and publication |

**The per-speaker reference microphone is what makes this dataset scientifically valuable.** It
gives real ground truth for real overlap — the thing synthetic mixing fundamentally cannot provide.
Without it, VVX is only a qualitative demo set. With it, it is the evaluation corpus.

Caveat to state in the report: headset references carry bleed from other speakers (typically
−15 to −25 dB). This bounds achievable measured SI-SDR. Measure the bleed floor and report it, so
readers can interpret the numbers correctly.

### Splits

| Split | Sessions | Use | Rule |
|---|---|---|---|
| VVX-Train | 18 | In-domain fine-tuning | — |
| VVX-Val | 6 | Hyperparameters, early stopping | — |
| **VVX-Eval** | **6** | **All reported results** | **Speaker-disjoint AND room-disjoint from train. Touched only at milestone evaluations.** |

**Leakage discipline:** no speaker and no room appears in both VVX-Train and VVX-Eval. Speaker
overlap between splits is the most common way separation results become quietly meaningless — the
model memorises voices instead of learning to separate.

### Demo set
3 additional sessions, consented explicitly for public demonstration, used for the landing page and
the demo reel. **Never** used for training or evaluation.

---

## 7. Storage plan

| Dataset | Size | Location |
|---|---|---|
| LibriSpeech + WHAM! | 90 GB | WSL2 local NVMe |
| Libri2Mix + Libri3Mix (16k min both) | 250 GB | WSL2 local NVMe |
| WHAMR! | 80 GB | local, deletable after ablation |
| VoxCeleb2 (face crops + audio) | 300 GB | local |
| AVSpeech (subset, 500 h) | 200 GB | local |
| AMI (headset + far-field + video) | 100 GB | local |
| VVX raw | 60 GB | local + **encrypted offsite backup** |
| VVX processed | 30 GB | local |
| **Total** | **~1.1 TB** | dedicated 2 TB NVMe recommended |

Per the original roadmap's advice, confirmed: keep everything in the **WSL2 filesystem**
(`~/data/...`), never on `/mnt/c`. The 9p filesystem bridge is several times slower and will
bottleneck the dataloader before the GPU is saturated.

**VVX raw is irreplaceable** — you cannot re-record it. Back it up encrypted, offsite, and verify
restores. Everything else is re-downloadable.

---

## 8. Ethics and consent

Two separate things, with very different costs:

### Written participant consent — always, no exceptions

- Written informed consent from every participant **before** recording
- Covers: research use, model training, retention period, and — as a **separate** checkbox — whether
  clips may appear in publications or public demos
- Right to withdraw → recordings deleted, model retrained without them at the next cycle
- No sensitive personal content; participants briefed on topic scope beforehand
- Raw recordings encrypted at rest; access limited to the project
- Consent records stored separately from the recordings

Template: [`templates/vvx-consent-form.md`](./templates/vvx-consent-form.md). Requires no approval to
use. This is the part that protects participants and makes the data usable in a report.

### Institutional ethics review — check whether it applies

Whether formal IRB/ethics-board review is required depends on your institution and on whether you
intend to publish. Many institutions exempt student coursework that records consenting adults on
non-sensitive topics, or handle it with a short exemption form.

**Ask your supervisor before assuming a six-week process.** If review is required, it gates only the
VVX recording (Phase 3) — not the baseline pipeline, the application, or training on public data.
See [`21-implementation-plan.md`](./21-implementation-plan.md) §Week 0.

**If VVX cannot happen:** AMI's headset channels are the documented fallback (R-25) — real
conversational overlap with per-speaker references, weaker than purpose-recorded data but publishable.
Note the substitution explicitly in the report.

---

## 9. Dataset card

Every trained model ships with a card recording:
- Which datasets, which splits, which versions
- Licence status of each and whether the checkpoint is commercially usable
- Simulation parameters used
- Known biases: language (English-dominant), accent, recording-device distribution, demographics
- Evaluation splits and their disjointness guarantees

Template: `docs/templates/model-card.md`.
