# 27 — Phase 1 baseline report (Tier 0)

The honest "before" measurement, and the empirical test of
[ADR-0001](./adr/0001-target-speaker-extraction-over-blind-separation.md).

**Headline: ADR-0001 holds.** Naive stitching of blind-separation output flips speaker identity on
**29.8%** of window transitions across three meetings, and the resulting tracks are **10.7 dB worse
than returning the unprocessed mixture**. The same separator output, stitched with an oracle, is
**+2.55 dB better** than the mixture — so the separator is not the bottleneck. Stitching is.

---

## 1. What was measured

| | |
|---|---|
| Material | AMI meetings ES2002a (Edinburgh), IS1000a (Idiap), TS3003a (TNO) |
| Full-length audio | 72.7 minutes total |
| Separator | `speechbrain/sepformer-whamr16k`, 2 sources, 16 kHz |
| Windows | 4.0 s, 2.0 s hop (50% overlap) |
| References | Per-participant headset microphones (AMI ships one per speaker) |

Using AMI rather than a generic video corpus was the decision that made this report possible: the
per-speaker headsets are ground truth, so the baseline is **measured** rather than described, and
Tier 1 has something to be compared against on in-domain data.

Mixtures are the sum of the headsets, not a far-field room mic, because that is the signal the
references actually decompose into. A room mic would make SI-SDR meaningless.

---

## 2. Permutation-error rate ⭐

The Phase 1 exit criterion. Measured on **full-length** output, per
[`02-approach-review.md`](./02-approach-review.md) §F1.1.

For each window, every output channel is scored against every reference with SI-SDR and matched
with the Hungarian algorithm; that gives the true channel→speaker mapping. The rate is the fraction
of consecutive scored windows where the mapping changes.

| Meeting | Site | Windows | Scored | Flips | Rate | 95% CI |
|---|---|---|---|---|---|---|
| ES2002a | Edinburgh | 636 | 260 | 62 | 23.9% | ±5.2% |
| IS1000a | Idiap | 791 | 439 | 152 | 34.7% | ±4.5% |
| TS3003a | TNO | 752 | 424 | 120 | 28.4% | ±4.3% |
| **Pooled** | | **2179** | **1123** | **334** | **29.8%** | |

Windows without two simultaneously active speakers are skipped — they carry no information about
ordering, and scoring them would drag the rate toward zero.

**Roughly one window transition in three swaps who is on which track.** A 21-minute meeting is
~630 windows. Identity survives a handful of transitions at best.

Interpretation thresholds were fixed in the code before the run, so the reading is not retrofitted:
`<1%` would have meant ADR-0001's premise was weak and the architecture decision needed revisiting;
`>10%` means naive stitching is unusable. The result is not near the boundary.

---

## 3. What the ambiguity costs

Both columns use **identical separator output** and differ only in how windows are assigned and
scaled, so the gap isolates stitching from separation quality.

| Clip | mixture | naive | oracle | naive SI-SDRi | oracle SI-SDRi | gap |
|---|---|---|---|---|---|---|
| ES2002a | −4.66 | −21.80 | −7.87 | **−17.14** | −3.20 | 13.93 |
| IS1000a | −3.44 | −4.90 | +5.08 | **−1.47** | +8.52 | 9.99 |
| TS3003a | −12.41 | −25.89 | −10.07 | **−13.48** | +2.34 | 15.82 |
| **mean** | | | | **−10.70** | **+2.55** | **13.25** |

SI-SDRi is improvement over doing nothing — returning the mixture unchanged as every track.

**Naive stitching is actively harmful**: −10.7 dB means the output is far worse than the input it
was derived from. A user would prefer the raw audio.

---

## 4. A second failure mode, not predicted by F1.1

F1.1 describes permutation. There is another problem underneath it.

On IS1000a the oracle assignment is **decisive** — median SI-SDR margin between the two possible
assignments is 16 dB, and never below 1 dB — yet it still flips on 25.6% of transitions. The flips
are real, not noise in the matching.

More importantly, **correct per-window identity is not sufficient**. The separator gives no
guarantee that the same speaker emerges at the same *scale or polarity* from two independent calls.
SI-SDR is scale invariant, so matching never notices; overlap-add does, and adjacent windows cancel
where they overlap. Before scale correction the oracle scored *worse* than naive on two of three
clips, which is how this surfaced.

The oracle in §3 therefore applies a least-squares projection onto the reference per window,
correcting scale and polarity together. **That correction is worth most of the 13.25 dB gap.**

This strengthens ADR-0001 rather than complicating it. A stitcher would have to solve identity *and*
gain continuity across ~630 windows without an answer key. TSE avoids both by construction: output
identity is bound to the conditioning signal, and there is one output per speaker, so there is
nothing to reorder or rescale.

---

## 5. What this does not yet establish

Stated plainly, because these caveats bound what the numbers may be quoted for.

| Caveat | Consequence |
|---|---|
| **2-source separator, 4-speaker meetings** | Two participants are unrecoverable by construction. The absolute SI-SDR floor is set by the mismatch, not only by model quality. |
| **Sparse overlap (5.6–14.6%)** | AMI is mostly single-talker. [ADR-0010](./adr/0010-single-talker-passthrough.md) says separating already-clean audio degrades it — so part of the −10.7 dB is the system doing something ADR-0010 exists to prevent. |
| **No single-talker routing yet** | The separator ran over the whole timeline. Applying ADR-0010 first is required before this is a fair Tier 0 number. |
| **docs/25 §4 expects Tier 0 at +6 to +9 dB SI-SDRi** | Oracle reaches +2.55 dB. The expectation likely assumes 2-speaker material with routing; it should not be treated as contradicted until measured under those conditions. |

**The permutation result is robust to all of these.** It measures assignment stability, not quality,
and it is consistent across three independent recording sites with 1123 scored windows.

The SI-SDR comparison is the part that needs the caveats.

---

## 6. Reproducing

```bash
uv run python scripts/fetch_testvideos.py          # build clips from AMI
uv run python scripts/measure_permutation_full.py  # full-length permutation rate
uv run python scripts/run_baseline.py              # naive vs oracle, SI-SDRi
```

Outputs land in `~/data/baseline/`. Stitched audio is written per clip under `baseline/` so the
failure is audible, not only tabulated — the speaker swap is obvious on the naive tracks.

---

## 7. Measurement bugs found and fixed

Recorded because each produced plausible numbers, and three of them favoured the wrong conclusion.

| Bug | Effect | Fix |
|---|---|---|
| Per-track VAD threshold | Counted −28 dB headset bleed as speech; reported 79.7% overlap on a clip that is 5.6% | Dominance test against the loudest channel |
| Absolute −50 dBFS activity floor | AMI meetings differ ~30 dB in gain; IS1000a scored 791/791 windows as overlapped | One shared, gain-relative VAD |
| Truncated download accepted | A headset arrived at 14 MB of 40 MB; measurement silently ran on 7.3 of 21 minutes | Verify against `Content-Length` |
| Oracle indexed by reference id | Both tracks read channel 0 when best matches were speakers 2–3 | Index by track; fix the target set globally |
| Oracle ignored scale/polarity | Oracle scored worse than naive — impossible for an upper bound | Least-squares projection per window |
