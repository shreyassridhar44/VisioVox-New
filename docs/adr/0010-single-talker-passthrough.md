# ADR-0010 — Route single-talker regions around the extractor

- **Status:** Accepted
- **Date:** 2026-08-25
- **Related:** [`02-approach-review.md`](../02-approach-review.md) §F11, [`05-ml-architecture.md`](../05-ml-architecture.md) §8.1, [`20-performance-cost.md`](../20-performance-cost.md) §2

## Context

Real conversations are **5–20% overlapped**. The remaining 80–95% of the timeline has exactly one
person speaking — audio that needs no separation at all.

Separation models trained predominantly on fully-overlapped mixtures behave badly on single-speaker
input: they attempt to split a single source, producing spectral holes and artifacts on audio that
was already clean. The archived roadmap does not mention this case.

## Options considered

### A — Process everything through the extractor
**Pros:** Uniform code path; one behaviour to reason about.
**Cons:** Degrades 80–95% of the output. Wastes 80–95% of the GPU time. Both problems are severe.

### B — Train the model to handle single-talker input
**Pros:** Uniform path; theoretically clean.
**Cons:** Requires substantial single-talker training data and does not fully remove the artifact —
it only reduces it. Still pays full GPU cost on regions that need no processing.

### C — Route around the extractor ✅
Where diarization says one speaker and overlap probability is low:
- If it is the target speaker → **pass the enhanced audio through unchanged**
- If it is another speaker → **emit silence**
Otherwise → run the extractor.
Equal-power 30 ms crossfade at every boundary; overlap regions dilated by 200 ms so the extractor has
context and the crossfade lands in processed territory.

## Decision

**Option C.** Combined with option B's training data as a safety net — the model still sees some
single-talker input during training so that routing errors degrade gracefully.

## Rationale

This is the highest ratio of benefit to effort in the entire pipeline, and it improves **three
independent axes at once**:

1. **Quality** — the majority of what the user hears is the original enhanced audio, untouched. No
   model can improve on that, and any model can damage it.
2. **Cost** — 60–80% reduction in S5 GPU time ([`20-performance-cost.md`](../20-performance-cost.md)
   §2). The single largest performance lever available.
3. **Faithfulness** — most of the output is verifiably unmodified source audio, which matters for the
   journalist use case and strengthens the Faithful-track guarantee (ADR-0005).

The general principle is worth stating because it recurs: **the biggest wins in this pipeline come
from not processing things that don't need processing.** The same reasoning drives VAD-gated ASR,
skipping the leakage audit at low overlap, and skipping denoising on already-clean audio.

The dependency this introduces is on diarization accuracy. A missed overlap region means leakage
passes through unprocessed. This is mitigated by the low threshold (overlap probability < 0.10 to
qualify for passthrough) — the router is deliberately biased toward processing when uncertain, since
unnecessary processing costs quality slightly while missed processing costs it severely.

## Consequences

**Positive** — Large quality gain, large cost reduction, stronger faithfulness. Makes NFR-PERF-02
achievable.

**Negative** — Introduces a dependency on diarization accuracy at region boundaries. Crossfade
boundaries are a potential artifact source if the window is wrong. Two code paths where there was
one. Requires an ablation to quantify the benefit honestly (A7).

**Neutral** — Thresholds (0.10 overlap probability, 200 ms dilation, 30 ms crossfade) are tunable;
the routing decision is not.

## Revisit when

- Ablation A7 shows the passthrough path is *worse* than processing everything — which would indicate
  a boundary-artifact problem, not a flaw in the principle.
- Diarization becomes the dominant error source at boundaries → widen dilation or raise the threshold.
