"""Forced alignment for word timings (S7, docs/05 §10).

docs/21 lists "Whisper + WhisperX" for this stage. WhisperX is not installed,
deliberately: it pins `torch~=2.8` and this project runs torch 2.13 with a
verified CUDA build. Downgrading torch to gain an alignment package would put
SpeechBrain, pyannote and the extractor at risk for a capability torchaudio
already ships.

What WhisperX actually contributes over Whisper's own timestamps is CTC forced
alignment against a phonetic acoustic model. `torchaudio.functional.forced_align`
plus the MMS_FA bundle is that same technique, so this module takes the
capability rather than the dependency.

The difference is worth having. Whisper infers word times from decoder
cross-attention, which drifts — commonly by 100-300 ms and worse after long
pauses. Forced alignment finds the actual acoustic boundary. docs/01 wants word
timing accurate enough to drive a karaoke-style caption view, and attention
timings are not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from .s7_transcribe import Segment, Transcript, Word
from .types import ANALYSIS_SAMPLE_RATE, StageResult, StageStatus

STAGE = "S7_align"
VERSION = "1.0.0"


@dataclass
class Aligner:
    """Holds the acoustic model and its tokenizer between calls."""

    bundle: object
    model: object
    tokenizer: object
    labels: tuple[str, ...]
    device: str


def load_aligner(device: str = "cuda") -> Aligner:
    import torch
    from torchaudio.pipelines import MMS_FA

    bundle = MMS_FA
    model = bundle.get_model().to(torch.device(device))
    tokenizer = bundle.get_tokenizer()
    return Aligner(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        labels=tuple(bundle.get_labels()),
        device=device,
    )


def _normalise(word: str) -> str:
    """Reduce a word to what the alignment model's vocabulary can represent.

    Punctuation and case are not in the CTC vocabulary; leaving them in makes
    the token lookup fail and drops the word from alignment entirely, which is
    worse than aligning a stripped form.
    """
    return "".join(c for c in word.lower() if c.isalpha() or c == "'").strip("'")


def align_transcript(
    transcript: Transcript,
    audio: np.ndarray,
    aligner: Aligner,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> tuple[Transcript, StageResult]:
    """Refine word timings by forced alignment.

    Falls back to the original timings rather than failing: a transcript with
    approximate times is far more useful than none, and this stage is an
    accuracy improvement, not a correctness requirement.
    """
    import torch

    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)

    words: list[tuple[int, int, Word]] = []
    for si, seg in enumerate(transcript.segments):
        for wi, w in enumerate(seg.words):
            if _normalise(w.text):
                words.append((si, wi, w))

    if not words:
        result.status = StageStatus.DEGRADED
        result.warn("nothing_to_align")
        result.seconds = time.perf_counter() - t0
        return transcript, result

    try:
        refined = _run_alignment(words, audio, aligner, sample_rate)
    except Exception as exc:
        result.status = StageStatus.DEGRADED
        result.warn("alignment_failed")
        result.detail = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - t0
        return transcript, result

    # Rebuild the transcript with refined times, leaving unaligned words alone.
    by_position = {(si, wi): timing for si, wi, timing in refined}
    segments: list[Segment] = []
    moved: list[float] = []
    for si, seg in enumerate(transcript.segments):
        new_words = []
        for wi, w in enumerate(seg.words):
            timing = by_position.get((si, wi))
            if timing is None:
                new_words.append(w)
                continue
            start_ms, end_ms = timing
            moved.append(abs(start_ms - w.start_ms))
            new_words.append(replace(w, start_ms=start_ms, end_ms=end_ms))
        tup = tuple(new_words)
        segments.append(
            replace(
                seg,
                words=tup,
                start_ms=tup[0].start_ms if tup else seg.start_ms,
                end_ms=tup[-1].end_ms if tup else seg.end_ms,
            )
        )

    _ = torch
    aligned = replace(transcript, segments=segments)
    result.seconds = time.perf_counter() - t0
    median_shift = float(np.median(moved)) if moved else 0.0
    result.detail = (
        f"aligned {len(by_position)}/{len(words)} words, median shift {median_shift:.0f} ms"
    )
    if len(by_position) < len(words) * 0.8:
        result.warn("partial_alignment")
    return aligned, result


def _run_alignment(
    words: list[tuple[int, int, Word]],
    audio: np.ndarray,
    aligner: Aligner,
    sample_rate: int,
) -> list[tuple[int, int, tuple[int, int]]]:
    """CTC forced alignment over the whole utterance."""
    import torch
    from torchaudio import functional as ta_functional

    device = torch.device(aligner.device)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)

    with torch.inference_mode():
        emission, _ = aligner.model(waveform)  # type: ignore[operator]

    transcript_tokens = [_normalise(w.text) for _, _, w in words]
    # The tokenizer returns one list of token ids PER WORD. forced_align
    # wants a single flat target sequence, so flatten for the call and keep the
    # per-word lengths to regroup the spans afterwards.
    token_ids = aligner.tokenizer(transcript_tokens)  # type: ignore[operator]
    flat = [tok for word_tokens in token_ids for tok in word_tokens]
    if not flat:
        return []
    targets = torch.tensor([flat], dtype=torch.int32, device=device)

    alignments, scores = ta_functional.forced_align(emission, targets, blank=0)
    spans = ta_functional.merge_tokens(alignments[0], scores[0].exp())

    # Group token spans back into words, in order.
    ratio = waveform.shape[1] / emission.shape[1]
    out: list[tuple[int, int, tuple[int, int]]] = []
    cursor = 0
    for (si, wi, _), ids in zip(words, token_ids, strict=True):
        n = len(ids)
        if n == 0 or cursor + n > len(spans):
            cursor += n
            continue
        group = spans[cursor : cursor + n]
        cursor += n
        start = int(group[0].start * ratio / sample_rate * 1000)
        end = int(group[-1].end * ratio / sample_rate * 1000)
        if end > start:
            out.append((si, wi, (start, end)))
    return out


def timing_error_ms(reference: Transcript, hypothesis: Transcript) -> list[float]:
    """Per-word start-time differences, for measuring alignment quality.

    Used by the evaluation harness to report the word-timing median error that
    docs/08 tracks. Compares only words at matching positions, since a
    mismatched word count means the transcripts are not comparable.
    """
    errors: list[float] = []
    for ref_seg, hyp_seg in zip(reference.segments, hypothesis.segments, strict=False):
        for ref_w, hyp_w in zip(ref_seg.words, hyp_seg.words, strict=False):
            if _normalise(ref_w.text) == _normalise(hyp_w.text):
                errors.append(abs(ref_w.start_ms - hyp_w.start_ms))
    return errors
