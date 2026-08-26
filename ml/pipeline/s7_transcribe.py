"""S7 — Transcription and word alignment (docs/05 §10).

Runs per speaker stream, on the **Faithful** track (invariant 6). That is not
an implementation detail: the Natural track may have been through generative
restoration, which can invent words, and captions must reflect what was
actually recovered rather than what sounded best. Transcribing the wrong track
undermines the whole hallucination-safety design at its source.

All timings are integer milliseconds at the API boundary (invariant 2).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .types import StageResult, StageStatus

STAGE = "S7_transcribe"
VERSION = "1.0.0"

DEFAULT_MODEL = "large-v3"


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int
    probability: float


@dataclass(frozen=True)
class Segment:
    text: str
    start_ms: int
    end_ms: int
    words: tuple[Word, ...]
    no_speech_prob: float

    @property
    def mean_word_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(np.mean([w.probability for w in self.words]))


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    language_probability: float

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def to_vtt(self) -> str:
        """WebVTT for the player. Timestamps derive from integer ms, not floats."""
        lines = ["WEBVTT", ""]
        for i, seg in enumerate(self.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_vtt_time(seg.start_ms)} --> {_vtt_time(seg.end_ms)}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)


def _vtt_time(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


class Transcriber(Protocol):
    """The slice of faster-whisper we use, so strict typing has something real.

    faster-whisper ships no stubs, so the element types stay Any; the shape of
    the call is still checked.
    """

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


def load_transcriber(
    model_size: str = DEFAULT_MODEL,
    device: str = "cuda",
    compute_type: str = "float16",
    cache: Path | None = None,
) -> Transcriber:
    from faster_whisper import WhisperModel

    model: Transcriber = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(cache) if cache else None,
    )
    return model


def transcribe(
    audio: np.ndarray,
    model: Transcriber,
    language: str | None = None,
    beam_size: int = 5,
) -> tuple[Transcript, StageResult]:
    """Transcribe one speaker stream with word-level timings."""
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)

    raw_segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=False,  # S2A already decided where speech is
    )

    segments: list[Segment] = []
    for seg in raw_segments:
        words = tuple(
            Word(
                text=w.word,
                start_ms=round(w.start * 1000),
                end_ms=round(w.end * 1000),
                probability=float(w.probability),
            )
            for w in (seg.words or [])
        )
        segments.append(
            Segment(
                text=seg.text,
                start_ms=round(seg.start * 1000),
                end_ms=round(seg.end * 1000),
                words=words,
                no_speech_prob=float(seg.no_speech_prob),
            )
        )

    transcript = Transcript(
        segments=segments,
        language=str(info.language),
        language_probability=float(info.language_probability),
    )

    if not segments:
        result.status = StageStatus.DEGRADED
        result.warn("empty_transcript")
    elif transcript.language_probability < 0.5:
        # Worth surfacing: a low-confidence language guess usually means the
        # stream is mostly noise or the extraction failed for this speaker.
        result.warn("low_language_confidence")

    result.seconds = time.perf_counter() - t0
    result.detail = f"{len(segments)} segments, lang={transcript.language}"
    return transcript, result
