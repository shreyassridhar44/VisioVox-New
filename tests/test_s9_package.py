"""S9 packaging tests.

The important one is that S9 output validates against the *frozen* schema in
packages/contracts. That schema is the contract the application track is being
built against while the model is still changing; if the packager and the schema
drift apart, the parallel-track plan quietly stops working and nobody finds out
until integration.

Invariant 1 gets a test of its own because its failure mode is invisible: a
one-sample mismatch produces accumulating A/V drift that only becomes obvious
deep into a long recording.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from pipeline.s7_transcribe import Segment, Transcript, Word
from pipeline.s9_package import PackagingError, SpeakerTrack, package
from pipeline.types import ANALYSIS_SAMPLE_RATE, Modality, StageStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "packages" / "contracts" / "schemas" / "manifest.schema.json"
RATE = ANALYSIS_SAMPLE_RATE

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")

PROJECT_ID = "prj_01HX8ZQ3M7N4P5R6S7T8V9W0XY"
SIGNED_UNTIL = "2026-08-26T12:45:00Z"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _tone(seconds: float, freq: float, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    return (0.2 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def _transcript() -> Transcript:
    words = (
        Word("hello", 100, 500, 0.94),
        Word("there", 520, 900, 0.91),
    )
    return Transcript(
        segments=[Segment("hello there", 100, 900, words, 0.01)],
        language="en",
        language_probability=0.99,
    )


def _tracks(n: int = 2, seconds: float = 3.0) -> list[SpeakerTrack]:
    return [
        SpeakerTrack(
            speaker_id=f"spk_01HX8ZQ3M7N4P5R6S7T8V9W0X{i}",
            ordinal=i + 1,
            label=f"Speaker {i + 1}",
            faithful=_tone(seconds, 220 * (i + 1), seed=i),
            modality=Modality.AUDIOVISUAL,
            speaking_ratio=0.4,
            mean_confidence=0.85,
            extraction_ok=True,
            transcript=_transcript(),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# invariant 1
# --------------------------------------------------------------------------


def test_rejects_mismatched_track_lengths(tmp_path: Path) -> None:
    """Invariant 1: differing sample counts must fail loudly, not drift silently."""
    tracks = _tracks()
    tracks[1].faithful = tracks[1].faithful[:-1]  # one sample short
    with pytest.raises(PackagingError, match="invariant 1"):
        package(
            PROJECT_ID,
            tracks,
            tmp_path,
            has_video=False,
            signed_until=SIGNED_UNTIL,
        )


def test_rejects_mismatched_natural_track(tmp_path: Path) -> None:
    tracks = _tracks()
    tracks[0].natural = tracks[0].faithful[:-100]
    with pytest.raises(PackagingError, match="invariant 1"):
        package(PROJECT_ID, tracks, tmp_path, has_video=False, signed_until=SIGNED_UNTIL)


# --------------------------------------------------------------------------
# manifest conformance
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_manifest_validates_against_the_frozen_schema(
    tmp_path: Path, validator: Draft202012Validator
) -> None:
    manifest, result = package(
        PROJECT_ID,
        _tracks(),
        tmp_path,
        has_video=True,
        video_url="https://cdn.example/video.mp4?sig=x",
        video_size=(1920, 1080),
        overlap_ratio=0.17,
        signed_until=SIGNED_UNTIL,
        base_url="https://cdn.example/",
    )
    errors = sorted(validator.iter_errors(manifest), key=lambda e: e.json_path)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)
    assert result.status in (StageStatus.OK, StageStatus.DEGRADED)


@needs_ffmpeg
def test_duration_is_integer_ms_from_sample_count(tmp_path: Path) -> None:
    """Invariant 2: derived from samples, never float seconds."""
    manifest, _ = package(
        PROJECT_ID,
        _tracks(seconds=2.5),
        tmp_path,
        has_video=False,
        signed_until=SIGNED_UNTIL,
    )
    assert isinstance(manifest["duration_ms"], int)
    assert manifest["duration_ms"] == 2500


@needs_ffmpeg
def test_audio_only_project_validates(tmp_path: Path, validator: Draft202012Validator) -> None:
    tracks = _tracks()
    for t in tracks:
        t.modality = Modality.AUDIO_ONLY
    manifest, result = package(
        PROJECT_ID,
        tracks,
        tmp_path,
        has_video=False,
        signed_until=SIGNED_UNTIL,
        base_url="https://cdn.example/",
    )
    assert "video" not in manifest
    assert not list(validator.iter_errors(manifest))
    assert any("no_face_track" in w for w in result.warnings)


@needs_ffmpeg
def test_failed_speaker_still_produces_a_valid_manifest(
    tmp_path: Path, validator: Draft202012Validator
) -> None:
    """Invariant 8: 2 of 3 speakers beats nothing."""
    tracks = _tracks(n=3)
    tracks[2].extraction_ok = False
    tracks[2].transcript = None
    manifest, result = package(
        PROJECT_ID,
        tracks,
        tmp_path,
        has_video=False,
        signed_until=SIGNED_UNTIL,
        base_url="https://cdn.example/",
    )
    assert not list(validator.iter_errors(manifest))
    assert result.status is StageStatus.DEGRADED
    assert manifest["speakers"][2]["extraction_ok"] is False
    assert "speaker_3_extraction_failed" in manifest["warnings"]


@needs_ffmpeg
def test_captions_are_written_and_referenced(tmp_path: Path) -> None:
    manifest, _ = package(
        PROJECT_ID, _tracks(), tmp_path, has_video=False, signed_until=SIGNED_UNTIL
    )
    vtt = tmp_path / "spk_1.vtt"
    assert vtt.exists()
    body = vtt.read_text()
    assert body.startswith("WEBVTT")
    assert "00:00:00.100 --> 00:00:00.900" in body, body
    assert manifest["speakers"][0]["captions"]["vtt"].endswith("spk_1.vtt")


def test_vtt_timestamps_come_from_integer_ms() -> None:
    t = Transcript(
        segments=[Segment("x", 3_661_001, 3_661_500, (), 0.0)],
        language="en",
        language_probability=1.0,
    )
    assert "01:01:01.001 --> 01:01:01.500" in t.to_vtt()
