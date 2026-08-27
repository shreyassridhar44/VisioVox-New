"""The caption serialisation contract (S7 -> player, docs/12 §5).

`Transcript.to_json` is read by `apps/web/src/lib/playback/captions.ts`, which
indexes segments by `start_ms` and bisects them. That parser skips any segment
missing `text`, `start_ms` or `end_ms`, and it does so silently — a renamed key
would not raise anywhere, it would just produce a recording with no captions.
These tests are the tripwire for that, and they are the reason the key names
are asserted literally rather than through a helper.

VTT is checked here too, but only as an export: it is what the download and the
HLS subtitle renditions carry, and it is deliberately not what the player reads.
"""

from __future__ import annotations

import itertools
import json

from pipeline.s7_transcribe import Segment, Transcript, Word


def _transcript() -> Transcript:
    first = (
        Word("Right", 0, 400, 0.98),
        Word("so", 400, 620, 0.41),
    )
    second = (Word("agreed", 2000, 2500, 0.93),)
    return Transcript(
        segments=[
            Segment("Right so", 0, 620, first, 0.01),
            Segment("agreed", 2000, 2500, second, 0.02),
        ],
        language="en",
        language_probability=0.97,
    )


def test_to_json_carries_the_keys_the_player_indexes_on() -> None:
    payload = _transcript().to_json()

    assert set(payload) >= {"language", "language_probability", "segments"}
    for segment in payload["segments"]:
        # The three the TypeScript parser requires. Anything else is additive.
        assert {"text", "start_ms", "end_ms"} <= set(segment)
        for word in segment["words"]:
            assert {"text", "start_ms", "end_ms", "probability"} <= set(word)


def test_timings_are_integer_milliseconds() -> None:
    """Invariant 2: never float seconds. The player multiplies by 1000 already."""
    payload = _transcript().to_json()
    for segment in payload["segments"]:
        assert isinstance(segment["start_ms"], int)
        assert isinstance(segment["end_ms"], int)
        for word in segment["words"]:
            assert isinstance(word["start_ms"], int)
            assert isinstance(word["end_ms"], int)


def test_segments_are_ordered_and_non_overlapping() -> None:
    """The player bisects, so out-of-order segments would silently mis-resolve.

    It sorts defensively on load, but emitting them in order means a caption
    bug can be localised to one side of the boundary rather than both.
    """
    segments = _transcript().to_json()["segments"]
    for earlier, later in itertools.pairwise(segments):
        assert earlier["end_ms"] <= later["start_ms"]
        assert earlier["start_ms"] < earlier["end_ms"]


def test_word_confidence_survives_serialisation() -> None:
    """Low-confidence words are dimmed rather than hidden, so the value has to
    reach the client intact — rounding it to a boolean would throw away the
    disclosure the product promises."""
    payload = _transcript().to_json()
    probabilities = [w["probability"] for s in payload["segments"] for w in s["words"]]
    assert probabilities == [0.98, 0.41, 0.93]


def test_payload_is_json_serialisable() -> None:
    """It is written to object storage as-is; a numpy float here would raise at
    upload time rather than in any test that only inspects the dict."""
    payload = _transcript().to_json()
    assert json.loads(json.dumps(payload)) == payload


def test_empty_transcript_round_trips() -> None:
    """A speaker who never speaks still gets a captions file, so the player has
    something to fetch and can distinguish 'silent' from 'failed'."""
    empty = Transcript(segments=[], language="en", language_probability=0.0)
    payload = empty.to_json()
    assert payload["segments"] == []
    assert json.loads(json.dumps(payload)) == payload


def test_vtt_remains_an_export_not_the_runtime_format() -> None:
    vtt = _transcript().to_vtt()
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:00.620" in vtt
    # Word timings cannot be expressed here, which is the whole reason the
    # player reads the JSON instead.
    assert "0.41" not in vtt
