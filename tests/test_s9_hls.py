"""HLS packaging tests (docs/12 §4).

The engine-choice thresholds get direct tests because the decision is made
server-side specifically so it is not duplicated in the client — and a rule
that lives in one place is only useful if that place is correct.

Playlist structure is checked as text rather than through a parser: what
matters is that hls.js can read it, and the tags it needs are exactly the ones
easy to omit by accident.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.s9_hls import (
    SEGMENT_SECONDS,
    AudioRendition,
    HlsError,
    SubtitleRendition,
    choose_engine,
    package_hls,
    segment_audio,
    write_master_playlist,
    write_subtitle_playlist,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
RATE = 16_000


def _wav(path: Path, seconds: float = 8.0, freq: float = 220.0) -> Path:
    t = np.arange(int(seconds * RATE)) / RATE
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), RATE)
    return path


# --------------------------------------------------------------------------
# engine choice
# --------------------------------------------------------------------------


def test_short_two_speaker_project_uses_webaudio() -> None:
    """Web Audio buys sample-accurate switching; use it whenever it is viable."""
    assert choose_engine(5 * 60 * 1000, 2) == "webaudio"


def test_long_project_switches_to_hls() -> None:
    assert choose_engine(20 * 60 * 1000, 2) == "hls"


def test_many_tracks_switch_to_hls() -> None:
    """Total bytes held in memory is duration times track count, so either
    dimension alone can force the switch."""
    assert choose_engine(5 * 60 * 1000, 6) == "hls"


@pytest.mark.parametrize(
    ("duration_min", "tracks", "expected"),
    [(10, 4, "webaudio"), (11, 4, "hls"), (10, 5, "hls"), (1, 1, "webaudio")],
)
def test_engine_thresholds(duration_min: int, tracks: int, expected: str) -> None:
    assert choose_engine(duration_min * 60 * 1000, tracks) == expected


# --------------------------------------------------------------------------
# segmenting
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_segmenting_produces_a_playlist_and_segments(tmp_path: Path) -> None:
    src = _wav(tmp_path / "spk1.wav", seconds=14.0)
    out = tmp_path / "audio"
    playlist = segment_audio(src, out, "a0")

    text = (out / playlist).read_text()
    assert text.startswith("#EXTM3U")
    assert "#EXT-X-ENDLIST" in text, "a VOD playlist must be terminated"
    assert "#EXT-X-PLAYLIST-TYPE:VOD" in text

    segments = sorted(out.glob("a0_*.ts"))
    assert len(segments) >= 2, "14 s should not fit in one 6 s segment"
    assert all(s.stat().st_size > 0 for s in segments)


@needs_ffmpeg
def test_segment_duration_is_respected(tmp_path: Path) -> None:
    src = _wav(tmp_path / "s.wav", seconds=20.0)
    out = tmp_path / "audio"
    segment_audio(src, out, "a0")
    text = (out / "a0.m3u8").read_text()
    assert f"#EXT-X-TARGETDURATION:{SEGMENT_SECONDS}" in text or "#EXT-X-TARGETDURATION:7" in text


def test_missing_source_fails_loudly(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    with pytest.raises(HlsError):
        segment_audio(tmp_path / "nope.wav", tmp_path / "out", "a0")


# --------------------------------------------------------------------------
# playlists
# --------------------------------------------------------------------------


def test_master_playlist_declares_every_rendition(tmp_path: Path) -> None:
    audio = [
        AudioRendition("Speaker 1", "aud", "audio/a0.m3u8", default=True, autoselect=True),
        AudioRendition("Speaker 2", "aud", "audio/a1.m3u8"),
    ]
    subs = [SubtitleRendition("Speaker 1", "subs", "subs/s0.m3u8")]
    text = write_master_playlist(tmp_path, audio, subs).read_text()

    assert text.startswith("#EXTM3U")
    assert text.count("#EXT-X-MEDIA:TYPE=AUDIO") == 2
    assert text.count("#EXT-X-MEDIA:TYPE=SUBTITLES") == 1
    assert 'GROUP-ID="aud"' in text
    assert "#EXT-X-STREAM-INF:" in text, "without a variant line no player will start"


def test_exactly_one_audio_rendition_is_default(tmp_path: Path) -> None:
    """Two defaults is ambiguous and players disagree on how to resolve it."""
    audio = [
        AudioRendition("Speaker 1", "aud", "audio/a0.m3u8", default=True, autoselect=True),
        AudioRendition("Speaker 2", "aud", "audio/a1.m3u8"),
        AudioRendition("Speaker 3", "aud", "audio/a2.m3u8"),
    ]
    text = write_master_playlist(tmp_path, audio, []).read_text()
    assert text.count("DEFAULT=YES") == 1


def test_master_playlist_needs_audio(tmp_path: Path) -> None:
    with pytest.raises(HlsError, match="at least one audio"):
        write_master_playlist(tmp_path, [], [])


def test_subtitle_playlist_references_the_vtt(tmp_path: Path) -> None:
    name = write_subtitle_playlist("spk_1.vtt", tmp_path, "s0", duration_ms=90_000)
    text = (tmp_path / name).read_text()
    assert "spk_1.vtt" in text
    assert "#EXT-X-TARGETDURATION:90" in text
    assert "#EXT-X-ENDLIST" in text


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_package_hls_builds_a_complete_tree(tmp_path: Path) -> None:
    a = _wav(tmp_path / "spk1.wav", seconds=10.0, freq=220)
    b = _wav(tmp_path / "spk2.wav", seconds=10.0, freq=330)
    (tmp_path / "spk_1.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
    (tmp_path / "spk_2.vtt").write_text("WEBVTT\n\n", encoding="utf-8")

    result = package_hls(
        tmp_path,
        tracks=[("Speaker 1", a), ("Speaker 2", b)],
        captions=[("Speaker 1", "spk_1.vtt"), ("Speaker 2", "spk_2.vtt")],
        duration_ms=10_000,
    )

    assert result.master.exists()
    assert len(result.audio) == 2
    assert len(result.subtitles) == 2
    assert result.segment_count >= 2

    master = result.master.read_text()
    for rendition in result.audio:
        assert rendition.playlist in master
    for sub in result.subtitles:
        assert sub.playlist in master

    # every referenced playlist must actually exist, or the player 404s
    for rel in [r.playlist for r in result.audio] + [s.playlist for s in result.subtitles]:
        assert (tmp_path / rel).exists(), f"master references missing {rel}"


@needs_ffmpeg
def test_audio_only_project_still_produces_a_variant(tmp_path: Path) -> None:
    a = _wav(tmp_path / "spk1.wav", seconds=8.0)
    result = package_hls(tmp_path, tracks=[("Speaker 1", a)], duration_ms=8000)
    master = result.master.read_text()
    assert "#EXT-X-STREAM-INF:" in master
    assert result.audio[0].playlist in master
