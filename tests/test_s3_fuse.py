"""S3 fusion tests.

Two behaviours are worth pinning because both fail silently:

- A constantly-moving face correlates with every speaker. If the score does not
  punish that, the loudest speaker claims it and everyone else is misbound.
- Greedy matching lets one speaker take the best face and pushes the rest onto
  leftovers. The binding conditions the extractor, so a swap here is not
  recoverable later — it just produces the wrong person's voice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.s2a_audio import AudioAnalysis, SpeakerTurn
from pipeline.s2b_video import Detection, FaceTrack, VideoAnalysis
from pipeline.s3_fuse import agreement_score, fuse
from pipeline.types import FRAME_RATE, Interval, Modality, StageStatus

RATE = 16_000
N_FRAMES = 250  # 10 s at 25 fps


def _turn(speaker: str, a: float, b: float) -> SpeakerTurn:
    return SpeakerTurn(speaker, Interval(int(a * RATE), int(b * RATE)))


def _audio(turns: list[SpeakerTurn], seconds: float = 10.0) -> AudioAnalysis:
    return AudioAnalysis(
        speech=[Interval(0, int(seconds * RATE))],
        turns=turns,
        overlap=[],
        speakers=sorted({t.speaker for t in turns}),
        total_samples=int(seconds * RATE),
    )


def _track(track_id: int, active_windows: list[tuple[float, float]]) -> FaceTrack:
    activity = np.zeros(N_FRAMES, dtype=np.float32)
    for a, b in active_windows:
        activity[int(a * FRAME_RATE) : int(b * FRAME_RATE)] = 1.0
    dets = [Detection(i, (0, 0, 40, 40), 0.9) for i in range(N_FRAMES)]
    t = FaceTrack(track_id, dets)
    t.activity = activity
    return t


# --------------------------------------------------------------------------
# agreement score
# --------------------------------------------------------------------------


def test_perfect_agreement_scores_high() -> None:
    voice = np.zeros(N_FRAMES, dtype=bool)
    voice[:125] = True
    activity = np.zeros(N_FRAMES, dtype=np.float32)
    activity[:125] = 1.0
    assert agreement_score(voice, activity) == pytest.approx(0.5, abs=1e-6)


def test_constantly_moving_face_scores_near_zero() -> None:
    """The failure this metric exists to prevent."""
    voice = np.zeros(N_FRAMES, dtype=bool)
    voice[:125] = True
    always_moving = np.ones(N_FRAMES, dtype=np.float32)
    assert abs(agreement_score(voice, always_moving)) < 0.01


def test_anticorrelated_face_scores_negative() -> None:
    voice = np.zeros(N_FRAMES, dtype=bool)
    voice[:125] = True
    activity = np.zeros(N_FRAMES, dtype=np.float32)
    activity[125:] = 1.0
    assert agreement_score(voice, activity) < -0.4


def test_score_is_zero_when_a_speaker_never_stops() -> None:
    voice = np.ones(N_FRAMES, dtype=bool)
    assert agreement_score(voice, np.ones(N_FRAMES, dtype=np.float32)) == 0.0


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------


def test_binds_each_speaker_to_its_own_face() -> None:
    audio = _audio([_turn("A", 0, 5), _turn("B", 5, 10)])
    video = VideoAnalysis([_track(10, [(0, 5)]), _track(20, [(5, 10)])], N_FRAMES)
    registry, result = fuse(audio, video)

    assert result.status is StageStatus.OK
    a = registry.by_speaker("A")
    b = registry.by_speaker("B")
    assert a is not None and b is not None
    assert a.face_track_id == 10
    assert b.face_track_id == 20
    assert a.modality is Modality.AUDIOVISUAL


def test_global_assignment_avoids_the_greedy_swap() -> None:
    """A dominant speaker must not be able to claim a quieter speaker's face.

    Speaker A talks most of the clip; its face also moves a little during B's
    turn. Greedy matching on best-score-first can hand A the wrong face and
    leave B with the leftover. Hungarian maximises the total instead.
    """
    audio = _audio([_turn("A", 0, 7), _turn("B", 7, 10)])
    face_a = _track(1, [(0, 7)])
    face_b = _track(2, [(7, 10)])
    # give A's face some spurious motion during B's turn
    assert face_a.activity is not None
    face_a.activity[int(7 * FRAME_RATE) : int(8 * FRAME_RATE)] = 1.0

    registry, _ = fuse(audio, VideoAnalysis([face_a, face_b], N_FRAMES))
    a = registry.by_speaker("A")
    b = registry.by_speaker("B")
    assert a is not None and b is not None
    assert a.face_track_id == 1, "dominant speaker took the wrong face"
    assert b.face_track_id == 2, "quiet speaker was left with the leftover"


def test_weak_agreement_falls_back_to_audio_only() -> None:
    """A wrong face is worse than no face — it feeds the visual pathway a lie."""
    audio = _audio([_turn("A", 0, 5), _turn("B", 5, 10)])
    noise = _track(1, [])
    assert noise.activity is not None
    noise.activity[:] = 0.0
    registry, result = fuse(audio, VideoAnalysis([noise], N_FRAMES))

    a = registry.by_speaker("A")
    assert a is not None
    assert a.modality is Modality.AUDIO_ONLY
    assert a.face_track_id is None
    assert any("no_face_track" in w for w in result.warnings)


def test_no_video_gives_all_audio_only() -> None:
    audio = _audio([_turn("A", 0, 5), _turn("B", 5, 10)])
    registry, result = fuse(audio, None)
    assert result.status is StageStatus.DEGRADED
    assert "no_face_tracks_to_bind" in result.warnings
    assert all(b.modality is Modality.AUDIO_ONLY for b in registry.bindings)
    assert len(registry.bindings) == 2


def test_ordinals_follow_speaking_time() -> None:
    """Speaker 1 should be the person who talks most, not an arbitrary label."""
    audio = _audio([_turn("Q", 0, 2), _turn("Z", 2, 10)])
    registry, _ = fuse(audio, None)
    z = registry.by_speaker("Z")
    q = registry.by_speaker("Q")
    assert z is not None and q is not None
    assert z.ordinal == 1 and q.ordinal == 2
    assert z.label == "Speaker 1"


def test_no_speakers_degrades_cleanly() -> None:
    registry, result = fuse(_audio([]), None)
    assert result.status is StageStatus.DEGRADED
    assert "no_speakers_to_bind" in result.warnings
    assert registry.bindings == []
