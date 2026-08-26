"""S2B video-analysis tests.

The association logic is tested directly rather than through the detector,
because its failure mode is subtle: a track that gets retired during a brief
occlusion and restarted under a new id becomes a phantom extra speaker in S3,
and the symptom appears two stages later as speaker fragmentation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.s2b_video import (
    HIGH_CONF,
    LOW_CONF,
    Detection,
    FaceTrack,
    VideoAnalysis,
    _associate,
    iou,
    score_activity,
)


def box(x: float, y: float, w: float = 40, h: float = 40) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_iou_identical_boxes() -> None:
    assert iou(box(0, 0), box(0, 0)) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert iou(box(0, 0), box(100, 100)) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap() -> None:
    # 40x40 boxes offset by 20 in x: intersection 20x40, union 2400
    assert iou(box(0, 0), box(20, 0)) == pytest.approx(800 / 2400)


# --------------------------------------------------------------------------
# association
# --------------------------------------------------------------------------


def _run(frames: list[list[Detection]]) -> list[FaceTrack]:
    tracks: list[FaceTrack] = []
    active: dict[int, int] = {}
    next_id = 0
    for i, dets in enumerate(frames):
        next_id = _associate(tracks, active, dets, i, next_id)
        for tid, last in list(active.items()):
            if i - last > 12:
                del active[tid]
    return tracks


def test_a_moving_face_stays_one_track() -> None:
    frames = [[Detection(i, box(i * 2, 0), 0.9)] for i in range(10)]
    tracks = _run(frames)
    assert len(tracks) == 1
    assert tracks[0].n_frames == 10


def test_two_faces_stay_separate_tracks() -> None:
    frames = [[Detection(i, box(0, 0), 0.9), Detection(i, box(200, 200), 0.9)] for i in range(8)]
    tracks = _run(frames)
    assert len(tracks) == 2
    assert all(t.n_frames == 8 for t in tracks)


def test_low_confidence_detection_continues_a_track() -> None:
    """The reason ByteTrack has two stages: brief blur must not split a track."""
    frames: list[list[Detection]] = []
    for i in range(10):
        score = 0.30 if 4 <= i <= 6 else 0.9  # blurred in the middle
        assert LOW_CONF <= score < HIGH_CONF or score >= HIGH_CONF
        frames.append([Detection(i, box(i, 0), score)])
    tracks = _run(frames)
    assert len(tracks) == 1, "a low-confidence run split the track"
    assert tracks[0].n_frames == 10


def test_low_confidence_alone_does_not_start_a_track() -> None:
    """A noisy low-confidence box must not spawn a phantom speaker."""
    frames = [[Detection(i, box(0, 0), 0.25)] for i in range(6)]
    assert _run(frames) == []


def test_track_retires_after_a_long_gap() -> None:
    early = [[Detection(i, box(0, 0), 0.9)] for i in range(5)]
    gap: list[list[Detection]] = [[] for _ in range(20)]
    late = [[Detection(25 + i, box(0, 0), 0.9)] for i in range(5)]
    tracks = _run(early + gap + late)
    assert len(tracks) == 2, "a 20-frame absence should end the track"


# --------------------------------------------------------------------------
# activity scoring
# --------------------------------------------------------------------------


def test_activity_is_zero_for_a_static_face() -> None:
    n = 20
    still = np.full((100, 100, 3), 128, dtype=np.uint8)
    frames = dict.fromkeys(range(n), still)
    track = FaceTrack(0, [Detection(i, (10, 10, 50, 50), 0.9) for i in range(n)])
    scores = score_activity(track, frames, n)
    assert scores.max() == 0.0


def test_activity_responds_to_lip_region_motion() -> None:
    n = 20
    rng = np.random.default_rng(0)
    frames = {}
    for i in range(n):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        if i >= 10:
            # change only the lower third of the face box
            img[40:50, 10:50] = rng.integers(0, 255, (10, 40, 3), dtype=np.uint8)
        frames[i] = img
    track = FaceTrack(0, [Detection(i, (10, 10, 50, 50), 0.9) for i in range(n)])
    scores = score_activity(track, frames, n)
    assert scores[10:].max() > 0.5
    assert scores[:10].max() == 0.0


def test_activity_is_normalised() -> None:
    n = 10
    rng = np.random.default_rng(1)
    frames = {i: rng.integers(0, 255, (100, 100, 3), dtype=np.uint8) for i in range(n)}
    track = FaceTrack(0, [Detection(i, (10, 10, 50, 50), 0.9) for i in range(n)])
    scores = score_activity(track, frames, n)
    assert scores.min() >= 0.0 and scores.max() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def test_frame_to_sample_is_exact_at_25fps() -> None:
    """S0 forces CFR, so this mapping must be exact, not approximate."""
    va = VideoAnalysis(tracks=[], n_frames=250)
    assert va.frame_to_sample(0) == 0
    assert va.frame_to_sample(25) == 16_000  # one second
    assert va.frame_to_sample(250) == 160_000  # ten seconds
