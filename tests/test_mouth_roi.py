"""Mouth ROI extraction: geometry, gaps, and the gate signal.

The geometry tests matter more than they look. A visual frontend is a function
of pixel framing, so a crop that drifts from the one the corpus was packed with
does not degrade gracefully -- it feeds the model plausible-looking rubbish,
and nothing downstream can detect that. These tests pin the framing to actual
numbers rather than to "it returns the right shape".
"""

from __future__ import annotations

import numpy as np

from pipeline.mouth_roi import (
    DEFAULT_MARGIN,
    MAX_HOLD_FRAMES,
    MOUTH,
    ROI_HEIGHT,
    ROI_WIDTH,
    ROI_X,
    ROI_Y,
    crop_roi,
    ffmpeg_filter,
    rois_from_frames,
)
from pipeline.s2b_video import Detection, FaceTrack

# The exact string scripts/prepare_voxceleb2.py used when the corpus on disk
# was packed. The packed clips cannot be re-cut cheaply, so this is a fixed
# point: if it has to change, the corpus has to be repacked with it.
PACKED_WITH = "fps=25,crop=iw*0.6:ih*0.36:iw*0.2:ih*0.56,scale=96:96,format=gray"


def test_the_filter_still_matches_the_corpus_on_disk() -> None:
    assert ffmpeg_filter(25, 96) == PACKED_WITH


def test_training_and_inference_share_one_geometry() -> None:
    """Both paths must read the same four constants, not two copies of them."""
    assert f"crop=iw*{ROI_WIDTH}:ih*{ROI_HEIGHT}:iw*{ROI_X}:ih*{ROI_Y}" in ffmpeg_filter()


# --------------------------------------------------------------------------
# crop geometry
# --------------------------------------------------------------------------


def _frame(height: int = 400, width: int = 400) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_the_crop_lands_on_the_expected_rectangle() -> None:
    """Worked by hand so the constants cannot be changed silently.

    box (100,100)-(300,300) is 200x200. A 0.25 margin grows it to 250x250
    anchored at (75,75), and the mouth sub-rectangle is then x 20%..80% and
    y 56%..92% of that: (125,215)-(275,305).
    """
    frame = _frame()
    frame[215:305, 125:275] = 255

    roi = crop_roi(frame, (100.0, 100.0, 300.0, 300.0), margin=DEFAULT_MARGIN)

    assert roi.shape == (MOUTH, MOUTH)
    assert roi.dtype == np.uint8
    assert roi.min() == 255, "the ROI strayed outside the painted mouth rectangle"


def test_a_face_at_the_frame_edge_is_clamped_not_crashed() -> None:
    frame = _frame()
    roi = crop_roi(frame, (-50.0, 320.0, 60.0, 460.0))
    assert roi.shape == (MOUTH, MOUTH)


def test_a_degenerate_box_gives_a_blank_roi() -> None:
    for bbox in [(10.0, 10.0, 10.0, 10.0), (200.0, 200.0, 100.0, 100.0)]:
        roi = crop_roi(_frame(), bbox)
        assert roi.shape == (MOUTH, MOUTH)
        assert roi.max() == 0


def test_a_larger_margin_takes_in_more_of_the_frame() -> None:
    """The margin is a calibration knob; it has to actually do something."""
    frame = _frame()
    frame[215:305, 125:275] = 255
    tight = crop_roi(frame, (100.0, 100.0, 300.0, 300.0), margin=0.0)
    loose = crop_roi(frame, (100.0, 100.0, 300.0, 300.0), margin=0.8)
    assert not np.array_equal(tight, loose)


# --------------------------------------------------------------------------
# tracks, gaps and confidence
# --------------------------------------------------------------------------


def _track(frames: list[int], score: float = 0.9) -> FaceTrack:
    return FaceTrack(
        track_id=1,
        detections=[
            Detection(frame=f, bbox=(100.0, 100.0, 300.0, 300.0), score=score) for f in frames
        ],
    )


def _decoded(n: int) -> dict[int, np.ndarray]:
    frame = _frame()
    frame[215:305, 125:275] = 255
    return dict.fromkeys(range(n), frame)


def test_every_frame_of_the_timeline_gets_a_roi() -> None:
    """S5 indexes ROIs by frame, so the array must span the whole clip."""
    rois = rois_from_frames(_decoded(50), _track(list(range(0, 50, 2))), 50)
    assert rois.frames.shape == (50, MOUTH, MOUTH)
    assert rois.confidence.shape == (50,)
    assert len(rois) == 50


def test_detected_frames_carry_the_detector_score_as_confidence() -> None:
    rois = rois_from_frames(_decoded(10), _track(list(range(10)), score=0.7), 10)
    assert np.allclose(rois.confidence, 0.7)
    assert rois.coverage == 1.0


def test_a_short_dropout_is_held_with_decaying_confidence() -> None:
    """A blink is not evidence the face has gone; a second of stale pixels is."""
    detected = [0, 1, 2]
    rois = rois_from_frames(_decoded(12), _track(detected), 12)

    held = rois.confidence[3 : 3 + MAX_HOLD_FRAMES]
    assert np.all(held > 0.0), "short gaps should stay usable"
    assert np.all(np.diff(held) < 0), "trust in held pixels must decay"
    # Held frames reuse the last real ROI rather than going black.
    assert rois.frames[3].max() > 0


def test_a_long_dropout_closes_the_gate_and_blanks_the_roi() -> None:
    rois = rois_from_frames(_decoded(30), _track([0, 1, 2]), 30)
    assert rois.confidence[3 + MAX_HOLD_FRAMES :].max() == 0.0
    assert rois.frames[-1].max() == 0


def test_a_track_with_no_detections_yields_no_confidence_anywhere() -> None:
    """Equivalent to having no face at all, which is the audio-only path."""
    rois = rois_from_frames(_decoded(20), _track([]), 20)
    assert rois.coverage == 0.0
    assert rois.frames.max() == 0


def test_frames_missing_from_the_decode_are_treated_as_dropouts() -> None:
    """Decoding can end early; the ROI array still has to span the timeline."""
    rois = rois_from_frames(_decoded(5), _track(list(range(20))), 20)
    assert len(rois) == 20
    assert rois.confidence[:5].min() > 0.0
    assert rois.confidence[-1] == 0.0
