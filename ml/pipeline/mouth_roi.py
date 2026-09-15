"""Mouth ROIs for a tracked face — the inference half of the visual pathway.

The visual frontend was trained on VoxCeleb2 clips, which are already tight,
face-centred crops; `scripts/prepare_voxceleb2.py` therefore takes a *fixed*
lower-centre box out of each frame rather than running a detector. A real
upload is not a face crop, so this module has to reproduce that same geometry
relative to each detected face box instead.

Reproducing it exactly is the whole job. A visual frontend is a function of
pixel geometry: if inference feeds it mouths framed differently from the ones
it trained on, the features are not wrong in some graceful way, they are
meaningless, and the reliability gate will not notice because the input still
looks like a mouth. So the geometry constants live here, once, and the packer
imports them — the two cannot drift apart without a test failing.

**One honest caveat.** A VoxCeleb2 frame and a face detector's bounding box are
not the same rectangle: VoxCeleb2 crops carry margin around the face, a bbox is
tight to it. `DEFAULT_MARGIN` expands the box to approximate the former, but
the value is an estimate rather than a measurement. It is a single scalar and
`scripts/eval_overlap.py` can sweep it against real SI-SDR once a trained
audio-visual checkpoint exists; until that sweep has been run, treat it as a
calibration knob that has not yet been calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .s2b_video import FaceTrack
from .types import FRAME_RATE

MOUTH = 96  # output ROI, pixels square

# The sub-rectangle of a face-centred frame that the mouth occupies, as
# fractions of width and height. These are the numbers baked into the training
# data; changing one without re-packing the corpus invalidates the frontend.
ROI_X, ROI_WIDTH = 0.20, 0.60
ROI_Y, ROI_HEIGHT = 0.56, 0.36

# How much to grow a tight detector box to look like a VoxCeleb2 crop.
DEFAULT_MARGIN = 0.25
# Frames a ROI may be held across a detection dropout before the gate is told
# to stop trusting it. Five frames is 200 ms, the same figure the router uses
# for the shortest audible state change.
MAX_HOLD_FRAMES = 5


def ffmpeg_filter(fps: int = FRAME_RATE, size: int = MOUTH) -> str:
    """The `-vf` string the packer uses, built from the constants above.

    Exists so that the training-time crop and the inference-time crop are the
    same definition rather than two copies that happen to agree today.
    """
    return (
        f"fps={fps},"
        f"crop=iw*{ROI_WIDTH}:ih*{ROI_HEIGHT}:iw*{ROI_X}:ih*{ROI_Y},"
        f"scale={size}:{size},format=gray"
    )


@dataclass
class MouthRois:
    """Per-frame mouth crops plus how much each one should be trusted.

    `confidence` feeds SEAVE's per-frame visual gate directly. That is the
    point of having it: a head turn, an occlusion or a tracker dropout should
    close the gate for exactly those frames and reopen it afterwards, which is
    what beta_t was designed for (docs/04 §3). Passing a constant 1.0 would
    make the gate decorative.
    """

    frames: np.ndarray  # (n_frames, MOUTH, MOUTH) uint8
    confidence: np.ndarray  # (n_frames,) float32 in 0..1

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def coverage(self) -> float:
        """Share of frames backed by a real detection rather than a hold."""
        if len(self) == 0:
            return 0.0
        return float(np.mean(self.confidence > 0.0))


def crop_roi(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    margin: float = DEFAULT_MARGIN,
    size: int = MOUTH,
) -> np.ndarray:
    """One frame plus one face box to one grayscale mouth ROI.

    Mirrors the packer: expand the box, take the fixed lower-centre
    sub-rectangle of it, greyscale, resize.
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return np.zeros((size, size), dtype=np.uint8)

    # Grow the tight detector box towards a VoxCeleb2-style crop.
    x1 -= box_w * margin / 2
    y1 -= box_h * margin / 2
    box_w *= 1 + margin
    box_h *= 1 + margin

    mx1 = round(x1 + box_w * ROI_X)
    my1 = round(y1 + box_h * ROI_Y)
    mx2 = round(mx1 + box_w * ROI_WIDTH)
    my2 = round(my1 + box_h * ROI_HEIGHT)

    # Clamp into the frame. A face at the edge gives a smaller ROI rather than
    # an exception; the resize below restores the expected shape.
    mx1, my1 = max(0, mx1), max(0, my1)
    mx2, my2 = min(width, mx2), min(height, my2)
    if mx2 - mx1 < 2 or my2 - my1 < 2:
        return np.zeros((size, size), dtype=np.uint8)

    patch = frame[my1:my2, mx1:mx2]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    resized: np.ndarray = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def rois_from_frames(
    frames: dict[int, np.ndarray],
    track: FaceTrack,
    n_frames: int,
    *,
    margin: float = DEFAULT_MARGIN,
    max_hold: int = MAX_HOLD_FRAMES,
) -> MouthRois:
    """Build the ROI sequence for one track from already-decoded frames.

    Takes decoded frames rather than a path so the caller can decode once and
    serve every speaker from it — a four-speaker meeting should not be read off
    disk four times.

    Detection gaps are held rather than blanked, with the confidence decaying
    over `max_hold` frames. A tracker that drops one frame in a blink is not
    evidence that the face has gone, but a second of held pixels is stale and
    the gate should say so.
    """
    out = np.zeros((n_frames, MOUTH, MOUTH), dtype=np.uint8)
    confidence = np.zeros(n_frames, dtype=np.float32)

    by_frame = {d.frame: d for d in track.detections}
    last_roi: np.ndarray | None = None
    held = 0

    for index in range(n_frames):
        detection = by_frame.get(index)
        image = frames.get(index)
        if detection is not None and image is not None:
            roi = crop_roi(image, detection.bbox, margin=margin)
            out[index] = roi
            # The detector's own score scales the gate: a marginal detection is
            # a marginal mouth, and the model should weigh it accordingly.
            confidence[index] = float(min(1.0, max(0.0, detection.score)))
            last_roi, held = roi, 0
            continue

        held += 1
        if last_roi is not None and held <= max_hold:
            out[index] = last_roi
            confidence[index] = float(max(0.0, 1.0 - held / (max_hold + 1)))
        # Beyond the hold window the frame stays black with zero confidence,
        # which is the same state as "no face at all" — deliberately, because
        # that is what it has become.

    return MouthRois(frames=out, confidence=confidence)


def decode_frames(video_path: Path, n_frames: int) -> dict[int, np.ndarray]:
    """Read frames into memory, indexed the way S2B indexes them.

    S2B reads sequentially and numbers frames from zero, treating that index as
    the 25 fps grid the rest of the pipeline uses; S0 is what makes that true by
    normalising the frame rate on ingest. Indexing here has to match, or every
    ROI is offset against the audio it is supposed to explain.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    index = 0
    while index < n_frames:
        ok, image = capture.read()
        if not ok:
            break
        frames[index] = image
        index += 1
    capture.release()
    return frames
