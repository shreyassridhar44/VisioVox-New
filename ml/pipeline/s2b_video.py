"""S2B — Video analysis: detect, track, active-speaker score (docs/05 §5).

Produces face tracks with a per-frame speaking score, which S3 binds to voice
clusters.

Frame indices are integers at a fixed 25 fps (S0 forces CFR), so a frame index
maps to a sample offset exactly. Nothing here uses float seconds.

**Active-speaker detection is a Phase 1 placeholder.** docs/05 specifies
Light-ASD; this uses lip-region motion energy instead — an algorithm with no
weights, which is honest about being weaker. It is adequate for the Tier 0
baseline, where the point is a working end-to-end pipeline and an honest
"before" measurement, and it emits `asd_is_motion_baseline` so no downstream
consumer mistakes it for the real thing. Swapping in Light-ASD changes only
`score_activity`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import FRAME_RATE, StageResult, StageStatus

STAGE = "S2B_video"
VERSION = "1.0.0"

DETECTOR_BUNDLE = "buffalo_sc"
DET_SIZE = (320, 320)

# ByteTrack-style two-stage association thresholds.
HIGH_CONF = 0.55
LOW_CONF = 0.20
IOU_MATCH = 0.30
MAX_MISSES = 12  # ~0.5 s at 25 fps before a track is retired


@dataclass
class Detection:
    frame: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float


@dataclass
class FaceTrack:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    activity: np.ndarray | None = None  # per-frame speaking score, 0..1

    @property
    def first_frame(self) -> int:
        return self.detections[0].frame

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame

    @property
    def n_frames(self) -> int:
        return len(self.detections)

    def bbox_at(self, frame: int) -> tuple[float, float, float, float] | None:
        for d in self.detections:
            if d.frame == frame:
                return d.bbox
        return None


@dataclass
class VideoAnalysis:
    tracks: list[FaceTrack]
    n_frames: int
    frame_rate: int = FRAME_RATE

    def frame_to_sample(self, frame: int, sample_rate: int = 16_000) -> int:
        """Exact because S0 forces constant frame rate."""
        return frame * sample_rate // self.frame_rate


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _associate(
    tracks: list[FaceTrack],
    active: dict[int, int],
    dets: list[Detection],
    frame: int,
    next_id: int,
) -> int:
    """One ByteTrack association pass: high-confidence first, then low.

    Two stages matter because a briefly occluded or motion-blurred face drops
    below the high threshold for a few frames. Matching those low-confidence
    boxes to existing tracks keeps identity, instead of retiring the track and
    starting a new one — which is what produces speaker fragmentation later.
    """
    high = [d for d in dets if d.score >= HIGH_CONF]
    low = [d for d in dets if LOW_CONF <= d.score < HIGH_CONF]

    for pool in (high, low):
        for det in pool:
            best_track, best_iou = None, IOU_MATCH
            for track in tracks:
                if track.track_id not in active:
                    continue
                last = track.detections[-1]
                if last.frame == frame:
                    continue
                score = iou(last.bbox, det.bbox)
                if score > best_iou:
                    best_track, best_iou = track, score
            if best_track is not None:
                best_track.detections.append(det)
                active[best_track.track_id] = frame
            elif pool is high:
                # Only high-confidence detections may start a track; a noisy
                # low-confidence box should not spawn a phantom speaker.
                track = FaceTrack(track_id=next_id, detections=[det])
                tracks.append(track)
                active[next_id] = frame
                next_id += 1
    return next_id


def score_activity(track: FaceTrack, frames: dict[int, np.ndarray], n_frames: int) -> np.ndarray:
    """Lip-region motion energy per frame, normalised to 0..1.

    Placeholder for Light-ASD (see module docstring). Motion in the lower third
    of the face box correlates with speaking; it is fooled by head movement and
    by chewing, which is exactly why the real model exists.
    """
    scores = np.zeros(n_frames, dtype=np.float32)
    previous: np.ndarray | None = None
    for det in track.detections:
        frame_img = frames.get(det.frame)
        if frame_img is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        h = y2 - y1
        # lower third of the face box, clipped to the image
        my1 = max(0, y1 + (2 * h) // 3)
        crop = frame_img[my1:y2, max(0, x1) : x2]
        if crop.size == 0:
            continue
        grey = crop.mean(axis=2) if crop.ndim == 3 else crop
        small = grey[:: max(1, grey.shape[0] // 16), :: max(1, grey.shape[1] // 16)]
        if previous is not None and previous.shape == small.shape:
            scores[det.frame] = float(np.abs(small - previous).mean())
        previous = small

    peak = float(scores.max())
    if peak > 0:
        scores /= peak
    return scores


def analyse_video(
    video_path: Path,
    max_frames: int | None = None,
) -> tuple[VideoAnalysis, StageResult]:
    """Detect and track faces, then score speaking activity."""
    import cv2
    from insightface.app import FaceAnalysis

    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)

    app = FaceAnalysis(name=DETECTOR_BUNDLE, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        result.status = StageStatus.FAILED
        result.warn("video_unreadable")
        return VideoAnalysis([], 0), result

    tracks: list[FaceTrack] = []
    active: dict[int, int] = {}
    frames: dict[int, np.ndarray] = {}
    next_id = 0
    frame_index = 0

    while True:
        ok, frame_img = capture.read()
        if not ok or (max_frames is not None and frame_index >= max_frames):
            break
        frames[frame_index] = frame_img

        dets = [
            Detection(frame_index, tuple(float(v) for v in f.bbox), float(f.det_score))  # type: ignore[arg-type]
            for f in app.get(frame_img)
        ]
        next_id = _associate(tracks, active, dets, frame_index, next_id)

        # retire tracks that have not been seen recently
        for tid, last_seen in list(active.items()):
            if frame_index - last_seen > MAX_MISSES:
                del active[tid]

        frame_index += 1

    capture.release()
    n_frames = frame_index

    # Drop blips: a track present for under 0.5 s is noise, not a participant.
    min_len = FRAME_RATE // 2
    kept = [t for t in tracks if t.n_frames >= min_len]
    if len(kept) < len(tracks):
        result.warn("short_face_tracks_discarded")

    for track in kept:
        track.activity = score_activity(track, frames, n_frames)

    if not kept:
        result.status = StageStatus.DEGRADED
        result.warn("no_face_tracks")
    result.warn("asd_is_motion_baseline")

    result.seconds = time.perf_counter() - t0
    result.detail = f"{len(kept)} face tracks over {n_frames} frames"
    return VideoAnalysis(tracks=kept, n_frames=n_frames), result
