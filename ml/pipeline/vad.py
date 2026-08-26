"""Multi-channel speech activity, robust to close-talk bleed.

Shared because two callers previously each had their own idea of what
"speaking" means, and they disagreed. Window selection used a dominance test;
the permutation measurement used an absolute -50 dBFS floor. AMI meetings are
recorded at very different gains -- IS1000a headsets sit roughly 30 dB hotter
than ES2002a -- so the absolute floor marked silence as speech on the louder
meeting and scored every window as overlapped. One definition, one place.
"""

from __future__ import annotations

import numpy as np

FRAME_SAMPLES = 160  # 10 ms at 16 kHz
DEFAULT_DOMINANCE_DB = 15.0
DEFAULT_FLOOR_BELOW_P99_DB = 20.0


def frame_energy_db(tracks: np.ndarray, frame: int = FRAME_SAMPLES) -> np.ndarray:
    """(n_tracks, n_frames) short-term energy in dB."""
    n_tracks = tracks.shape[0]
    n = tracks.shape[1] // frame
    energy = (tracks[:, : n * frame] ** 2).reshape(n_tracks, n, frame).mean(axis=2)
    return 10.0 * np.log10(energy + 1e-12)


def speech_masks(
    tracks: np.ndarray,
    dominance_db: float = DEFAULT_DOMINANCE_DB,
    floor_below_p99_db: float = DEFAULT_FLOOR_BELOW_P99_DB,
) -> np.ndarray:
    """Per-speaker speech mask at 10 ms resolution.

    A close-talking headset picks up the other participants. Measured on AMI
    that bleed sits about 28 dB below the actual talker, so a per-track
    threshold alone marks bleed as speech and reports near-total overlap.

    Two conditions: loud relative to that speaker's own speech level, and
    within `dominance_db` of the loudest channel at that instant. Bleed fails
    the second. Both are relative, so recording gain does not change the answer.

    Returns (n_tracks, n_frames) of bool.
    """
    db = frame_energy_db(tracks)
    own_floor = np.percentile(db, 99, axis=1, keepdims=True) - floor_below_p99_db
    frame_max = db.max(axis=0, keepdims=True)
    mask: np.ndarray = (db > own_floor) & (db > frame_max - dominance_db)
    return mask


def overlap_ratio(masks: np.ndarray) -> float:
    """Fraction of frames with at least two speakers active."""
    return float((masks.sum(axis=0) >= 2).mean())
