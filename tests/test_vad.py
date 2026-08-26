"""Multi-channel VAD tests.

These exist because two real bugs got through: a per-track threshold that
counted close-talk bleed as speech, and an absolute dB floor that behaved
differently on two AMI meetings recorded ~30 dB apart. Both produced
plausible-looking numbers, which is what made them dangerous.
"""

from __future__ import annotations

import numpy as np

from pipeline.vad import overlap_ratio, speech_masks

RATE = 16_000


def _talker(n: int, seed: int, level_db: float = -20.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amp = 10 ** (level_db / 20)
    return (rng.standard_normal(n) * amp).astype(np.float32)


def _two_speaker_scene(bleed_db: float = -28.0) -> np.ndarray:
    """Speaker A talks first, then B, with a short genuine overlap.

    Each headset also carries the other talker at `bleed_db`, which is what a
    close-talking mic actually picks up.
    """
    n = 30 * RATE
    a_src = np.zeros(n, dtype=np.float32)
    b_src = np.zeros(n, dtype=np.float32)
    a_src[: 12 * RATE] = _talker(12 * RATE, 1)
    b_src[10 * RATE : 24 * RATE] = _talker(14 * RATE, 2)  # 2 s overlap

    bleed = 10 ** (bleed_db / 20)
    head_a = a_src + b_src * bleed
    head_b = b_src + a_src * bleed
    return np.stack([head_a, head_b])


def test_bleed_is_not_counted_as_speech() -> None:
    """Overlap must be ~2 s of 30, not the whole recording."""
    tracks = _two_speaker_scene()
    masks = speech_masks(tracks)
    ratio = overlap_ratio(masks)
    assert 0.02 < ratio < 0.15, f"overlap {ratio:.3f} — bleed is being counted"


def test_vad_is_invariant_to_recording_gain() -> None:
    """The bug: an absolute dB floor made a hotter meeting look fully overlapped.

    AMI meetings differ by ~30 dB in level. Scaling every channel must not
    change who is judged to be speaking.
    """
    tracks = _two_speaker_scene()
    quiet = speech_masks(tracks * 0.01)  # -40 dB
    loud = speech_masks(tracks * 10.0)  # +20 dB
    assert np.array_equal(quiet, loud)


def test_detects_the_genuine_overlap_region() -> None:
    tracks = _two_speaker_scene()
    masks = speech_masks(tracks)
    both = masks.sum(axis=0) >= 2
    # the overlap sits at 10-12 s; frames are 10 ms
    assert both[1000:1200].mean() > 0.5, "genuine overlap not detected"
    assert both[400:900].mean() < 0.1, "single-talker region reported as overlap"


def test_silent_channel_is_never_active() -> None:
    n = 10 * RATE
    tracks = np.stack([_talker(n, 3), np.zeros(n, dtype=np.float32)])
    masks = speech_masks(tracks)
    assert masks[1].sum() == 0
