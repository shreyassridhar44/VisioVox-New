"""Single-talker passthrough routing (F11, ADR-0010).

Between 80% and 95% of a real timeline is one person talking. Running an
extractor over those regions is not merely wasteful — it is harmful. Measured
on AMI-Eval, where a window already sits ~20 dB clean because only the target
is speaking, the model still degrades it: there is nothing to remove, so
whatever it changes is damage.

So the router decides, per frame, which of three things a region needs:

    PASSTHROUGH  only the target is speaking — copy the mixture through
    EXTRACT      others are speaking too — the model earns its place
    SILENCE      the target is not speaking — emit nothing

That last case matters as much as the first. A model asked to extract a
speaker who is silent will hallucinate something, and what it hallucinates is
whoever *is* talking — which is the leakage the product exists to prevent.

The decision is made on frame-level speech activity, so it composes with S2A's
diarization in the real pipeline and with reference activity in evaluation.
Transitions are smoothed and hysteresis applied, because a router that flips
state every other frame produces audible chattering at the boundaries.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

FRAME_SAMPLES = 640  # 40 ms at 16 kHz, the 25 fps grid used throughout


class Route(Enum):
    SILENCE = 0
    PASSTHROUGH = 1
    EXTRACT = 2


def decide(
    target_active: np.ndarray,
    others_active: np.ndarray,
    *,
    min_run: int = 5,
) -> np.ndarray:
    """Per-frame routing decision.

    `target_active` and `others_active` are frame-level 0/1 masks; the second
    is the union over every non-target speaker. `min_run` is the shortest run
    of frames a decision may occupy — 5 frames is 200 ms, below which a change
    of state is heard as a click rather than as a transition.
    """
    if target_active.shape != others_active.shape:
        raise ValueError("activity masks must be the same length")

    route = np.where(
        target_active < 0.5,
        Route.SILENCE.value,
        np.where(others_active >= 0.5, Route.EXTRACT.value, Route.PASSTHROUGH.value),
    ).astype(np.int8)

    return _smooth(route, min_run)


def _smooth(route: np.ndarray, min_run: int) -> np.ndarray:
    """Absorb runs shorter than `min_run` into whichever neighbour precedes them.

    Deliberately biased to the earlier state rather than to the majority: a
    short EXTRACT inside a long PASSTHROUGH is usually a diarization wobble,
    and extending the safer decision costs nothing, whereas a 120 ms burst of
    model output inside untouched audio is audible.
    """
    if route.size == 0 or min_run <= 1:
        return route

    out = route.copy()
    start = 0
    for i in range(1, len(out) + 1):
        if i == len(out) or out[i] != out[start]:
            if i - start < min_run and start > 0:
                out[start:i] = out[start - 1]
            start = i
    return out


def apply(
    mixture: np.ndarray,
    extracted: np.ndarray,
    route: np.ndarray,
    *,
    fade_frames: int = 2,
) -> np.ndarray:
    """Combine mixture and model output according to the route.

    Crossfaded at every boundary. Splicing two signals that disagree about
    phase produces a click at the join, and the join happens at every turn
    change — the most common event in a conversation.
    """
    n = len(mixture)
    gain_mix = np.zeros(n, dtype=np.float32)
    gain_est = np.zeros(n, dtype=np.float32)

    for frame, decision in enumerate(route):
        lo = frame * FRAME_SAMPLES
        hi = min(n, lo + FRAME_SAMPLES)
        if lo >= n:
            break
        if decision == Route.PASSTHROUGH.value:
            gain_mix[lo:hi] = 1.0
        elif decision == Route.EXTRACT.value:
            gain_est[lo:hi] = 1.0

    if fade_frames > 0:
        window = fade_frames * FRAME_SAMPLES
        kernel = np.ones(window, dtype=np.float32) / window
        gain_mix = np.convolve(gain_mix, kernel, mode="same").astype(np.float32)
        gain_est = np.convolve(gain_est, kernel, mode="same").astype(np.float32)

    return (mixture * gain_mix + extracted[:n] * gain_est).astype(np.float32)


def frame_activity(audio: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Frame-level speech activity, relative to the clip's own peak frame."""
    frames = len(audio) // FRAME_SAMPLES
    if frames == 0:
        return np.zeros(0, dtype=np.float32)
    energy = (audio[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES) ** 2).mean(axis=1)
    peak = float(energy.max())
    if peak <= 0:
        return np.zeros(frames, dtype=np.float32)
    return (energy > peak * threshold).astype(np.float32)


def balance(references: list[np.ndarray], target_rms: float = 0.05) -> list[np.ndarray]:
    """Scale each speaker to a common level over the frames they are speaking.

    AMI sums four headset channels, and headset gain varies enormously between
    participants — measured on AMI-Eval, the target can sit 20 dB above the
    others in one meeting and 14 dB below them in another. Neither resembles a
    camera microphone, where everyone arrives at broadly comparable level, so
    an evaluation built on the raw sum is measuring the gain staging rather
    than the model.

    Normalising on *speech-active* frames rather than the whole clip matters:
    scaling by overall RMS would amplify whoever talks least.
    """
    out: list[np.ndarray] = []
    for audio in references:
        active = frame_activity(audio)
        if active.sum() == 0:
            out.append(audio.astype(np.float32))
            continue
        frames = len(active)
        usable = audio[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES)
        speech = usable[active > 0.5]
        rms = float(np.sqrt((speech**2).mean())) if speech.size else 0.0
        gain = (target_rms / rms) if rms > 1e-9 else 1.0
        out.append((audio * gain).astype(np.float32))
    return out


__all__ = ["FRAME_SAMPLES", "Route", "apply", "balance", "decide", "frame_activity"]
