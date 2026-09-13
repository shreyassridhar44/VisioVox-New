"""Single-talker passthrough routing (F11, ADR-0010).

The router decides where the model is allowed to touch the signal. Getting it
wrong is expensive in both directions: routing overlap to passthrough leaves
the interferer in, and routing clean speech to the model damages audio that
was already correct — measured at roughly -12 dB SI-SDRi on windows that
needed no separation at all.
"""

from __future__ import annotations

import numpy as np

from pipeline.passthrough import FRAME_SAMPLES, Route, apply, balance, decide, frame_activity


def mask(pattern: str) -> np.ndarray:
    """'0011' -> array([0,0,1,1]), so cases read as pictures."""
    return np.array([float(c) for c in pattern], dtype=np.float32)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def test_target_alone_is_passed_through() -> None:
    route = decide(mask("11111111"), mask("00000000"), min_run=1)
    assert set(route) == {Route.PASSTHROUGH.value}


def test_overlap_is_extracted() -> None:
    route = decide(mask("11111111"), mask("11111111"), min_run=1)
    assert set(route) == {Route.EXTRACT.value}


def test_target_silent_is_silenced_even_when_others_talk() -> None:
    """The leakage case, and the reason SILENCE is a separate decision.

    Asked to extract someone who is not speaking, a model returns whoever is —
    which is precisely the cross-speaker leakage the product exists to avoid.
    """
    route = decide(mask("00000000"), mask("11111111"), min_run=1)
    assert set(route) == {Route.SILENCE.value}


def test_mixed_timeline_routes_each_region() -> None:
    route = decide(mask("00111111"), mask("00001111"), min_run=1)
    assert list(route) == [
        Route.SILENCE.value,
        Route.SILENCE.value,
        Route.PASSTHROUGH.value,
        Route.PASSTHROUGH.value,
        Route.EXTRACT.value,
        Route.EXTRACT.value,
        Route.EXTRACT.value,
        Route.EXTRACT.value,
    ]


def test_short_bursts_are_absorbed() -> None:
    """A two-frame flicker is a diarization wobble, not a turn change.

    80 ms of model output spliced into untouched audio is audible as a
    artefact; leaving the earlier decision in place is inaudible.
    """
    route = decide(mask("1111111111"), mask("0000110000"), min_run=5)
    assert set(route) == {Route.PASSTHROUGH.value}


def test_long_runs_survive_smoothing() -> None:
    route = decide(mask("1111111111"), mask("0000011111"), min_run=5)
    assert list(route[:5]) == [Route.PASSTHROUGH.value] * 5
    assert list(route[5:]) == [Route.EXTRACT.value] * 5


# --------------------------------------------------------------------------
# applying the route
# --------------------------------------------------------------------------


def test_passthrough_returns_the_mixture_untouched() -> None:
    n = FRAME_SAMPLES * 8
    mixture = np.random.default_rng(0).normal(0, 0.1, n).astype(np.float32)
    extracted = np.zeros(n, dtype=np.float32)
    route = np.full(8, Route.PASSTHROUGH.value, dtype=np.int8)

    out = apply(mixture, extracted, route, fade_frames=0)
    np.testing.assert_allclose(out, mixture, atol=1e-6)


def test_silence_emits_nothing() -> None:
    n = FRAME_SAMPLES * 8
    rng = np.random.default_rng(0)
    mixture = rng.normal(0, 0.1, n).astype(np.float32)
    extracted = rng.normal(0, 0.1, n).astype(np.float32)
    route = np.full(8, Route.SILENCE.value, dtype=np.int8)

    out = apply(mixture, extracted, route, fade_frames=0)
    assert float(np.abs(out).max()) == 0.0


def test_boundaries_are_crossfaded() -> None:
    """A hard splice between two signals clicks, and turn changes are frequent."""
    n = FRAME_SAMPLES * 8
    mixture = np.ones(n, dtype=np.float32)
    extracted = -np.ones(n, dtype=np.float32)
    route = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8)

    hard = apply(mixture, extracted, route, fade_frames=0)
    faded = apply(mixture, extracted, route, fade_frames=2)

    jump_hard = float(np.abs(np.diff(hard)).max())
    jump_faded = float(np.abs(np.diff(faded)).max())
    assert jump_faded < jump_hard


# --------------------------------------------------------------------------
# level balancing
# --------------------------------------------------------------------------


def test_balance_equalises_speech_level() -> None:
    rng = np.random.default_rng(0)
    loud = (rng.normal(0, 0.5, FRAME_SAMPLES * 40)).astype(np.float32)
    quiet = (rng.normal(0, 0.005, FRAME_SAMPLES * 40)).astype(np.float32)

    out = balance([loud, quiet], target_rms=0.05)
    levels = [float(np.sqrt((a**2).mean())) for a in out]
    assert abs(levels[0] - levels[1]) / max(levels) < 0.15


def test_balance_measures_speech_not_silence() -> None:
    """Scaling by whole-clip RMS would amplify whoever talks least.

    Two speakers at the same level while talking, one of whom is silent most
    of the time, must come out at the same level — not with the sparse talker
    boosted to compensate for their silence.
    """
    rng = np.random.default_rng(1)
    frames = 40
    talkative = (rng.normal(0, 0.2, FRAME_SAMPLES * frames)).astype(np.float32)
    sparse = np.zeros(FRAME_SAMPLES * frames, dtype=np.float32)
    sparse[: FRAME_SAMPLES * 8] = rng.normal(0, 0.2, FRAME_SAMPLES * 8)

    out = balance([talkative, sparse], target_rms=0.05)
    speaking = [float(np.sqrt((a[: FRAME_SAMPLES * 8] ** 2).mean())) for a in out]
    assert abs(speaking[0] - speaking[1]) / max(speaking) < 0.25


def test_balance_leaves_a_silent_channel_alone() -> None:
    silent = np.zeros(FRAME_SAMPLES * 10, dtype=np.float32)
    out = balance([silent])
    assert float(np.abs(out[0]).max()) == 0.0


def test_frame_activity_finds_the_speech() -> None:
    audio = np.zeros(FRAME_SAMPLES * 10, dtype=np.float32)
    audio[FRAME_SAMPLES * 3 : FRAME_SAMPLES * 6] = 0.4
    active = frame_activity(audio)
    assert list(active) == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
