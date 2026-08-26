"""Realistic mixture simulation (docs/06 §5).

The naive recipe — `mix = a + b`, fully overlapped, anechoic, level-balanced —
is what LibriMix does and what most papers train on. It produces models that
score well on benchmarks and collapse on real recordings, because none of its
assumptions hold outside the benchmark.

docs/06 §5 singles out steps 1, 3 and 6 as the ones usually skipped and the
ones that matter most. This module implements the audio side:

1. **Turn-taking, not constant overlap.** Real conversation runs 5-20%
   overlapped; LibriMix is 100%. A model trained on permanent overlap learns to
   always separate, and separating already-clean single-talker audio degrades
   it — measured, not assumed (docs/27 §3).
2. **Per-source room impulse responses.** Each speaker sits somewhere different
   and gets their own RIR. Convolving everyone with one response is a
   fundamentally easier problem than the real one.
3. **Level imbalance.** One speaker is nearer the microphone. Balanced mixtures
   remove a cue the model would otherwise have to cope without.
4. **Additive noise** at realistic SNR.
5. **Codec round-trip.** Real uploads arrive compressed.

Video degradation is `degrade_video` in the same spirit, applied separately
because it operates on frames rather than samples.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RATE = 16_000


@dataclass(frozen=True)
class SimConfig:
    """Ranges follow docs/06 §5."""

    overlap_ratio: tuple[float, float] = (0.05, 0.35)
    room_dim_low: tuple[float, float, float] = (3.0, 3.0, 2.4)
    room_dim_high: tuple[float, float, float] = (10.0, 8.0, 3.5)
    rt60: tuple[float, float] = (0.15, 0.8)
    level_spread_db: float = 6.0
    snr_db: tuple[float, float] = (5.0, 25.0)
    codec_probability: float = 0.5
    codecs: tuple[str, ...] = ("aac_64k", "opus_32k", "mp3_96k")
    seed: int | None = None


@dataclass
class Simulated:
    mixture: np.ndarray
    sources: list[np.ndarray]  # post-RIR, post-gain, pre-sum — the references
    timeline: list[tuple[int, int]]  # per-source (start, end) in samples
    rt60: float
    snr_db: float
    codec: str | None
    overlap_ratio: float
    room_dim: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))


def sample_turn_schedule(
    n_sources: int,
    total_samples: int,
    overlap_ratio: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Contiguous speaking spans producing approximately `overlap_ratio`.

    Deliberately simple: speaker 0 anchors the timeline, and each other speaker
    is placed so their span overlaps the anchor by the requested fraction. The
    point is to break the "everyone talks constantly" assumption, not to model
    turn-taking dynamics — that would need the AMI turn/gap distributions and
    is only worth doing once the simpler version is shown insufficient.
    """
    spans: list[tuple[int, int]] = []
    anchor_len = int(total_samples * rng.uniform(0.6, 1.0))
    anchor_start = int(rng.integers(0, max(1, total_samples - anchor_len + 1)))
    spans.append((anchor_start, anchor_start + anchor_len))

    for _ in range(1, n_sources):
        want_overlap = int(anchor_len * overlap_ratio)
        length = int(total_samples * rng.uniform(0.4, 0.9))
        # place so the intersection with the anchor is about want_overlap
        if rng.random() < 0.5:
            start = anchor_start + anchor_len - want_overlap
        else:
            start = anchor_start - length + want_overlap
        start = int(np.clip(start, 0, max(0, total_samples - 1)))
        end = int(min(total_samples, start + length))
        if end <= start:
            start, end = 0, min(total_samples, length)
        spans.append((start, end))
    return spans


def _rt60_to_absorption(rt60: float, dim: tuple[float, float, float]) -> float:
    """Sabine, inverted, clamped to what the image-source method can realise."""
    x, y, z = dim
    volume = x * y * z
    surface = 2 * (x * y + y * z + x * z)
    absorption = 24 * np.log(10.0) * volume / (343.0 * surface * max(rt60, 1e-3))
    return float(np.clip(absorption, 0.05, 0.95))


def sample_rirs(
    n_sources: int, rng: np.random.Generator, cfg: SimConfig
) -> tuple[list[np.ndarray], float, tuple[float, float, float]]:
    """One RIR per source, all from the same room but different positions."""
    import pyroomacoustics as pra

    dim = tuple(
        float(rng.uniform(lo, hi))
        for lo, hi in zip(cfg.room_dim_low, cfg.room_dim_high, strict=True)
    )
    rt60 = float(np.exp(rng.uniform(np.log(cfg.rt60[0]), np.log(cfg.rt60[1]))))
    absorption = _rt60_to_absorption(rt60, dim)  # type: ignore[arg-type]

    try:
        room = pra.ShoeBox(list(dim), fs=RATE, materials=pra.Material(absorption), max_order=12)
        mic = np.array([[dim[0] / 2], [dim[1] / 2], [1.2]])
        room.add_microphone_array(pra.MicrophoneArray(mic, RATE))
        for _ in range(n_sources):
            # keep sources off the walls; image-source degenerates at the boundary
            pos = [
                float(rng.uniform(0.5, dim[0] - 0.5)),
                float(rng.uniform(0.5, dim[1] - 0.5)),
                float(rng.uniform(1.0, min(2.0, dim[2] - 0.3))),
            ]
            room.add_source(pos)
        room.compute_rir()
        rirs = [np.asarray(room.rir[0][i], dtype=np.float32) for i in range(n_sources)]
    except Exception:
        # A degenerate geometry should not abort a training run; fall back to
        # anechoic and let the caller see rt60 for what it is.
        rirs = [np.array([1.0], dtype=np.float32) for _ in range(n_sources)]
        rt60 = 0.0

    return rirs, rt60, dim  # type: ignore[return-value]


def _convolve_keep_length(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if len(rir) <= 1:
        return x.astype(np.float32)
    y = np.convolve(x, rir)[: len(x)]
    return y.astype(np.float32)


def codec_roundtrip(audio: np.ndarray, codec: str) -> np.ndarray:
    """Encode and decode through ffmpeg. Returns the input unchanged on failure."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return audio
    settings = {
        "aac_64k": ("aac", "64k", "m4a"),
        "opus_32k": ("libopus", "32k", "ogg"),
        "mp3_96k": ("libmp3lame", "96k", "mp3"),
    }
    if codec not in settings:
        return audio
    encoder, bitrate, ext = settings[codec]

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "in.f32"
        enc = Path(tmp) / f"mid.{ext}"
        raw.write_bytes(audio.astype(np.float32).tobytes())
        common = ["-loglevel", "error", "-y"]
        a = subprocess.run(  # noqa: S603 - resolved path, fixed argv
            [
                ffmpeg,
                *common,
                "-f",
                "f32le",
                "-ar",
                str(RATE),
                "-ac",
                "1",
                "-i",
                str(raw),
                "-c:a",
                encoder,
                "-b:a",
                bitrate,
                str(enc),
            ],
            check=False,
            capture_output=True,
        )
        if a.returncode != 0:
            return audio
        b = subprocess.run(  # noqa: S603 - resolved path, fixed argv
            [ffmpeg, *common, "-i", str(enc), "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-"],
            check=False,
            capture_output=True,
        )
        if b.returncode != 0 or not b.stdout:
            return audio
        out = np.frombuffer(b.stdout, dtype=np.float32).copy()

    if len(out) >= len(audio):
        return out[: len(audio)]
    return np.pad(out, (0, len(audio) - len(out)))


def simulate(
    sources: list[np.ndarray],
    noise: np.ndarray | None = None,
    cfg: SimConfig | None = None,
) -> Simulated:
    """Build one realistic mixture from clean single-speaker sources."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    n = len(sources)
    if n < 2:
        raise ValueError("need at least two sources")

    total = min(len(s) for s in sources)
    sources = [s[:total].astype(np.float32) for s in sources]

    # 1. turn-taking
    target_overlap = float(rng.uniform(*cfg.overlap_ratio))
    spans = sample_turn_schedule(n, total, target_overlap, rng)
    gated: list[np.ndarray] = []
    for src, (a, b) in zip(sources, spans, strict=True):
        g = np.zeros(total, dtype=np.float32)
        g[a:b] = src[a:b]
        gated.append(g)

    # 2. per-source room response
    rirs, rt60, dim = sample_rirs(n, rng, cfg)
    reverbed = [_convolve_keep_length(g, r) for g, r in zip(gated, rirs, strict=True)]

    # 3. level imbalance
    leveled = [s * float(10 ** (rng.normal(0.0, cfg.level_spread_db) / 20)) for s in reverbed]

    mixture = np.sum(leveled, axis=0).astype(np.float32)

    # 4. additive noise at a sampled SNR
    snr = float(rng.uniform(*cfg.snr_db))
    if noise is not None and len(noise) > 0:
        tiled = np.resize(noise.astype(np.float32), total)
        sig_p = float(np.mean(mixture**2)) + 1e-12
        noi_p = float(np.mean(tiled**2)) + 1e-12
        scale = np.sqrt(sig_p / (noi_p * (10 ** (snr / 10))))
        mixture = mixture + tiled * scale

    # 5. codec round-trip
    codec: str | None = None
    if rng.random() < cfg.codec_probability:
        codec = str(rng.choice(np.array(cfg.codecs)))
        mixture = codec_roundtrip(mixture, codec)

    peak = float(np.abs(mixture).max())
    if peak > 0.99:
        k = 0.99 / peak
        mixture = mixture * k
        leveled = [s * k for s in leveled]

    active = np.zeros(total, dtype=np.int16)
    for a, b in spans:
        active[a:b] += 1
    measured_overlap = float((active >= 2).mean())

    return Simulated(
        mixture=mixture.astype(np.float32),
        sources=[s.astype(np.float32) for s in leveled],
        timeline=spans,
        rt60=rt60,
        snr_db=snr,
        codec=codec,
        overlap_ratio=measured_overlap,
        room_dim=dim,
    )


def degrade_video(frames: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Visual degradation matched to the audio conditions (docs/06 §5 step 6).

    Occlusion, blur, frame drops and resolution loss are what actually happens
    to uploaded video, and a visual pathway trained only on clean frames
    over-trusts the modality — which is precisely what the reliability gating
    in Novelty 2 exists to handle. It has to see degraded input to learn that.
    """
    rng = rng or np.random.default_rng()
    out = frames.copy()
    n, h, w = out.shape[0], out.shape[1], out.shape[2]

    if rng.random() < 0.15:  # occlusion over a contiguous run of frames
        start = int(rng.integers(0, max(1, n // 2)))
        length = int(rng.integers(1, max(2, n // 4)))
        y0 = int(rng.integers(0, max(1, h // 2)))
        x0 = int(rng.integers(0, max(1, w // 2)))
        out[start : start + length, y0 : y0 + h // 3, x0 : x0 + w // 3] = 0

    if rng.random() < 0.20:  # motion blur, temporal
        k = 3
        smoothed = out.astype(np.float32).copy()
        for i in range(k, n):
            smoothed[i] = out[i - k : i + 1].mean(axis=0)
        out = smoothed.astype(out.dtype)

    if rng.random() < 0.10:  # dropped frames held at the previous value
        for i in range(1, n):
            if rng.random() < 0.05:
                out[i] = out[i - 1]

    if rng.random() < 0.20:  # resolution loss via decimate and repeat
        factor = int(rng.integers(2, 5))
        small = out[:, ::factor, ::factor]
        out = np.repeat(np.repeat(small, factor, axis=1), factor, axis=2)[:, :h, :w]

    return out
