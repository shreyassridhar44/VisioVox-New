"""Phase 0 pretrained-model smoke test.

Runs every pretrained component the pipeline depends on, once, on one real
two-speaker mixture. It proves the plumbing before any pipeline code is
written -- which is the whole point of doing it now (docs/21 Phase 0).

It deliberately does NOT check quality. A stage passes if it loads, runs on
real audio and returns output with the right shape. Quality is Phase 1's job.

Usage:  make smoke
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mix_2spk.wav"
SAMPLE_RATE = 16_000

Status = Literal["OK", "SKIP", "FAIL"]


@dataclass
class Result:
    stage: str
    component: str
    status: Status
    detail: str
    seconds: float = 0.0


def load_fixture() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(FIXTURE, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def check_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable — every GPU stage depends on this")
    props = torch.cuda.get_device_properties(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 unsupported — docs/25 §5 training config assumes it")
    return (
        f"{props.name}, sm_{props.major}{props.minor}, {props.total_memory / 1024**3:.0f} GiB, bf16"
    )


def check_wpe() -> str:
    """S1 dereverberation — an algorithm, no weights."""
    from nara_wpe.wpe import wpe_v8 as wpe

    audio, sr = load_fixture()
    stft = torch.stft(
        torch.from_numpy(audio), n_fft=512, hop_length=128, return_complex=True
    ).numpy()
    # nara_wpe expects (channels, frequency, frames)
    out = wpe(stft[None, ...].transpose(0, 1, 2), taps=10, delay=3, iterations=3)
    return f"dereverbed {out.shape[-1]} frames, {sr} Hz"


def check_silero_vad() -> str:
    """S2A voice activity detection."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    audio, _ = load_fixture()
    ts = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=SAMPLE_RATE)
    if not ts:
        raise RuntimeError("no speech detected in a speech fixture")
    span = (ts[-1]["end"] - ts[0]["start"]) / SAMPLE_RATE
    return f"{len(ts)} speech segments spanning {span:.2f}s"


def check_speaker_embedding() -> str:
    """S2A speaker embeddings (ECAPA stands in for ReDimNet at Phase 0)."""
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(REPO_ROOT / "models" / "ecapa"),
        run_opts={"device": "cuda"},
    )
    a, _ = sf.read(REPO_ROOT / "tests" / "fixtures" / "ref_spkA.wav", dtype="float32")
    b, _ = sf.read(REPO_ROOT / "tests" / "fixtures" / "ref_spkB.wav", dtype="float32")
    ea = model.encode_batch(torch.from_numpy(a).unsqueeze(0)).squeeze()
    eb = model.encode_batch(torch.from_numpy(b).unsqueeze(0)).squeeze()
    cos = torch.nn.functional.cosine_similarity(ea, eb, dim=0).item()
    # Two different speakers must not be near-identical; this catches a model
    # that silently returns constant embeddings.
    if cos > 0.95:
        raise RuntimeError(f"distinct speakers scored {cos:.3f} — embeddings look degenerate")
    return f"dim {ea.shape[-1]}, cross-speaker cosine {cos:.3f}"


def check_separation() -> str:
    """S5 Tier-0 baseline: off-the-shelf blind separation (docs/25 §4).

    Uses the 16 kHz checkpoint deliberately. `sepformer-wsj02mix` is 8 kHz and
    resamples 16 kHz input down without warning -- it only says so in a log
    line -- which would make the Tier 0 baseline quietly incomparable with the
    Tier 1 numbers it exists to be measured against.
    """
    from speechbrain.inference.separation import SepformerSeparation

    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k",
        savedir=str(REPO_ROOT / "models" / "sepformer16k"),
        run_opts={"device": "cuda"},
    )
    n_in = len(load_fixture()[0])
    est = model.separate_file(path=str(FIXTURE))
    n_src = est.shape[-1]
    if n_src != 2:
        raise RuntimeError(f"expected 2 estimated sources, got {n_src}")
    # A silent downsample halves the sample count; catch it as a hard failure
    # rather than trusting the model to honour the input rate.
    n_out = int(est.shape[1])
    if abs(n_out - n_in) > 0.01 * n_in:
        raise RuntimeError(
            f"sample-rate mismatch: {n_in} in, {n_out} out — checkpoint is not 16 kHz"
        )
    return f"{n_src} sources, {n_out} samples @ {SAMPLE_RATE} Hz (rate preserved)"


def check_whisper() -> str:
    """S7 transcription."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
        download_root=str(REPO_ROOT / "models" / "whisper"),
    )
    segments, info = model.transcribe(str(FIXTURE), beam_size=1)
    text = " ".join(s.text.strip() for s in segments)
    if not text:
        raise RuntimeError("empty transcript from a speech fixture")
    preview = text[:60] + ("…" if len(text) > 60 else "")
    return f'lang={info.language} "{preview}"'


def check_pyannote() -> str:
    """S2A diarization — gated on HuggingFace, needs an accepted licence."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise PermissionError(
            "HF_TOKEN not set. Accept the terms for pyannote/speaker-diarization-3.1 "
            "and pyannote/segmentation-3.0, then put the token in .env.local"
        )
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

    # pyannote 3.x renamed use_auth_token -> token, and from_pretrained
    # returns None when the licence has not been accepted for the account.
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    if pipeline is None:
        raise PermissionError(
            "pyannote returned no pipeline — accept the terms for "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0"
        )
    pipeline.to(torch.device("cuda"))
    # Feed a waveform rather than a path. pyannote 3.x reads files through
    # torchcodec, whose shared library needs an FFmpeg build matching the one
    # torch was compiled against; passing the tensor skips that dependency
    # entirely and we have already decoded the audio anyway.
    audio, sr = load_fixture()
    result = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr})
    # pyannote 4.x wraps the annotation in a DiarizeOutput dataclass; 3.x
    # returned the Annotation directly. Accept either.
    annotation = getattr(result, "speaker_diarization", result)
    if not isinstance(annotation, Annotation):
        raise RuntimeError(f"unexpected diarization result: {type(annotation).__name__}")

    # ADR-0008: the pipeline can also return speaker embeddings. We do not
    # retain them here, and the real S2A must delete them when the job ends
    # unless the user has explicitly opted in.
    turns = list(annotation.itertracks(yield_label=True))
    return f"{len(annotation.labels())} speakers over {len(turns)} turns"


STAGES: list[tuple[str, str, Callable[[], str]]] = [
    ("--", "CUDA / GPU", check_cuda),
    ("S1", "WPE dereverberation", check_wpe),
    ("S2A", "Silero VAD", check_silero_vad),
    ("S2A", "ECAPA speaker embedding", check_speaker_embedding),
    ("S2A", "pyannote 3.1 diarization", check_pyannote),
    ("S5", "SepFormer (Tier 0 baseline)", check_separation),
    ("S7", "Whisper large-v3", check_whisper),
]


def main() -> int:
    if not FIXTURE.exists():
        print(f"missing fixture: {FIXTURE}", file=sys.stderr)
        return 2

    print(f"fixture: {FIXTURE.relative_to(REPO_ROOT)}\n")
    results: list[Result] = []

    for stage, component, fn in STAGES:
        t0 = time.perf_counter()
        try:
            detail = fn()
            status: Status = "OK"
        except PermissionError as exc:  # gated model, not a plumbing failure
            detail, status = str(exc), "SKIP"
        except Exception as exc:
            detail, status = f"{type(exc).__name__}: {exc}", "FAIL"
            traceback.print_exc(limit=3)
        dt = time.perf_counter() - t0
        results.append(Result(stage, component, status, detail, dt))
        print(f"  [{status:4}] {stage:4} {component:30} {dt:6.1f}s  {detail}")

    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIP"]
    ok = [r for r in results if r.status == "OK"]

    print(f"\n{len(ok)} ok · {len(skipped)} skipped · {len(failed)} failed")
    for r in skipped:
        print(f"  SKIP {r.component}: {r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
