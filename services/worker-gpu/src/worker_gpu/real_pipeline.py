"""The real pipeline: S0-S9 over user media (docs/05 §2, docs/28 §W8).

This is the module the mock has been standing in for. It orchestrates the stages
that already exist in `ml/pipeline` and takes responsibility for what a stage
cannot know: where the bytes came from, what happens when one stage degrades,
who pays for the GPU time, and what must be deleted afterwards.

Four properties it has to hold:

- **Confinement.** Every ffmpeg/ffprobe touch of user media goes through
  `pipeline.sandbox` (invariant 5, ADR-0009), and the probe runs before anything
  is decoded.
- **Partial results.** Two speakers out of three beats nothing (invariant 8). A
  speaker whose extraction fails is dropped with a warning and the job ends
  `partial`, not `failed`.
- **Resume.** Stages are keyed `(job_id, stage, version)` and a succeeded stage
  is skipped (invariant 7), so a killed worker restarts where it died.
- **Ephemeral biometrics.** Embeddings and mouth crops are deleted when the job
  ends, whatever the outcome (invariant 3, ADR-0008). A legal requirement, not
  an optimisation, so it lives in a `finally`.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visiovox_api.config import Settings

from .stagerunner import StagePlan, StageRunner

logger = logging.getLogger(__name__)

# The 25 fps grid the router and the extractor both work on (passthrough.py).
FRAME_SAMPLES = 640

# Progress weights. Extraction dominates because it is the only stage whose cost
# scales with speaker count as well as duration; a bar that ignores that stalls
# for most of its life.
PLAN: list[StagePlan] = [
    StagePlan("S0_ingest", 0, 8),
    StagePlan("S2a_audio", 1, 22),
    StagePlan("S2b_video", 2, 34),
    StagePlan("S3_fuse", 3, 38),
    StagePlan("S4_enrol", 4, 46),
    StagePlan("S5_extract", 5, 80),
    StagePlan("S7_transcribe", 6, 92),
    StagePlan("S9_package", 7, 100),
]
BY_NAME = {p.stage: p for p in PLAN}


@dataclass
class PipelineOutput:
    manifest: dict[str, Any]
    artifact_dir: Path
    speaker_count: int
    overlap_ratio: float | None
    difficulty: str | None
    warnings: list[str]
    gpu_seconds: float


class PipelineError(RuntimeError):
    """The job cannot produce anything useful."""


def _purge(path: Path) -> None:
    """Delete the scratch tree; loud on failure, never raising.

    It holds mouth crops and speaker embeddings — biometric data with a legal
    deletion requirement — so a failure here is a compliance event worth a log
    line rather than something to swallow.
    """
    try:
        shutil.rmtree(path)
    except Exception:
        logger.exception("failed to purge scratch directory %s", path)


def _read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    data, _ = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32)


def _frame_masks(analysis: Any, speaker: str, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame activity for this speaker and for everyone else.

    On the 25 fps grid the router expects. These two masks are what decide,
    per frame, whether the GPU runs at all — a frame where the target speaks
    alone is passed through untouched (ADR-0010), which is where most of the
    saving comes from on a real recording.
    """
    n_frames = max(1, n_samples // FRAME_SAMPLES)
    target = np.zeros(n_frames, dtype=bool)
    others = np.zeros(n_frames, dtype=bool)

    for turn in analysis.turns:
        lo = max(0, turn.interval.start // FRAME_SAMPLES)
        hi = min(n_frames, turn.interval.end // FRAME_SAMPLES)
        if hi <= lo:
            continue
        if turn.speaker == speaker:
            target[lo:hi] = True
        else:
            others[lo:hi] = True

    return target, others


def _load_embedder(settings: Settings) -> Any:
    """A speaker embedder for enrolment scoring (ECAPA, docs/04 §2).

    S4 without one is not merely less accurate — it is inert. `score_candidates`
    only attaches embeddings when an embedder is supplied, `aggregate_embedding`
    returns None without them, and every speaker then reports `has_audio_cue`
    False. The extractor has nothing to condition on and the job produces
    nothing, having spent the GPU time to get there.

    Loaded from the local model directory rather than fetched, so a job does not
    depend on the network.
    """
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = settings.torch_device if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(
        source=settings.speaker_embedder_source,
        savedir=str(Path(settings.speaker_embedder_dir).expanduser()),
        run_opts={"device": device},
    )

    def embed(segment: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            wav = torch.from_numpy(np.asarray(segment, dtype=np.float32)).unsqueeze(0)
            vector = encoder.encode_batch(wav.to(device))
        return np.asarray(vector.squeeze().detach().cpu().numpy(), dtype=np.float32)

    return embed


def _load_extractor(settings: Settings) -> Any:
    """Load the trained checkpoint once per job.

    The weights are ~200 MB; moving them to the GPU once per speaker is pure
    waste on a multi-speaker recording.
    """
    from pipeline.s5_extract import SeaveExtractor

    path = Path(settings.extractor_checkpoint).expanduser()
    if not path.is_file():
        raise PipelineError(f"extractor checkpoint not found at {path}; set EXTRACTOR_CHECKPOINT")
    return SeaveExtractor.load(path, device=settings.torch_device)


def run_pipeline(
    *,
    runner: StageRunner,
    source: Path,
    settings: Settings,
    project_id: str,
) -> PipelineOutput:
    """Execute the pipeline for one project."""
    from pipeline import s0_ingest, s2a_audio, s3_fuse, s4_enrol, s5_extract, s9_package, sandbox
    from pipeline.s9_package import SpeakerTrack
    from pipeline.types import StageStatus

    scratch = Path(settings.media_work_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="visiovox-job-", dir=scratch))
    keep = work / "out"

    try:
        # ---- S0: probe under confinement, then normalise -------------------
        with runner.stage(BY_NAME["S0_ingest"]) as report:
            staged = work / source.name
            shutil.copy2(source, staged)

            facts = sandbox.facts_from(sandbox.probe(staged, image=settings.media_sandbox_image))
            sandbox.assert_admissible(facts, max_duration_seconds=settings.max_duration_seconds)

            media, s0 = s0_ingest.ingest(staged, work / "media")
            for w in s0.warnings:
                report.warn(w)

        audio = _read_audio(media.analysis_wav)

        # ---- S2a: who speaks when -------------------------------------------
        with runner.stage(BY_NAME["S2a_audio"], gpu=True) as report:
            token = settings.hf_token.get_secret_value()
            analysis, s2a = s2a_audio.analyse(
                audio, hf_token=token or None, device=settings.torch_device
            )
            for w in s2a.warnings:
                report.warn(w)

        if not analysis.speakers:
            raise PipelineError("No distinct speakers were found in this recording.")

        # ---- S2b: faces -------------------------------------------------------
        video_analysis = None
        with runner.stage(BY_NAME["S2b_video"], gpu=True) as report:
            if media.video_mp4 is None:
                report.warn("no_video_track")
            else:
                try:
                    from pipeline import s2b_video

                    video_analysis, s2b = s2b_video.analyse_video(media.video_mp4)
                    for w in s2b.warnings:
                        report.warn(w)
                except Exception as exc:
                    # Invariant 8: losing the visual path costs quality, not the
                    # job. The extractor has an audio-only route for exactly this.
                    logger.warning("visual analysis failed: %s", exc)
                    report.warn("visual_analysis_failed")

        # ---- S3: bind voices to faces -----------------------------------------
        with runner.stage(BY_NAME["S3_fuse"]) as report:
            registry, s3 = s3_fuse.fuse(analysis, video_analysis)
            for w in s3.warnings:
                report.warn(w)

        if not registry.bindings:
            raise PipelineError("No speakers could be registered from this recording.")

        # ---- S4: learn each voice ----------------------------------------------
        enrolments: dict[str, Any] = {}
        with runner.stage(BY_NAME["S4_enrol"], gpu=True) as report:
            try:
                embed = _load_embedder(settings)
            except Exception as exc:
                # Without an embedder every speaker ends up without a cue, so
                # this is fatal rather than a degradation worth continuing past.
                raise PipelineError(
                    "the speaker embedder could not be loaded, so no voice could be learned"
                ) from exc

            for binding in registry.bindings:
                enrolment, s4 = s4_enrol.enrol(analysis, audio, binding.speaker, embed=embed)
                enrolments[binding.speaker] = enrolment
                if s4.status is StageStatus.DEGRADED:
                    report.warn(f"weak_enrolment_speaker_{binding.ordinal}")

        # ---- S5: separate --------------------------------------------------------
        tracks: list[SpeakerTrack] = []
        with runner.stage(BY_NAME["S5_extract"], gpu=True) as report:
            extractor = _load_extractor(settings)

            for binding in registry.bindings:
                enrolment = enrolments[binding.speaker]
                if not enrolment.has_audio_cue:
                    # Without a cue there is nothing to condition on, and an
                    # unconditioned extraction is not this speaker's voice.
                    report.warn(f"no_enrolment_cue_speaker_{binding.ordinal}")
                    continue

                target_active, others_active = _frame_masks(analysis, binding.speaker, len(audio))
                try:
                    extraction, _ = s5_extract.extract(
                        audio,
                        enrolment.audio_embedding,
                        extractor,
                        target_active=target_active,
                        others_active=others_active,
                    )
                except Exception as exc:
                    # One speaker failing must not lose the others.
                    logger.warning("extraction failed for %s: %s", binding.speaker, exc)
                    report.warn(f"extraction_failed_speaker_{binding.ordinal}")
                    continue

                confidence = (
                    float(np.mean(extraction.confidence)) if extraction.confidence.size else 0.0
                )
                tracks.append(
                    SpeakerTrack(
                        speaker_id=binding.speaker,
                        ordinal=binding.ordinal,
                        label=binding.label,
                        faithful=extraction.audio,
                        modality=binding.modality,
                        speaking_ratio=binding.speaking_ratio,
                        mean_confidence=confidence,
                        extraction_ok=True,
                    )
                )

        if not tracks:
            raise PipelineError("No speaker could be separated from this recording.")

        # ---- S7: captions ---------------------------------------------------------
        with runner.stage(BY_NAME["S7_transcribe"], gpu=True) as report:
            try:
                from pipeline import s7_transcribe

                transcriber = s7_transcribe.load_transcriber(device=settings.torch_device)
                for track in tracks:
                    # Invariant 6: the FAITHFUL track is what gets transcribed.
                    # Captions must reflect what was recovered, not what a
                    # generative model found plausible.
                    transcript, _ = s7_transcribe.transcribe(track.faithful, transcriber)
                    track.transcript = transcript
            except Exception as exc:
                logger.warning("transcription unavailable: %s", exc)
                report.warn("captions_unavailable")

        # ---- S9: package -----------------------------------------------------------
        with runner.stage(BY_NAME["S9_package"]) as report:
            manifest, s9 = s9_package.package(
                project_id,
                tracks,
                keep,
                has_video=media.video_mp4 is not None,
                overlap_ratio=analysis.overlap_ratio,
                signed_until="",
                base_url="",
            )
            for w in s9.warnings:
                report.warn(w)

        # Copied out of the scratch tree before the purge below removes it.
        staging = Path(tempfile.mkdtemp(prefix="visiovox-out-", dir=scratch))
        shutil.copytree(keep, staging / "out")

        speakers = manifest.get("speakers", [])
        return PipelineOutput(
            manifest=manifest,
            artifact_dir=staging / "out",
            speaker_count=len(speakers) if isinstance(speakers, list) else 0,
            overlap_ratio=analysis.overlap_ratio,
            difficulty=manifest.get("difficulty"),
            warnings=list(runner.report.warnings),
            gpu_seconds=runner.report.gpu_seconds,
        )
    finally:
        # Invariant 3 / ADR-0008: embeddings and mouth crops go, whatever
        # happened.
        _purge(work)
