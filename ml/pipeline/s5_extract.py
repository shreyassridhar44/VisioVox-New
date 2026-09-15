"""S5 — target-speaker extraction with SEAVE (docs/05 §8). **Novelty 2 & 3.**

The product path. `s5_separate.py` is the Tier 0 blind baseline this is measured
against (ADR-0001 decision 2); the two are kept side by side rather than one
replacing the other, because the comparison is a deliverable.

Two structural differences follow from extracting rather than separating:

- **No permutation.** One speaker in, one speaker out. A PIT separator assigns
  its output channels arbitrarily per call, which is why `s5_separate` refuses
  to stitch and makes the caller supply an assignment. Here the identity of the
  output is pinned by the enrolment, so windows stitch directly.
- **Most of the timeline never reaches the model.** The F11 router sends
  single-talker regions through untouched. That is not only a cost saving
  (docs/05 §8.1 point 2): running a separator over already-clean speech
  *removes* quality, so passthrough is also the higher-fidelity choice.

What this module does not do is decide who is speaking. It takes the activity
timeline from S3 and the enrolment from S4 and does as it is told, so that the
routing policy stays testable without a GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from models.seave import Seave, SeaveConfig
from models.visual import VisualFrontend

from . import passthrough
from .passthrough import FRAME_SAMPLES, Route
from .types import ANALYSIS_SAMPLE_RATE, StageResult, StageStatus

STAGE = "S5_extract"
VERSION = "1.0.0"

WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0  # 50% overlap: every interior sample is covered twice
DEFAULT_DEVICE = "cuda:0"


@dataclass
class ExtractionResult:
    """One speaker's isolated track, full length, plus why it sounds that way."""

    audio: np.ndarray
    route: np.ndarray
    confidence: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def processed_fraction(self) -> float:
        """Share of frames that actually went through the GPU.

        The headline number for NFR-PERF-01, and the one to watch if latency
        regresses: a router that has stopped working shows up here long before
        it shows up in the audio.
        """
        if len(self.route) == 0:
            return 0.0
        return float(np.mean(self.route == Route.EXTRACT.value))


def _match_scale(estimate: np.ndarray, mixture: np.ndarray) -> np.ndarray:
    """Put a window's estimate back on the mixture's scale.

    SI-SDR is scale invariant, so a model trained against it has no reason to
    emit a consistent output level -- two windows of the same speaker can come
    back at different gains, and overlap-add turns that into audible amplitude
    wobble at every hop boundary.

    The least-squares fit onto the mixture fixes it against a fixed external
    reference rather than a per-window guess. For an estimate that is roughly
    the target times some constant, the fitted coefficient is that constant's
    inverse, so the result lands on the target's true scale -- the interferer
    does not bias it, being uncorrelated with the target.
    """
    denom = float(np.dot(estimate, estimate))
    if denom < 1e-12:
        return estimate
    alpha = float(np.dot(estimate, mixture)) / denom
    return (estimate * alpha).astype(np.float32)


class SeaveExtractor:
    """A loaded SEAVE checkpoint, ready to run over windows.

    Holds the visual frontend too, when the checkpoint has one. A C1/C3/C4
    checkpoint is audio-only and loads fine here; `has_video` then reports
    False and the extractor runs the audio-only path, which is the same
    degradation the pipeline applies when a face cannot be found.
    """

    def __init__(
        self,
        model: Seave,
        visual: VisualFrontend | None = None,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.visual = visual.to(self.device).eval() if visual is not None else None

    @property
    def has_video(self) -> bool:
        return self.visual is not None

    @classmethod
    def load(cls, path: Path, device: str = DEFAULT_DEVICE) -> SeaveExtractor:
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = dict(state.get("model_config") or {})
        # Filter rather than splat: a checkpoint written by an older revision
        # carries fields this SeaveConfig no longer has, and an unexpected
        # keyword would fail the load of an otherwise usable model.
        known = set(SeaveConfig.__dataclass_fields__)
        cfg = SeaveConfig(**{k: v for k, v in saved.items() if k in known})

        model = Seave(cfg)
        model.load_state_dict(state["model"])

        visual = None
        if "visual" in state:
            visual = VisualFrontend()
            visual.load_state_dict(state["visual"])
        return cls(model, visual, device=device)

    @torch.no_grad()
    def run_window(
        self,
        mixture: np.ndarray,
        enrolment: np.ndarray,
        mouth: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Extract one window. Returns the estimate and the model's confidence."""
        mix = torch.from_numpy(mixture).float().unsqueeze(0).to(self.device)
        emb = torch.from_numpy(enrolment).float().unsqueeze(0).to(self.device)
        a_conf = torch.ones(1, device=self.device)

        features = None
        v_conf = None
        if mouth is not None and self.visual is not None:
            rois = torch.from_numpy(mouth.astype(np.float32) / 255.0)
            rois = rois.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, frames, H, W)
            features = self.visual(rois)
            v_conf = torch.ones(1, features.shape[1], device=self.device)

        out = self.model(mix, emb, a_conf, features, v_conf)
        estimate = out["estimate"].squeeze(0).cpu().numpy().astype(np.float32)
        confidence = float(out["confidence"].squeeze(0)) if "confidence" in out else 1.0
        return estimate, confidence


def _window_starts(n: int, win: int, hop: int) -> list[int]:
    starts = list(range(0, max(1, n - win + 1), hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


def extract_speaker(
    mixture: np.ndarray,
    enrolment: np.ndarray,
    extractor: SeaveExtractor,
    *,
    target_active: np.ndarray,
    others_active: np.ndarray,
    mouth: np.ndarray | None = None,
    min_run: int = 5,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> ExtractionResult:
    """Isolate one speaker across a whole recording, routing as docs/05 §8.1.

    `target_active` and `others_active` are per-frame booleans on the 25 fps
    grid, from S3. `mouth` is that speaker's ROI sequence on the same grid, or
    None when S2B found no usable face -- in which case the audio-only path
    runs and the result is still full length.
    """
    n = len(mixture)
    route = passthrough.decide(target_active, others_active, min_run=min_run)

    win = int(window_seconds * ANALYSIS_SAMPLE_RATE)
    hop = int(hop_seconds * ANALYSIS_SAMPLE_RATE)
    estimate = np.zeros(n, dtype=np.float32)
    envelope = np.zeros(n, dtype=np.float64)
    confidences: list[float] = []

    if np.any(route == Route.EXTRACT.value):
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
        for start in _window_starts(n, win, hop):
            lo_frame = start // FRAME_SAMPLES
            hi_frame = min(len(route), (start + win) // FRAME_SAMPLES)
            # Skip windows the router will discard anyway. This is where the
            # 80-95% GPU saving actually happens; deciding the route alone
            # would still have paid for every window.
            if not np.any(route[lo_frame:hi_frame] == Route.EXTRACT.value):
                continue

            chunk = mixture[start : start + win]
            span = len(chunk)
            if span < win:
                chunk = np.pad(chunk, (0, win - span))

            roi = None
            if mouth is not None and extractor.has_video:
                frames_per_window = win // FRAME_SAMPLES
                roi = mouth[lo_frame : lo_frame + frames_per_window]
                if len(roi) < frames_per_window:
                    pad = frames_per_window - len(roi)
                    roi = np.concatenate([roi, np.zeros((pad, *roi.shape[1:]), roi.dtype)])

            est, conf = extractor.run_window(chunk, enrolment, roi)
            est = _match_scale(est, chunk)
            confidences.append(conf)

            estimate[start : start + span] += (est[:span] * hann[:span]).astype(np.float32)
            envelope[start : start + span] += hann[:span]

        envelope[envelope < 1e-8] = 1.0
        estimate = (estimate / envelope).astype(np.float32)

    audio = passthrough.apply(mixture, estimate, route)
    return ExtractionResult(
        audio=audio,
        route=route,
        confidence=np.asarray(confidences, dtype=np.float32),
    )


def extract(
    mixture: np.ndarray,
    enrolment: np.ndarray,
    extractor: SeaveExtractor,
    *,
    target_active: np.ndarray,
    others_active: np.ndarray,
    mouth: np.ndarray | None = None,
) -> tuple[ExtractionResult, StageResult]:
    """Stage wrapper: same call, plus the timing and status the runner records."""
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)
    extraction = extract_speaker(
        mixture,
        enrolment,
        extractor,
        target_active=target_active,
        others_active=others_active,
        mouth=mouth,
    )
    result.seconds = time.perf_counter() - t0
    modality = "audio+video" if (mouth is not None and extractor.has_video) else "audio-only"
    result.detail = f"{extraction.processed_fraction:.0%} of frames extracted, {modality}"
    return extraction, result
