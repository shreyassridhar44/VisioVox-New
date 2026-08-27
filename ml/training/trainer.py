"""Training loop for SEAVE (docs/07).

Deliberately small and inspectable. The interesting parts of this project are
the objective, the conditioning and the data; a training loop that hides those
behind a framework makes them harder to reason about, and the failure modes
here — a loss term silently contributing nothing, a gate stuck closed — are
found by looking at per-term numbers rather than by a progress bar.

Every run logs a seed and writes per-step metrics, because a result that cannot
be reproduced is not a result.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.conditioning import DropoutConfig, apply_modality_dropout
from models.seave import Seave, SeaveConfig
from training.losses import LossBreakdown, LossWeights, seave_sfo_loss


@dataclass
class TrainConfig:
    steps: int = 1000
    lr: float = 1.5e-4
    grad_accum: int = 4  # with batch 4 this is the documented effective 16
    grad_clip: float = 5.0
    warmup_steps: int = 100
    log_every: int = 10
    seed: int = 20260826
    amp: bool = True  # bf16 on Ampere; no loss scaler needed
    modality_dropout: bool = True
    device: str = "cuda"
    out_dir: Path = field(default_factory=lambda: Path("runs/seave"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay.

    Warmup matters more than usual here: the conditioning path starts at
    identity and the reliability gates start open, so the first steps move
    weights that have no useful gradient signal yet.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + float(np.cos(np.pi * min(1.0, progress))))


@dataclass
class StepResult:
    step: int
    loss: float
    terms: dict[str, float]
    grad_norm: float
    lr: float
    seconds: float


class Trainer:
    def __init__(
        self,
        model: Seave,
        cfg: TrainConfig | None = None,
        weights: LossWeights | None = None,
        dropout: DropoutConfig | None = None,
    ) -> None:
        self.cfg = cfg or TrainConfig()
        self.weights = weights or LossWeights()
        self.dropout = dropout or DropoutConfig()
        self.device = torch.device(self.cfg.device)
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=1e-2)
        self.step = 0
        self.history: list[StepResult] = []
        set_seed(self.cfg.seed)

    def _forward_loss(self, batch: dict[str, torch.Tensor]) -> tuple[LossBreakdown, torch.Tensor]:
        mixture = batch["mixture"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        interferer = batch["interferer"].to(self.device, non_blocking=True)
        active = batch["active"].to(self.device, non_blocking=True)

        speaker = batch.get("speaker_embedding")
        speaker = speaker.to(self.device) if speaker is not None else None
        a_conf = batch.get("audio_confidence")
        a_conf = (
            a_conf.to(self.device)
            if a_conf is not None
            else torch.ones(mixture.shape[0], device=self.device)
        )
        visual = batch.get("visual")
        visual = visual.to(self.device) if visual is not None else None
        v_conf = batch.get("visual_confidence")
        v_conf = v_conf.to(self.device) if v_conf is not None else None

        if self.cfg.modality_dropout and self.model.training:
            if v_conf is None and visual is not None:
                v_conf = torch.ones(visual.shape[:2], device=self.device)
            if v_conf is None:
                v_conf = torch.zeros(mixture.shape[0], 1, device=self.device)
            speaker, a_conf, visual, v_conf = apply_modality_dropout(
                speaker, a_conf, visual, v_conf, self.dropout
            )

        out = self.model(mixture, speaker, a_conf, visual, v_conf)
        estimate = out["estimate"]

        breakdown = seave_sfo_loss(
            estimate,
            target,
            interferers=interferer.unsqueeze(1),
            active=active,
            weights=self.weights,
        )
        return breakdown, estimate

    def train_step(self, batches: list[dict[str, torch.Tensor]]) -> StepResult:
        """One optimiser step over `grad_accum` micro-batches."""
        t0 = time.perf_counter()
        self.model.train()
        lr = lr_at(self.step, self.cfg)
        for group in self.opt.param_groups:
            group["lr"] = lr

        self.opt.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        n = len(batches)

        for batch in batches:
            autocast = torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.cfg.amp and self.device.type == "cuda",
            )
            with autocast:
                breakdown, _ = self._forward_loss(batch)
            # Scale before backward so accumulated gradients average rather
            # than sum; otherwise the effective learning rate scales with
            # grad_accum and the documented lr means something different.
            (breakdown.total / n).backward()  # type: ignore[no-untyped-call]
            for k, v in breakdown.as_floats().items():
                totals[k] = totals.get(k, 0.0) + v / n

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        )
        self.opt.step()

        result = StepResult(
            step=self.step,
            loss=totals["total"],
            terms=totals,
            grad_norm=grad_norm,
            lr=lr,
            seconds=time.perf_counter() - t0,
        )
        self.history.append(result)
        self.step += 1
        return result

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "step": self.step,
                "config": asdict(self.cfg) | {"out_dir": str(self.cfg.out_dir)},
                "model_config": asdict(self.model.cfg),
                "extra": extra or {},
            },
            path,
        )

    def load(self, path: Path) -> dict[str, Any]:
        """Restore model, optimiser and step counter. Returns the saved `extra`.

        Optimiser state is not optional. AdamW carries first and second moment
        estimates per parameter, and dropping them restarts the moment
        estimates from zero — which reads as a loss spike and several thousand
        steps of recovery, on a run that was resumed precisely to avoid losing
        that much work.

        The learning rate is not restored because it is not state: `lr_at`
        derives it from the step, so resuming the counter resumes the schedule.
        A run resumed with a different `--steps` therefore continues on the new
        cosine curve rather than the old one, which is what you want when a run
        is being extended and worth knowing when it is not.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.opt.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])
        extra: dict[str, Any] = dict(checkpoint.get("extra") or {})
        return extra

    def write_history(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.history], indent=2), encoding="utf-8")


def build_model(cfg: SeaveConfig | None = None) -> Seave:
    model = Seave(cfg)
    # Xavier on the projections keeps the initial conditioning contribution at
    # a sane scale; FiLM is left at its identity initialisation deliberately.
    for name, p in model.named_parameters():
        if "film" in name or p.dim() < 2:
            continue
        if name.endswith("weight") and isinstance(
            dict(model.named_modules()).get(name.rsplit(".", 1)[0]), nn.Linear
        ):
            nn.init.xavier_uniform_(p)
    return model
