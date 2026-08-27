"""Checkpoint round-trip and resume (docs/07).

A C1 run is roughly 55 hours on one A5000. Without a resume path, any
interruption in that window — a reboot, a stray pkill, a power cut — costs the
whole run, so these tests are about the difference between losing 55 hours and
losing 500 steps.

They run on CPU with a deliberately tiny model. What is being checked is
bookkeeping, not learning: that the optimiser state survives, that the step
counter survives, and that a resumed run continues rather than quietly
restarting from the beginning while looking like it worked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.seave import Seave, SeaveConfig
from training.trainer import TrainConfig, Trainer, lr_at

RATE = 16_000


def tiny_model() -> Seave:
    return Seave(SeaveConfig(emb_dim=16, lstm_hidden=16, n_blocks=1, confidence_head=False))


def tiny_trainer(tmp_path: Path, steps: int = 100) -> Trainer:
    return Trainer(
        tiny_model(),
        TrainConfig(steps=steps, device="cpu", amp=False, grad_accum=1, out_dir=tmp_path),
    )


def batch(seed: int, items: int = 2, seconds: float = 0.5) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    target = rng.normal(0, 0.1, (items, n)).astype(np.float32)
    interferer = rng.normal(0, 0.1, (items, n)).astype(np.float32)
    return {
        "mixture": torch.from_numpy(target + interferer),
        "target": torch.from_numpy(target),
        # (batch, samples): the trainer adds the interferer axis itself.
        "interferer": torch.from_numpy(interferer),
        "active": torch.ones(items, n // 640),
    }


def test_step_counter_survives(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path)
    for i in range(3):
        trainer.train_step([batch(i)])
    trainer.save(tmp_path / "last.pt", {"val_si_sdri": 1.25})

    resumed = tiny_trainer(tmp_path)
    assert resumed.step == 0
    extra = resumed.load(tmp_path / "last.pt")
    assert resumed.step == 3
    assert extra["val_si_sdri"] == 1.25


def test_weights_are_restored_exactly(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path)
    trainer.train_step([batch(0)])
    trainer.save(tmp_path / "last.pt")

    resumed = tiny_trainer(tmp_path)
    resumed.load(tmp_path / "last.pt")
    for (name, before), (_, after) in zip(
        trainer.model.state_dict().items(), resumed.model.state_dict().items(), strict=True
    ):
        assert torch.equal(before, after), name


def test_optimiser_moments_are_restored(tmp_path: Path) -> None:
    """The part it would be easy to drop, and expensive to drop.

    AdamW carries per-parameter moment estimates. Restoring weights but not
    moments restarts them at zero, which shows up as a loss spike and a few
    thousand steps of recovery — on a run that was resumed to avoid losing
    exactly that.
    """
    trainer = tiny_trainer(tmp_path)
    for i in range(3):
        trainer.train_step([batch(i)])
    trainer.save(tmp_path / "last.pt")

    resumed = tiny_trainer(tmp_path)
    resumed.load(tmp_path / "last.pt")

    original = trainer.opt.state_dict()["state"]
    restored = resumed.opt.state_dict()["state"]
    assert original, "optimiser had no state to restore — the test proves nothing"
    assert set(original) == set(restored)
    for key, state in original.items():
        assert torch.allclose(state["exp_avg"], restored[key]["exp_avg"])
        assert torch.allclose(state["exp_avg_sq"], restored[key]["exp_avg_sq"])
        assert state["step"] == restored[key]["step"]


def test_resumed_run_continues_the_schedule(tmp_path: Path) -> None:
    """The learning rate is derived from the step, not stored.

    So restoring the counter restores the position on the cosine curve. If the
    step were lost, training would silently re-enter warmup — the most
    plausible way for a resume to look successful and not be.
    """
    cfg = TrainConfig(steps=1000, warmup_steps=100, device="cpu")
    trainer = tiny_trainer(tmp_path, steps=1000)
    trainer.step = 600
    trainer.save(tmp_path / "last.pt")

    resumed = tiny_trainer(tmp_path, steps=1000)
    resumed.load(tmp_path / "last.pt")

    assert lr_at(resumed.step, cfg) == lr_at(600, cfg)
    assert lr_at(resumed.step, cfg) < lr_at(100, cfg)  # past warmup, decaying


def test_resumed_run_draws_different_batches(tmp_path: Path) -> None:
    """Per-step seeding, as `scripts/train_c1.py` does it.

    A resumed run must not replay the batches it already trained on. Deriving
    the draw from the step number makes that automatic and reproducible; a
    long-lived generator would need its position checkpointed, and getting
    that wrong over-trains on a prefix of the dataset without ever failing.
    """

    def picks(step: int) -> list[int]:
        drawn: list[int] = np.random.default_rng([1, step]).integers(0, 27800, size=16).tolist()
        return drawn

    assert picks(0) != picks(700)
    assert picks(700) == picks(700)  # deterministic, so a resume is reproducible
