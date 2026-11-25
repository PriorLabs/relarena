"""Numerical settings, optimizer schedule and runtime checks for the fixed recipes."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from statistics import median
from typing import Any

import lightning.pytorch as lit
import torch


@contextmanager
def numerical_settings() -> Iterator[None]:
    """Apply the seeded kernel settings for relflow 0.5.3 fit and prediction."""
    # PyTorch fixes the inter-op pool process-wide after the first parallel work.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    cuda = torch.backends.cuda
    cudnn = torch.backends.cudnn
    flags = (
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cudnn.benchmark,
        cudnn.deterministic,
        cudnn.allow_tf32,
        cuda.matmul.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.allow_tf32 = False
        cuda.matmul.allow_tf32 = False
        cuda.enable_flash_sdp(False)
        cuda.enable_mem_efficient_sdp(False)
        cuda.enable_math_sdp(True)
        yield
    finally:
        (
            flash,
            efficient,
            math_sdp,
            benchmark,
            deterministic,
            tf32,
            matmul_tf32,
            algorithms,
            warn_only,
            threads,
        ) = flags
        cuda.enable_flash_sdp(flash)
        cuda.enable_mem_efficient_sdp(efficient)
        cuda.enable_math_sdp(math_sdp)
        cudnn.benchmark = benchmark
        cudnn.deterministic = deterministic
        cudnn.allow_tf32 = tf32
        cuda.matmul.allow_tf32 = matmul_tf32
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
        torch.set_num_threads(threads)


def scheduler(steps: int, floor: float) -> Callable | None:
    """Decay from the preset learning rate to its floor over the actual updates."""
    if floor == 1.0:
        return None

    def multiplier(step: int) -> float:
        if steps <= 1 or step <= 0:
            return 1.0
        if step >= steps - 1:
            return floor
        return (
            floor + (1.0 - floor) * (1.0 + math.cos(math.pi * step / (steps - 1))) / 2
        )

    def bind(
        model: lit.LightningModule, optimizer: torch.optim.Optimizer
    ) -> dict[str, Any]:
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
        return {"scheduler": schedule, "interval": "step", "frequency": 1}

    return bind


class RuntimeGuard(lit.Callback):
    """Stop an outer fit after two successive first-pass estimates exceed the target."""

    def __init__(
        self, started: float, steps: int, epoch: int, policy: dict[str, float]
    ) -> None:
        """Bind the full-fit budget and first-pass extrapolation settings."""
        self.started = started
        self.steps = steps
        self.epoch = epoch
        self.policy = policy
        self.rates: deque[float] = deque(maxlen=3)
        self.over = 0
        self.last: float | None = None

    def on_train_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Estimate the remaining training and prediction time every 32 updates."""
        step = trainer.global_step
        if step % 32 or step > self.epoch or step >= self.steps:
            return
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        now = time.monotonic()
        last, self.last = self.last, now
        if last is None:
            return
        self.rates.append((now - last) / 32)
        if len(self.rates) < 2:
            return
        cold = median(self.rates)
        forecast = (
            now
            - self.started
            + cold * (self.epoch - step)
            + cold * self.policy["warm_fraction"] * (self.steps - self.epoch)
            + self.policy["prediction_reserve_s"]
        )
        self.over = self.over + 1 if forecast > self.policy["target_s"] else 0
        if self.over >= 2:
            raise TimeoutError(
                f"relflow projects {forecast / 3600:.2f} hours at update {step}."
            )


class PredictionDeadline(lit.Callback):
    """Include materialization and final predictions in the outer runtime budget."""

    def __init__(self, deadline: float) -> None:
        """Use the absolute deadline established when fitting began."""
        self.deadline = deadline

    def check(self) -> None:
        """Require the refit and prediction to finish within the declared budget."""
        if time.monotonic() >= self.deadline:
            raise TimeoutError("relflow exhausted its outer fit and prediction budget.")

    def on_predict_batch_start(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Check before starting the next prediction batch."""
        self.check()
