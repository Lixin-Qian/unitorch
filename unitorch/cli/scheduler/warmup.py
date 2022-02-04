# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch.optim as optim
from unitorch.scheduler.warmup import (
    CosineWarmupScheduler,
    LinearWarmupScheduler,
)
from unitorch.cli import add_default_section_for_init, register_scheduler
from unitorch.cli.scheduler import SchedulerMixin


@register_scheduler("core/scheduler/cosine_warmup")
class CosineWarmupScheduler(CosineWarmupScheduler, SchedulerMixin):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        super().__init__(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    @classmethod
    @add_default_section_for_init("core/scheduler/cosine_warmup")
    def from_core_configure(cls, config, **kwargs):
        num_warmup_steps = config.getdefault(
            "core/scheduler/cosine_warmup", "num_warmup_steps", -1
        )
        num_warmup_rate = config.getdefault(
            "core/scheduler/cosine_warmup", "num_warmup_rate", 0.001
        )
        num_cycles = config.getdefault(
            "core/scheduler/cosine_warmup", "num_cycles", 0.5
        )
        num_training_steps = kwargs.get("num_training_steps", 1000000)
        if num_warmup_steps < 0:
            num_warmup_steps = int(num_training_steps * num_warmup_rate)
        return dict(
            {
                "num_warmup_steps": num_warmup_steps,
                "num_training_steps": num_training_steps,
                "num_cycles": num_cycles,
            }
        )


@register_scheduler("core/scheduler/linear_warmup")
class LinearWarmupScheduler(LinearWarmupScheduler, SchedulerMixin):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        super().__init__(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    @classmethod
    @add_default_section_for_init("core/scheduler/linear_warmup")
    def from_core_configure(cls, config, **kwargs):
        num_warmup_steps = config.getdefault(
            "core/scheduler/linear_warmup", "num_warmup_steps", -1
        )
        num_warmup_rate = config.getdefault(
            "core/scheduler/linear_warmup", "num_warmup_rate", 0.001
        )
        num_training_steps = kwargs.get("num_training_steps", 1000000)
        if num_warmup_steps < 0:
            num_warmup_steps = int(num_training_steps * num_warmup_rate)
        return dict(
            {
                "num_warmup_steps": num_warmup_steps,
                "num_training_steps": num_training_steps,
            }
        )
