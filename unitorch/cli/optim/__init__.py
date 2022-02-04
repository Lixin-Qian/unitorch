# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging
from unitorch.optim import SGD, Adam, AdamW
from unitorch.cli import add_default_section_for_init, register_optim


class OptimMixin(object):
    def save_checkpoint(
        self,
        ckpt_dir: str = ".",
        weight_name="pytorch_optim.bin",
    ):
        state_dict = self.state_dict()
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(state_dict, weight_path)
        logging.info(
            f"{type(self).__name__} optimizer save checkpoint to {weight_path}"
        )

    def from_checkpoint(
        self,
        ckpt_dir: str = ".",
        weight_name="pytorch_optim.bin",
    ):
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict)
        logging.info(
            f"{type(self).__name__} optimizer load weight from checkpoint {weight_path}"
        )


@register_optim("core/optim/sgd")
class SGDOptimizer(SGD, OptimMixin):
    def __init__(
        self,
        params,
        learning_rate: float = 0.00001,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
        )

    @classmethod
    @add_default_section_for_init("core/optim/sgd")
    def from_core_configure(cls, config, **kwargs):
        pass


@register_optim("core/optim/adam")
class AdamOptimizer(Adam, OptimMixin):
    def __init__(
        self,
        params,
        learning_rate: float = 0.00001,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
        )

    @classmethod
    @add_default_section_for_init("core/optim/adam")
    def from_core_configure(cls, config, **kwargs):
        pass


@register_optim("core/optim/adamw")
class AdamWOptimizer(AdamW, OptimMixin):
    def __init__(
        self,
        params,
        learning_rate: float = 0.00001,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
        )

    @classmethod
    @add_default_section_for_init("core/optim/adamw")
    def from_core_configure(cls, config, **kwargs):
        pass
