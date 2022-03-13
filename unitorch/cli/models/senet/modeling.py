# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.models.senet import SeResNet
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from transformers.file_utils import is_remote_url
from unitorch.cli.models.senet import pretrained_senet_infos
from torch.hub import load_state_dict_from_url


@register_model("core/model/classification/senet")
class SeNetForImageClassification(SeResNet):
    def __init__(
        self,
        arch: str,
        num_class: int,
        progress: bool,
        pretrain_weight_path=None,
        freeze_base_model=False,
    ):
        super().__init__(arch=arch, num_class=num_class)
        self.progress = progress
        self.pretrain_weight_path = pretrain_weight_path

        if freeze_base_model:
            print("freeze base model of ", arch)
            for n, p in self.named_parameters():
                if n in ["model.fc.weight", "model.fc.bias"] or (len(n.split(".")) > 3 and n.split(".")[3] == "se"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    @classmethod
    @add_default_section_for_init("core/model/classification/senet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/senet")
        arch = config.getoption("arch", "resnet50")
        num_class = config.getoption("num_class", 1000)
        progress = config.getoption("progress", True)
        pretrain_weight_path = config.getoption("pretrain_weight_path", "resnet50")
        freeze_base_model = config.getoption("freeze_base_model", False)

        inst = cls(
            arch=arch,
            num_class=num_class,
            progress=progress,
            pretrain_weight_path=pretrain_weight_path,
            freeze_base_model=freeze_base_model,
        )

        inst.from_pretrained(pretrain_weight_path)

        return inst

    def from_pretrained(self, pretrain_weight_path):
        if not (pretrain_weight_path in pretrained_senet_infos or os.path.exists(pretrain_weight_path)):
            return

        state_dict = (
            load_state_dict_from_url(pretrained_senet_infos[pretrain_weight_path], progress=self.progress)
            if pretrain_weight_path in pretrained_senet_infos
            else torch.load(pretrain_weight_path, map_location="cpu")
        )

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")

            if new_key:
                new_key = "model." + new_key
            else:
                new_key = "model." + key
            old_keys.append(key)
            new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }
        self.load_state_dict(state_dict, False)
        logging.info(f"{type(self).__name__} model load weight from pretrain {pretrain_weight_path}")

    @autocast()
    def forward(
        self,
        image_input=None,
    ):
        outputs = super().forward(image_input=image_input)
        return ClassificationOutputs(outputs=outputs)
