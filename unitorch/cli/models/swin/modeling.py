# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch import hf_cached_path
from unitorch.models.swin import (
    SwinForImageClassification as _SwinForImageClassification,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.swin import pretrained_swin_infos


@register_model("core/model/classification/swin")
class SwinForImageClassification(_SwinForImageClassification):
    def __init__(
        self,
        config_path: str,
        num_class: int = 1,
    ):
        super().__init__(
            config_path=config_path,
            num_class=num_class,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/swin")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/swin")
        pretrained_name = config.getoption("pretrained_name", "default-swin")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_swin_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_swin_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)
        num_class = config.getoption("num_class", 1)

        inst = cls(
            config_path=config_path,
            num_class=num_class,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_swin_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_swin_infos
                and "weight" in pretrained_swin_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        pixel_values=None,
    ):
        outputs = super().forward(pixel_values=pixel_values)
        return ClassificationOutputs(outputs=outputs)
