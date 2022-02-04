# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.models.senet import SeNetProcessor as _SeNetProcessor
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    BaseInputs,
    BaseOutputs,
    BaseTargets,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.senet import pretrained_senet_infos


class SeNetProcessor(_SeNetProcessor):
    def __init__(
        self,
        pixel_mean=[103.53, 116.28, 123.675],
        pixel_std=[1.0, 1.0, 1.0],
        resize_shape=[224, 224],
        crop_shape=[224, 224],
    ):
        super().__init__(
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
        )

    @classmethod
    @add_default_section_for_init("core/process/senet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/senet")
        pixel_mean = config.getoption("pixel_mean", [0.4039, 0.4549, 0.4823])
        pixel_std = config.getoption("pixel_std", [1.0, 1.0, 1.0])
        resize_shape = config.getoption("resize_shape", [224, 224])
        crop_shape = config.getoption("crop_shape", [224, 224])

        return {
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "resize_shape": resize_shape,
            "crop_shape": crop_shape,
        }

    @register_process("core/process/senet_classification")
    def _processing(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().processing(image=image)
        return BaseInputs(image_input=outputs.image)
