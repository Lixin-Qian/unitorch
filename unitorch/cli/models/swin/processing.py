# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch import hf_cached_path
from unitorch.models.swin import SwinProcessor as _SwinProcessor
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
from unitorch.cli.models.swin import pretrained_swin_infos


class SwinProcessor(_SwinProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/swin")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/swin")
        pretrained_name = config.getoption("pretrained_name", "default-swin")
        vision_config_name_or_path = config.getoption(
            "vision_config_path", pretrained_name
        )
        vision_config_path = (
            pretrained_swin_infos[vision_config_name_or_path]["vision_config"]
            if vision_config_name_or_path in pretrained_swin_infos
            else vision_config_name_or_path
        )

        vision_config_path = hf_cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/swin_image_classification")
    def _processing_image_classifictaion(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().processing_image_classifictaion(image=image)
        return BaseInputs(pixel_values=outputs.image)
