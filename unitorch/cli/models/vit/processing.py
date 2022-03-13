# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.models.vit import ViTProcessor as _ViTProcessor
from unitorch.cli import cached_path
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
from unitorch.cli.models.vit import pretrained_vit_infos


class ViTProcessor(_ViTProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/vit")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/vit")
        pretrained_name = config.getoption("pretrained_name", "default-vit")
        vision_config_name_or_path = config.getoption("vision_config_path", pretrained_name)
        vision_config_path = (
            pretrained_vit_infos[vision_config_name_or_path]["vision_config"]
            if vision_config_name_or_path in pretrained_vit_infos
            else vision_config_name_or_path
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/vit_image_classification")
    def _processing_image_classifictaion(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().processing_image_classifictaion(image=image)
        return BaseInputs(pixel_values=outputs.image)
