# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.models.detectron2 import YoloForDetection as _YoloForDetection
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import detection_model_decorator, DetectionOutputs, LossOutputs
from unitorch.cli.models.detectron2 import pretrained_detectron2_infos


@register_model("core/model/detection/yolo", detection_model_decorator)
class YoloForDetection(_YoloForDetection):
    def __init__(self, detectron2_config_path):
        super().__init__(detectron2_config_path=detectron2_config_path)

    @classmethod
    @add_default_section_for_init("core/model/detection/yolo")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/detection/yolo")
        pretrained_name = config.getoption("pretrained_name", "default-yolo")
        config_name_or_path = config.getoption("config_name_or_path", pretrained_name)
        config_path = (
            pretrained_detectron2_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_detectron2_infos
            and "config" in pretrained_detectron2_infos[config_name_or_path]
            else config_name_or_path
        )

        config_path = cached_path(config_path)

        inst = cls(config_path)
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption("pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_detectron2_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_detectron2_infos
                and "weight" in pretrained_detectron2_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(self, images, bboxes, classes):
        outputs = super().forward(
            images=images,
            bboxes=bboxes,
            classes=classes,
        )
        return LossOutputs(loss=outputs)

    @add_default_section_for_function("core/model/detection/yolo")
    def detect(
        self,
        images,
        norm_bboxes=False,
    ):
        outputs = super().detect(
            images=images,
            norm_bboxes=norm_bboxes,
        )

        return DetectionOutputs(
            bboxes=outputs.bboxes,
            scores=outputs.scores,
            features=None,
            classes=outputs.classes,
        )
