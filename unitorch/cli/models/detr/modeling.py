# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch import hf_cached_path
from unitorch.models.detr import (
    DetrForDetection as _DetrForDetection,
    DetrForSegmentation as _DetrForSegmentation,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import (
    detection_model_decorator,
    segmentation_model_decorator,
    DetectionOutputs,
    SegmentationOutputs,
    LossOutputs,
)
from unitorch.cli.models.detr import pretrained_detr_infos


@register_model("core/model/detection/detr", detection_model_decorator)
class DetrForDetection(_DetrForDetection):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = None,
    ):
        super().__init__(
            config_path=config_path,
            num_class=num_class,
        )

    @classmethod
    @add_default_section_for_init("core/model/detection/detr")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/detection/detr")
        pretrained_name = config.getoption("pretrained_name", "default-detr")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_detr_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_detr_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)
        num_class = config.getoption("num_class", None)

        inst = cls(
            config_path=config_path,
            num_class=num_class,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_detr_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_detr_infos
                and "weight" in pretrained_detr_infos[pretrained_name_or_path]
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

    @add_default_section_for_function("core/model/detection/detr")
    def detect(
        self,
        images,
        norm_bboxes: bool = False,
    ):
        outputs = super().detect(
            images=images,
            norm_bboxes=norm_bboxes,
        )
        return DetectionOutputs(
            bboxes=outputs.bboxes,
            scores=outputs.scores,
            classes=outputs.classes,
        )


@register_model("core/model/segmentation/detr", segmentation_model_decorator)
class DetrForSegmentation(_DetrForSegmentation):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = None,
        enable_bbox_loss: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            num_class=num_class,
            enable_bbox_loss=enable_bbox_loss,
        )

    @classmethod
    @add_default_section_for_init("core/model/segmentation/detr")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/segmentation/detr")
        pretrained_name = config.getoption("pretrained_name", "default-detr")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_detr_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_detr_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)
        num_class = config.getoption("num_class", None)
        enable_bbox_loss = config.getoption("enable_bbox_loss", False)

        inst = cls(
            config_path=config_path,
            num_class=num_class,
            enable_bbox_loss=enable_bbox_loss,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_detr_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_detr_infos
                and "weight" in pretrained_detr_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        images,
        masks,
        bboxes,
        classes,
    ):
        outputs = super().forward(
            images=images,
            bboxes=bboxes,
            classes=classes,
            masks=masks,
        )
        return LossOutputs(loss=outputs)

    @add_default_section_for_function("core/model/segmentation/detr")
    def segment(
        self,
        images,
        norm_bboxes: bool = False,
    ):
        outputs = super().segment(
            images=images,
            norm_bboxes=norm_bboxes,
        )
        return SegmentationOutputs(
            bboxes=outputs.bboxes,
            scores=outputs.scores,
            classes=outputs.classes,
            masks=outputs.masks,
        )
