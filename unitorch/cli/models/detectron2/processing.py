# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.detectron2 import (
    GeneralizedRCNNProcessor as _GeneralizedRCNNProcessor,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    ListInputs,
    BaseOutputs,
    DetectionOutputs,
    DetectionTargets,
)
from unitorch.cli.models.detectron2 import pretrained_generalized_rcnn_infos


class GeneralizedRCNNProcessor(_GeneralizedRCNNProcessor):
    def __init__(
        self,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__(
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )

    @classmethod
    @add_default_section_for_init("core/process/generalized_rcnn")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("core/process/generalized_rcnn_detection")
    def _processing_detection(
        self,
        image,
        bboxes,
        classes,
    ):
        outputs = super().processing_detection(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )
        return ListInputs(
            images=outputs.image,
            bboxes=outputs.bboxes,
            classes=outputs.classes,
        )

    @register_process("core/process/generalized_rcnn_image")
    def _processing_image(
        self,
        image,
    ):
        outputs = super().processing_image(
            image=image,
        )
        return ListInputs(
            images=outputs.image,
        )

    @register_process("core/process/generalized_rcnn_detection_evaluation")
    def _processing_detection_evaluation(self, image, bboxes, classes):
        outputs = super().processing_detection(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )
        return ListInputs(images=outputs.image), DetectionTargets(
            bboxes=outputs.bboxes,
            classes=outputs.classes,
        )

    @register_process("core/postprocess/generalized_rcnn")
    def _processing_generalized_rcnn(self, outputs: DetectionOutputs):
        _infos = outputs.to_dict()
        bboxes = [b.tolist() for b in _infos.pop("bboxes")]
        scores = [s.tolist() for s in _infos.pop("scores")]
        classes = [c.tolist() for c in _infos.pop("classes")]
        _infos.pop("features")
        return BaseOutputs(
            **_infos,
            bboxes=bboxes,
            scores=scores,
            classes=classes,
        )
