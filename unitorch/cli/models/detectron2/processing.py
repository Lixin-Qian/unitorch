# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.detectron2 import (
    GeneralizedProcessor as _GeneralizedProcessor,
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
from unitorch.cli.models.detectron2 import pretrained_detectron2_infos


class GeneralizedProcessor(_GeneralizedProcessor):
    def __init__(
        self,
        pixel_mean: List[float],
        pixel_std: List[float],
        resize_shape: Optional[List[int]] = None,
        min_size_test: Optional[int] = None,
        max_size_test: Optional[int] = None,
    ):
        super().__init__(
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            resize_shape=resize_shape,
            min_size_test=min_size_test,
            max_size_test=max_size_test,
        )

    @classmethod
    @add_default_section_for_init("core/process/detectron2/generalized")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("core/process/detectron2/generalized_detection")
    def _processing_detection(
        self,
        image: Union[Image.Image, str],
        bboxes: List[List[int]],
        classes: List[int],
        do_eval: Optional[bool] = False,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_detection(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )
        if do_eval:
            return ListInputs(images=outputs.image), DetectionTargets(
                bboxes=outputs.bboxes,
                classes=outputs.classes,
            )

        return ListInputs(
            images=outputs.image,
            bboxes=outputs.bboxes,
            classes=outputs.classes,
        )

    @register_process("core/process/detectron2/generalized_image")
    def _processing_image(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_image(
            image=image,
        )
        return ListInputs(
            images=outputs.image,
        )

    @register_process("core/postprocess/detectron2/generalized_detection")
    def _processing_generalized_detection(self, outputs: DetectionOutputs):
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
