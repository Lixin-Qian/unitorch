# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.models.detr import DetrProcessor as _DetrProcessor
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
    ListInputs,
    DetectionOutputs,
    DetectionTargets,
    SegmentationOutputs,
    SegmentationTargets,
)
from unitorch.cli.models.detr import pretrained_detr_infos


class DetrProcessor(_DetrProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/detr")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/detr")
        pretrained_name = config.getoption("pretrained_name", "default-detr")
        vision_config_name_or_path = config.getoption("vision_config_path", pretrained_name)
        vision_config_path = (
            pretrained_detr_infos[vision_config_name_or_path]["vision_config"]
            if vision_config_name_or_path in pretrained_detr_infos
            else vision_config_name_or_path
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/detr_image")
    def _processing_image(
        self,
        image: Union[Image.Image, str],
    ):
        outputs = super().processing_image(
            image=image,
        )
        return ListInputs(
            images=outputs.image,
        )

    @register_process("core/process/detr_detection")
    def _processing_detection(
        self,
        image: Union[Image.Image, str],
        bboxes: List[List[float]],
        classes: List[int],
        do_eval: Optional[bool] = False,
    ):
        outputs = super().processing_detection(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )
        if do_eval:
            new_h, new_w = outputs.image.size()[1:]
            bboxes = outputs.bboxes
            bboxes[:, 0] = bboxes[:, 0] * new_w
            bboxes[:, 1] = bboxes[:, 1] * new_h
            bboxes[:, 2] = bboxes[:, 2] * new_w
            bboxes[:, 3] = bboxes[:, 3] * new_h
            return ListInputs(images=outputs.image), DetectionTargets(
                bboxes=bboxes,
                classes=outputs.classes,
            )

        return ListInputs(
            images=outputs.image,
            bboxes=outputs.bboxes,
            classes=outputs.classes,
        )

    @register_process("core/postprocess/detr_detection")
    def _postprocessing_dectection(self, outputs: DetectionOutputs):
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

    @register_process("core/process/detr_segmentation")
    def _processing_segmentation(
        self,
        image: Union[Image.Image, str],
        gt_image: Union[Image.Image, str],
        bboxes: List[List[float]] = None,
        classes: List[int] = None,
        do_eval: Optional[bool] = False,
    ):
        if bboxes is not None and classes is not None:
            outputs1 = super().processing_detection(
                image=image,
                bboxes=bboxes,
                classes=classes,
            )
            bboxes = outputs1.bboxes
            classes = outputs1.classes
            if do_eval:
                new_h, new_w = outputs1.image.size()[1:]
                bboxes[:, 0] = bboxes[:, 0] * new_w
                bboxes[:, 1] = bboxes[:, 1] * new_h
                bboxes[:, 2] = bboxes[:, 2] * new_w
                bboxes[:, 3] = bboxes[:, 3] * new_h
        else:
            bboxes, classes = None, None

        outputs2 = super().processing_segmentation(
            image=image,
            gt_image=gt_image,
        )

        if do_eval:
            return ListInputs(images=outputs2.image,), SegmentationTargets(
                targets=outputs2.gt_image,
                bboxes=bboxes,
                classes=classes,
            )

        return ListInputs(
            images=outputs2.image,
            masks=outputs2.gt_image,
            bboxes=bboxes,
            classes=classes,
        )

    @register_process("core/postprocess/detr_segmentation")
    def _postprocessing_segmentation(self, outputs: SegmentationOutputs):
        _infos = outputs.to_dict()
        bboxes = [b.tolist() for b in _infos.pop("bboxes")]
        scores = [s.tolist() for s in _infos.pop("scores")]
        classes = [c.tolist() for c in _infos.pop("classes")]
        masks = [m.tolist() for m in _infos.pop("masks")]
        _infos.pop("features")
        return BaseOutputs(
            **_infos,
            bboxes=bboxes,
            scores=scores,
            classes=classes,
            masks=masks,
        )
