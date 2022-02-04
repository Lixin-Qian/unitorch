# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from detectron2.config import get_cfg
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import EventStorage
from detectron2.modeling import (
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
)
from detectron2.modeling.roi_heads import (
    Res5ROIHeads,
    StandardROIHeads,
    CascadeROIHeads,
    RROIHeads,
)
from unitorch.models import GenericModel, GenericOutputs


class GeneralizedRCNN(GenericModel):
    def __init__(self, detectron2_config_path):
        super().__init__()
        config = get_cfg()
        config.merge_from_file(detectron2_config_path)
        self.config = config
        self.backbone = build_backbone(config)
        self.proposal_generator = build_proposal_generator(
            config, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(config, self.backbone.output_shape())
        self.init_weights()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ):
        """
        images: list of image tensor
        bboxes: list of boxes tensor
        classes: list of classes tensor
        """
        if isinstance(images, torch.Tensor):
            assert images.dim() == 4
            _images = ImageList(
                images, [(images.size(-2), images.size(-1))] * images.size(0)
            )
        else:
            _images = ImageList.from_tensors(images)
        _instances = [
            Instances(
                image_size=_images.image_sizes[i],
                gt_boxes=Boxes(bboxes[i]),
                gt_classes=classes[i],
            )
            for i in range(len(bboxes))
        ]
        with EventStorage() as storage:
            _features = self.backbone(_images.tensor.to(self.dtype))
            _proposals, _rpn_loss = self.proposal_generator(
                _images, _features, _instances
            )
            _, _roi_loss = self.roi_heads(_images, _features, _proposals, _instances)
        _loss = torch.tensor(0.0).to(self.device)
        for k, v in {**_rpn_loss, **_roi_loss}.items():
            _loss += v
        return _loss

    def get_box_features(self, features, proposals):
        if isinstance(self.roi_heads, Res5ROIHeads):
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.roi_heads._shared_roi_transform(
                [features[f] for f in self.roi_heads.in_features], proposal_boxes
            )
            predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))
            _, prediction_indexes = self.roi_heads.box_predictor.inference(
                predictions, proposals
            )
            num_proposals = [0] + [len(prop) for prop in proposals]
            num_proposals = list(accumulate(num_proposals))
            return [
                box_features[num_proposals[i] : num_proposals[i + 1]][
                    prediction_indexes[i]
                ]
                for i in range(len(prediction_indexes))
            ]

        if isinstance(self.roi_heads, StandardROIHeads):
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.roi_heads.box_pooler(
                [features[f] for f in self.roi_heads.box_in_features], proposal_boxes
            )
            box_features = self.roi_heads.box_head(box_features)
            predictions = self.roi_heads.box_predictor(box_features)
            _, prediction_indexes = self.roi_heads.box_predictor.inference(
                predictions, proposals
            )
            num_proposals = [0] + [len(prop) for prop in proposals]
            num_proposals = list(accumulate(num_proposals))
            return [
                box_features[num_proposals[i] : num_proposals[i + 1]][
                    prediction_indexes[i]
                ]
                for i in range(len(prediction_indexes))
            ]

        raise "ROIHead boxes feature extraction is not supported now."

    @torch.no_grad()
    def detect(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        return_features: bool = False,
        norm_bboxes: bool = False,
    ):
        """
        images: list of image tensor
        """
        if isinstance(images, torch.Tensor):
            assert images.dim() == 4
            _images = ImageList(
                images, [(images.size(-2), images.size(-1))] * images.size(0)
            )
        else:
            _images = ImageList.from_tensors(images)
        _features = self.backbone(_images.tensor.to(self.dtype))
        _proposals, _ = self.proposal_generator(_images, _features)
        _instances, _ = self.roi_heads(_images, _features, _proposals)

        bboxes = [_instance.pred_boxes.tensor for _instance in _instances]
        scores = [_instance.scores for _instance in _instances]
        classes = [_instance.pred_classes for _instance in _instances]

        if norm_bboxes:
            sizes = _images.image_sizes
            bboxes = [
                b / torch.tensor([s[1], s[0], s[1], s[0]]).to(b)
                for b, s in zip(bboxes, sizes)
            ]

        outputs = dict(
            {
                "bboxes": bboxes,
                "scores": scores,
                "classes": classes,
            }
        )

        if return_features:
            outputs["features"] = self.get_box_features(_features, _proposals)

        return GenericOutputs(outputs)
