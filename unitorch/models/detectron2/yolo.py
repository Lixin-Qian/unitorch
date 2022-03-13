# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict
from transformers.file_utils import is_remote_url

from detectron2.config import CfgNode, get_cfg
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
from unitorch import hf_cached_path
from unitorch.models import GenericModel, GenericOutputs


def _add_yolo_config(cfg):
    cfg.MODEL.YOLOV5 = CfgNode()
    cfg.MODEL.YOLOV5.FOCUS = True
    cfg.MODEL.YOLOV5.VERSION = "m"
    cfg.MODEL.YOLOV5.NUM_CLASSES = 80
    cfg.MODEL.YOLOV5.ANCHORS = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    cfg.MODEL.YOLOV5.FOCAL_LOSS_GAMMA = 0.0
    cfg.MODEL.YOLOV5.BOX_LOSS_GAIN = 0.05
    cfg.MODEL.YOLOV5.CLS_LOSS_GAIN = 0.3
    cfg.MODEL.YOLOV5.CLS_POSITIVE_WEIGHT = 1.0
    cfg.MODEL.YOLOV5.OBJ_LOSS_GAIN = 0.7
    cfg.MODEL.YOLOV5.OBJ_POSITIVE_WEIGHT = 1.0
    cfg.MODEL.YOLOV5.LABEL_SMOOTHING = 0.0
    cfg.MODEL.YOLOV5.ANCHOR_T = 4.0
    cfg.MODEL.YOLOV5.CONF_THRESH = 0.001
    cfg.MODEL.YOLOV5.IOU_THRES = 0.65
    cfg.MODEL.PIXEL_MEAN: [0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD: [255.0, 255.0, 255.0]
    cfg.INPUT.SIZE = 416
    cfg.INPUT.HSV_H = 0.015
    cfg.INPUT.HSV_S = 0.7
    cfg.INPUT.HSV_V = 0.4
    cfg.INPUT.DEGREES = 0.0
    cfg.INPUT.TRANSLATE = 0.1
    cfg.INPUT.SCALE = 0.5
    cfg.INPUT.SHEAR = 0.0
    cfg.INPUT.PERSPECTIVE = 0.0
    cfg.INPUT.FLIPUD = 0.0
    cfg.INPUT.FLIPLR = 0.5
    cfg.INPUT.MOSAIC = 1.0  # IGNORED
    cfg.INPUT.MIXUP = 0.0
    cfg.INPUT.FORMAT = "BGR"
    cfg.TEST.AUG.SIZE = 416


class YoloForDetection(GenericModel):
    def __init__(self, detectron2_config_path: str):
        """
        Args:
            detectron2_config_path: config file path to generalized yolo model
        """
        super().__init__()
        config = get_cfg()
        _add_yolo_config(config)
        config.merge_from_file(detectron2_config_path)
        self.config = config
        self.yolo = build_model(config)
        self.init_weights()

    @property
    def dtype(self):
        """
        `torch.dtype`: which dtype the parameters are (assuming that all the parameters are the same dtype).
        """
        return next(self.parameters()).dtype

    @property
    def device(self):
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        """
        return next(self.parameters()).device

    def from_pretrained(
        self,
        weight_path: str,
        replace_keys: Dict = OrderedDict(),
        **kwargs,
    ):
        """
        Load model's pretrained weight
        Args:
            weight_path: the path of pretrained weight of mbart
        """
        if weight_path is None:
            return

        if not (is_remote_url(weight_path) or os.path.exists(weight_path)):
            return

        weight_path = hf_cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")

        _self_state_dict = self.yolo.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        self.load_state_dict(state_dict, False)
        logging.info(f"{type(self).__name__} model load weight from pretrain {weight_path}")

    def forward(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ):
        """
        Args:
            images: list of image tensor
            bboxes: list of boxes tensor
            classes: list of classes tensor
        """
        if isinstance(images, torch.Tensor):
            assert images.dim() == 4
            _images = ImageList(images, [(images.size(-2), images.size(-1))] * images.size(0))
        else:
            _images = ImageList.from_tensors(images, self.yolo.backbone.size_divisibility)

        _instances = [
            Instances(
                image_size=_images.image_sizes[i],
                gt_boxes=Boxes(bboxes[i]),
                gt_classes=classes[i],
            )
            for i in range(len(bboxes))
        ]
        with EventStorage() as storage:
            _features = self.yolo.backbone(_images.tensor.to(self.dtype))
            _features = list(_features.values())

            _pred = self.yolo.head(_features)
            _losses = self.yolo.loss(_pred, _instances)

        _loss = torch.tensor(0.0).to(self.device)
        for k, v in _losses.items():
            _loss += v.sum()
        return _loss

    @torch.no_grad()
    def detect(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        norm_bboxes: Optional[bool] = False,
    ):
        """
        Args:
            images: list of image tensor
        """
        if isinstance(images, torch.Tensor):
            assert images.dim() == 4
            _images = ImageList(images, [(images.size(-2), images.size(-1))] * images.size(0))
        else:
            _images = ImageList.from_tensors(images, self.yolo.backbone.size_divisibility)

        _features = self.yolo.backbone(_images.tensor.to(self.dtype))
        _features = list(_features.values())
        _pred = self.yolo.head(_features)
        _instances = self.yolo.inference(_pred, _images.image_sizes)

        bboxes = [_instance.pred_boxes.tensor for _instance in _instances]
        scores = [_instance.scores for _instance in _instances]
        classes = [_instance.pred_classes for _instance in _instances]

        if norm_bboxes:
            sizes = _images.image_sizes
            bboxes = [b / torch.tensor([s[1], s[0], s[1], s[0]]).to(b) for b, s in zip(bboxes, sizes)]

        outputs = dict(
            {
                "bboxes": bboxes,
                "scores": scores,
                "classes": classes,
            }
        )

        return GenericOutputs(outputs)
