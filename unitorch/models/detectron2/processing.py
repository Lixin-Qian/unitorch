# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data.transforms import ResizeShortestEdge
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from unitorch.models import GenericOutputs
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class GeneralizedProcessor(object):
    def __init__(
        self,
        pixel_mean: List[float],
        pixel_std: List[float],
        resize_shape: List[int] = [224, 224],
        min_size_test: Optional[int] = 800,
        max_size_test: Optional[int] = 1333,
    ):
        """
        Args:
            pixel_mean: pixel means to process image
            pixel_std: pixel std to process image
            resize_shape: shape to resize image
            min_size_test: resize shortest edge parameters
            max_size_test: resize shortest edge parameters
        """
        self.pixel_mean = torch.tensor(pixel_mean)
        self.pixel_std = torch.tensor(pixel_std)
        self.process_resize = Resize(resize_shape) if resize_shape is not None else None
        self.process_resize_shortest_edge = (
            ResizeShortestEdge([min_size_test, min_size_test], max_size_test)
            if (min_size_test is not None and max_size_test is not None)
            else None
        )

    def processing_transform(
        self,
        image: Image.Image,
    ):
        """
        Args:
            image: input image
        """
        height, width = image.size
        if self.process_resize is not None:
            image = self.process_resize(image)
        image = convert_PIL_to_numpy(image, "BGR")
        if self.process_resize_shortest_edge is not None:
            image = self.process_resize_shortest_edge.get_transform(image).apply_image(image)

        image = image.copy()

        image = torch.tensor(image).float()
        image = image - self.pixel_mean
        image = image / self.pixel_std

        image = image.permute(2, 0, 1)

        return GenericOutputs(
            image=torch.tensor(image).float(),
            sizes=torch.tensor([height, width]),
        )

    def processing_detection(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        classes: List[int],
    ):
        """
        Args:
            image: input image
            bboxes: bboxes to image
            classes: class to each bbox
        """
        outputs = self.processing_transform(image)
        new_h, new_w = outputs.image.size()[1:]
        org_h, org_w = outputs.sizes
        image = outputs.image
        bboxes = torch.tensor(bboxes).float()
        bboxes[:, 0] = bboxes[:, 0] * (new_w / org_w)
        bboxes[:, 1] = bboxes[:, 1] * (new_h / org_h)
        bboxes[:, 2] = bboxes[:, 2] * (new_w / org_w)
        bboxes[:, 3] = bboxes[:, 3] * (new_h / org_h)
        bboxes = torch.tensor(bboxes)
        classes = torch.tensor(classes)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)

        assert bboxes.size(-1) == 4 and classes.dim() == 1 and len(classes) == len(bboxes)
        return GenericOutputs(image=image, bboxes=bboxes, classes=classes)

    def processing_image(self, image: Image.Image):
        """
        Args:
            image: input image
        """
        outputs = self.processing_transform(image)
        return GenericOutputs(image=outputs.image)
