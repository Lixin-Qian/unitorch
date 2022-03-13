# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericOutputs


class SeNetProcessor(object):
    def __init__(
        self,
        pixel_mean: List[float] = [0.4039, 0.4549, 0.4823],
        pixel_std: List[float] = [1.0, 1.0, 1.0],
        resize_shape: Optional[List[int]] = [224, 224],
        crop_shape: Optional[List[int]] = [224, 224],
    ):
        """
        Args:
            pixel_mean: pixel means to process image
            pixel_std: pixel std to process image
            resize_shape: shape to resize image
            crop_shape: shape to crop image
        """
        super().__init__()
        self.pixel_mean = torch.tensor(pixel_mean)
        self.pixel_std = torch.tensor(pixel_std)
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.image_transform = Compose(
            [
                Resize(self.resize_shape),
                CenterCrop(self.crop_shape),
                ToTensor(),
                Normalize(self.pixel_mean, self.pixel_std),
            ]
        )

    def processing(self, image: Image.Image):

        return GenericOutputs(image=self.image_transform(image))
