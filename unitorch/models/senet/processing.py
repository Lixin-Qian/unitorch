# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from unitorch.models import GenericOutputs


class SeNetProcessor(object):
    def __init__(
        self,
        pixel_mean=[0.4039, 0.4549, 0.4823],
        pixel_std=[1.0, 1.0, 1.0],
        resize_shape=[224, 224],
        crop_shape=[224, 224],
    ):
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
