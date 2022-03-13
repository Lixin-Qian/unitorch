# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import ViTFeatureExtractor
from unitorch.functions import pop_first_non_none_value
from unitorch.models import GenericOutputs


class SwinProcessor(object):
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Args:
            vision_config_path: vision config path to swin processor
        """
        self.vision_processor = ViTFeatureExtractor.from_json_file(vision_config_path)

        self.size = self.vision_processor.size
        self.resample = self.vision_processor.resample
        self.image_mean = self.vision_processor.image_mean
        self.image_std = self.vision_processor.image_std

    def processing_image_classifictaion(
        self,
        image: Image.Image,
    ):
        """
        Args:
            image: input image
        """
        image = self.vision_processor.resize(
            image=image,
            size=self.size,
            resample=self.resample,
        )
        image = self.vision_processor.normalize(
            image=image,
            mean=self.image_mean,
            std=self.image_std,
        )

        return GenericOutputs(
            image=torch.tensor(image),
        )
