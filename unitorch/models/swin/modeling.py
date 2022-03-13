# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.swin import SwinConfig, SwinModel
from unitorch.models import GenericModel


class SwinForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = 1,
    ):
        """
        Args:
            config_path: config file path to swin model
            num_class: num class to classification
        """
        super().__init__()
        config = SwinConfig.from_json_file(config_path)

        self.swin = SwinModel(config)
        self.classifier = nn.Linear(self.swin.num_features, num_class)
        self.init_weights()

    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        Args:
            pixel_values: pixels of image
        """
        outputs = self.swin(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        return self.classifier(pooled_output)
