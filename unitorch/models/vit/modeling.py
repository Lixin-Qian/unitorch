# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.vit import ViTConfig, ViTModel
from unitorch.models import GenericModel


class ViTForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: int = 1,
    ):
        super().__init__()
        config = ViTConfig.from_json_file(config_path)

        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_class)
        self.init_weights()

    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        vision_outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = vision_outputs[1]
        return self.classifier(pooled_output)
