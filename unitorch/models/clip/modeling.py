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

from transformers.models.clip.modeling_clip import (
    CLIPConfig,
    CLIPTextTransformer,
    CLIPVisionTransformer,
)
from unitorch.models import GenericModel


def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = _contrastive_loss(similarity)
    image_loss = _contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class CLIPForPretrain(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Args:
            config_path: config file path to clip model
            projection_dim: dimension to image/text output embedding
            num_class: num class to classification
            freeze_base_model: if to freeze base model
            gradient_checkpointing: if to enable gradient_checkpointing
        """
        super().__init__()

        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        Args:
            input_ids: tokens of text
            pixel_values: pixels of image
            attention_mask: attention mask of tokens
            position_ids: position ids
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        return _clip_loss(logits_per_text)


class CLIPForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_class: int = 1,
        freeze_base_model=True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Args:
            config_path: config file path to clip model
            projection_dim: dimension to image/text output embedding
            num_class: num class to classification
            freeze_base_model: if to freeze base model
            gradient_checkpointing: if to enable gradient_checkpointing
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(self.projection_dim * 2, num_class)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        Args:
            input_ids: tokens of text
            pixel_values: pixels of image
            attention_mask: attention mask of tokens
            position_ids: position ids
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return self.classifier(F.relu(torch.cat([image_embeds, text_embeds], axis=1)))


class CLIPForTextClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_class: int = 1,
        freeze_base_model=True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Args:
            config_path: config file path to clip model
            projection_dim: dimension to image/text output embedding
            num_class: num class to classification
            freeze_base_model: if to freeze base model
            gradient_checkpointing: if to enable gradient_checkpointing
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        text_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(self.projection_dim, num_class)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        Args:
            input_ids: tokens of text
            attention_mask: attention mask of tokens
            position_ids: position ids
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return self.classifier(F.relu(text_embeds))


class CLIPForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_class: int = 1,
        freeze_base_model=True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Args:
            config_path: config file path to clip model
            projection_dim: dimension to image/text output embedding
            num_class: num class to classification
            freeze_base_model: if to freeze base model
            gradient_checkpointing: if to enable gradient_checkpointing
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.classifier = nn.Linear(self.projection_dim, num_class)
        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

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
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return self.classifier(F.relu(image_embeds))
