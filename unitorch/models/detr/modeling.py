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
from transformers.models.detr import DetrConfig
from transformers.models.detr.modeling_detr import (
    DetrModel,
    DetrMLPPredictionHead,
    DetrHungarianMatcher,
    DetrMaskHeadSmallConv,
    DetrMHAttentionMap,
    DetrLoss,
)
from detectron2.structures import ImageList
from unitorch.models import GenericModel, GenericOutputs


class DetrForDetection(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = None,
    ):
        super().__init__()
        """
        Args:
            config_path: config file path to detr model
            num_class: num class to classification
        """
        config = DetrConfig.from_json_file(config_path)

        self.model = DetrModel(config)
        if num_class is not None:
            config.num_labels = num_class
        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels + 1)
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3,
        )
        self.config = config
        self.init_weights()

        # modules for loss
        self.enable_auxiliary_loss = config.auxiliary_loss
        matcher = DetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
        )
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = DetrLoss(
            matcher=matcher,
            num_classes=config.num_labels,
            eos_coef=config.eos_coefficient,
            losses=losses,
        )
        self.criterion.to(self.device)
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
        }
        if self.enable_auxiliary_loss:
            aux_weight_dict = {}
            for i in range(config.decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)

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

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

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
            _images = ImageList.from_tensors(images)

        outputs = self.model(_images.tensor.to(self.dtype))
        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        outputs_loss = {}
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if self.enable_auxiliary_loss:
            intermediate = outputs[4]
            outputs_class = self.class_labels_classifier(intermediate)
            outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            auxiliary_outputs = self._set_aux_loss(
                outputs_class,
                outputs_coord,
            )
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        labels = [{"class_labels": c, "boxes": b} for b, c in zip(bboxes, classes)]
        loss_dict = self.criterion(outputs_loss, labels)
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return loss

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
            _images = ImageList.from_tensors(images)

        outputs = self.model(_images.tensor.to(self.dtype))
        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        logits = logits.softmax(dim=-1)

        scores, classes = list(zip(*[p.max(-1) for p in logits]))

        if not norm_bboxes:
            sizes = _images.image_sizes
            bboxes = [b * torch.tensor([s[1], s[0], s[1], s[0]]).to(b) for b, s in zip(pred_boxes, sizes)]
        else:
            bboxes = pred_boxes

        bboxes, scores, classes = list(
            zip(
                *[
                    (
                        b[c != self.config.num_labels],
                        s[c != self.config.num_labels],
                        c[c != self.config.num_labels],
                    )
                    for b, s, c in zip(bboxes, scores, classes)
                ]
            )
        )

        outputs = dict(
            {
                "bboxes": bboxes,
                "scores": scores,
                "classes": classes,
            }
        )
        return GenericOutputs(outputs)


class DetrForSegmentation(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = None,
        enable_bbox_loss: Optional[bool] = False,
    ):
        """
        Args:
            config_path: config file path to detr model
            num_class: num class to classification
            enable_bbox_loss: if enable bbox loss for segmentation
        """
        super().__init__()
        config = DetrConfig.from_json_file(config_path)

        self.model = DetrModel(config)
        if num_class is not None:
            config.num_labels = num_class

        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels + 1)
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3,
        )

        for p in self.model.parameters():
            p.requires_grad = False

        # segmentation head
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        intermediate_channel_sizes = self.model.backbone.conv_encoder.intermediate_channel_sizes

        self.mask_head = DetrMaskHeadSmallConv(
            hidden_size + number_of_heads,
            intermediate_channel_sizes[::-1][-3:],
            hidden_size,
        )

        self.bbox_attention = DetrMHAttentionMap(
            hidden_size,
            hidden_size,
            number_of_heads,
            dropout=0.0,
            std=config.init_xavier_std,
        )
        self.config = config
        self.init_weights()

        # modules for loss
        self.enable_auxiliary_loss = config.auxiliary_loss
        self.enable_bbox_loss = enable_bbox_loss

        matcher = DetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
        )
        losses = ["masks"]
        if self.enable_bbox_loss:
            losses += ["labels", "boxes", "cardinality"]
        self.criterion = DetrLoss(
            matcher=matcher,
            num_classes=config.num_labels,
            eos_coef=config.eos_coefficient,
            losses=losses,
        )
        self.criterion.to(self.device)
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
            "loss_mask": config.mask_loss_coefficient,
            "loss_dice": config.dice_loss_coefficient,
        }
        if self.enable_auxiliary_loss:
            aux_weight_dict = {}
            for i in range(config.decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)

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

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        masks: Union[List[torch.Tensor], torch.Tensor],
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ):
        """
        Args:
            images: list of image tensor
            masks: list of mask tensor
            bboxes: list of boxes tensor
            classes: list of classes tensor
        """
        if isinstance(images, torch.Tensor):
            assert images.dim() == 4
            _images = ImageList(images, [(images.size(-2), images.size(-1))] * images.size(0))
        else:
            _images = ImageList.from_tensors(images)

        batch_size, num_channels, height, width = _images.tensor.shape
        image_masks = torch.ones((batch_size, height, width), device=self.device)
        features, position_embeddings_list = self.model.backbone(
            _images.tensor.to(self.dtype),
            pixel_mask=image_masks,
        )

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_map, mask = features[-1]
        batch_size, num_channels, height, width = feature_map.shape
        projected_feature_map = self.model.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        position_embeddings = position_embeddings_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)
        encoder_outputs = self.model.encoder(
            inputs_embeds=flattened_features,
            attention_mask=flattened_mask,
            position_embeddings=position_embeddings,
        )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
        )

        sequence_output = decoder_outputs[0]

        # class logits + predicted bounding boxes
        outputs_loss = {}
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes

        memory = encoder_outputs[0].permute(0, 2, 1).view(batch_size, self.config.d_model, height, width)
        mask = flattened_mask.view(batch_size, height, width)

        bbox_mask = self.bbox_attention(sequence_output, memory, mask=~mask)
        seg_masks = self.mask_head(
            projected_feature_map,
            bbox_mask,
            [features[2][0], features[1][0], features[0][0]],
        )
        pred_masks = seg_masks.view(
            batch_size,
            self.config.num_queries,
            seg_masks.shape[-2],
            seg_masks.shape[-1],
        )
        outputs_loss["pred_masks"] = pred_masks

        if self.enable_auxiliary_loss:
            intermediate = outputs[4]
            outputs_class = self.class_labels_classifier(intermediate)
            outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            auxiliary_outputs = self._set_aux_loss(
                outputs_class,
                outputs_coord,
            )
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        labels = [
            {
                "class_labels": c,
                "boxes": b,
                "masks": m if m.dim() == 3 else m.unsqueeze(0),
            }
            for b, c, m in zip(bboxes, classes, masks)
        ]
        loss_dict = self.criterion(outputs_loss, labels)
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return loss

    @torch.no_grad()
    def segment(
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
            _images = ImageList.from_tensors(images)

        batch_size, num_channels, height, width = _images.tensor.shape
        image_masks = torch.ones((batch_size, height, width), device=self.device)
        features, position_embeddings_list = self.model.backbone(
            _images.tensor.to(self.dtype),
            pixel_mask=image_masks,
        )

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_map, mask = features[-1]
        batch_size, num_channels, height, width = feature_map.shape
        projected_feature_map = self.model.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        position_embeddings = position_embeddings_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)
        encoder_outputs = self.model.encoder(
            inputs_embeds=flattened_features,
            attention_mask=flattened_mask,
            position_embeddings=position_embeddings,
        )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
        )

        sequence_output = decoder_outputs[0]

        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        memory = encoder_outputs[0].permute(0, 2, 1).view(batch_size, self.config.d_model, height, width)
        mask = flattened_mask.view(batch_size, height, width)

        bbox_mask = self.bbox_attention(sequence_output, memory, mask=~mask)
        seg_masks = self.mask_head(
            projected_feature_map,
            bbox_mask,
            [features[2][0], features[1][0], features[0][0]],
        )
        pred_masks = seg_masks.view(
            batch_size,
            self.config.num_queries,
            seg_masks.shape[-2],
            seg_masks.shape[-1],
        )

        sizes = _images.image_sizes
        if not norm_bboxes:
            bboxes = [b * torch.tensor([s[1], s[0], s[1], s[0]]).to(b) for b, s in zip(pred_boxes, sizes)]
        else:
            bboxes = pred_boxes

        scores, classes = list(zip(*[p.max(-1) for p in logits]))

        outputs = dict(
            {
                "bboxes": bboxes,
                "scores": scores,
                "classes": classes,
            }
        )
        outputs["masks"] = [
            nn.functional.interpolate(mask[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
            for mask, size in zip(pred_masks, sizes)
        ]
        return GenericOutputs(outputs)
