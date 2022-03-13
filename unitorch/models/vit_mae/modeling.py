# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEConfig,
    ViTMAEModel,
    ViTMAEDecoder,
)
from unitorch.models import GenericModel


class ViTMAEForPretrain(GenericModel):
    def __init__(
        self,
        config_path: str,
    ):
        """
        Args:
            config_path: config file path to vit mae model
        """
        super().__init__()
        self.config = ViTMAEConfig.from_json_file(config_path)

        self.vit = ViTMAEModel(self.config)
        self.decoder = ViTMAEDecoder(self.config, num_patches=self.vit.embeddings.num_patches)
        self.init_weights()

    def _patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def _forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self._patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

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
        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
        logits = decoder_outputs.logits

        return self._forward_loss(pixel_values, logits, mask)
