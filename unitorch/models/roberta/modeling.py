# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
    RobertaLMHead,
)
from unitorch.models import GenericModel


class RobertaForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = RobertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = RobertaModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_class)
        self.init_weights()

    def forward(
        self,
        tokens_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.roberta(
            tokens_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids,
            position_ids=pos_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaForMaskLM(GenericModel):
    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = RobertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = RobertaModel(self.config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(self.config)
        self.init_weights()
        self.roberta.embeddings.word_embeddings.weight = self.lm_head.decoder.weight

    def forward(
        self,
        tokens_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.roberta(
            tokens_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids,
            position_ids=pos_ids,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        return logits
