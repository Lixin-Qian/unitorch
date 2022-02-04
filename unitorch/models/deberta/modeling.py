# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.deberta.modeling_deberta import (
    DebertaConfig,
    DebertaModel,
    DebertaOnlyMLMHead,
    ContextPooler,
    StableDropout,
)
from unitorch.models import GenericModel


class DebertaForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = DebertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.deberta = DebertaModel(self.config)
        self.pooler = ContextPooler(self.config)
        self.dropout = StableDropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_class)
        self.init_weights()

    def forward(
        self,
        tokens_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.deberta(
            tokens_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids,
            position_ids=pos_ids,
        )
        pooled_output = self.pooler(outputs[0])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class DebertaForMaskLM(GenericModel):
    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = DebertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.deberta = DebertaModel(self.config)
        self.cls = DebertaOnlyMLMHead(self.config)
        self.init_weights()
        self.deberta.embeddings.word_embeddings.weight = (
            self.cls.predictions.decoder.weight
        )

    def forward(
        self,
        tokens_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.deberta(
            tokens_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids,
            position_ids=pos_ids,
        )
        sequence_output = outputs[0]
        logits = self.cls(sequence_output)
        return logits
