# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import BertConfig, BertModel
from unitorch.models import GenericModel


class BertForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Args:
            config_path: config file path to bert model
            num_class: num class to classification
            gradient_checkpointing: if to enable gradient_checkpointing
        """
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.bert = BertModel(self.config)
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
        """
        Args:
            tokens_ids: tokens of text
            attn_mask: attention mask of tokens
            seg_ids: token type ids
            pos_ids: position ids
        """
        outputs = self.bert(
            tokens_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids,
            position_ids=pos_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
