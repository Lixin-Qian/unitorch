# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class ProphetnetLoss(nn.Module):
    """
    Creates a criterion for prophetnet
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        """
        Args:
            reduction (string, optional): specifies the reduction to apply to the output.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input: output tensor from model
            target: target tensor for model
            masks: mask matrix for target
            sample_weight: weight for each sample in a batch
        """
        assert input.dim() == 4 and target.dim() == 2

        batch_size, ngram, seq_len, num_class = input.size()
        input = input.contiguous().view(batch_size * ngram * seq_len, num_class)
        target = target.repeat(1, ngram).contiguous().view(-1)
        target = target.long()
        if masks is None:
            masks = torch.ones(batch_size, seq_len).to(target)
        masks = masks.repeat(1, ngram).contiguous().view(-1)

        loss = nn.CrossEntropyLoss(reduction="none")(input, target)
        loss = loss * masks.float()
        loss = loss.contiguous().view(batch_size, ngram * seq_len).sum(1) / torch.max(
            masks.contiguous().view(batch_size, ngram * seq_len).float().sum(1),
            torch.ones(batch_size).to(masks.device),
        )

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss
