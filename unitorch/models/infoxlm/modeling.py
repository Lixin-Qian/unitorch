# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.models.unilm.modeling import UnilmForGeneration
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class InfoXLMForGeneration(UnilmForGeneration):
    def __init__(
        self,
        config_path: str,
        freeze_word_embedding: Optional[bool] = True,
    ):
        """
        Args:
            config_path: config file path to infoxlm model
            freeze_word_embedding: if to freeze word embedding in infoxlm model
        """
        super().__init__(config_path)
        if freeze_word_embedding:
            self.cls.predictions.decoder.weight.requires_grad = False
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
