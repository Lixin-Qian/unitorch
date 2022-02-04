# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.models.unilm.modeling import UnilmForGeneration


class InfoXLMForGeneration(UnilmForGeneration):
    def __init__(
        self,
        config_path,
        freeze_word_embedding=True,
    ):
        super().__init__(config_path)
        if freeze_word_embedding:
            self.cls.predictions.decoder.weight.requires_grad = False
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
