# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.models.deberta import (
    DebertaV2ForClassification as _DebertaV2ForClassification,
)
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.deberta import pretrained_deberta_v2_infos


@register_model("core/model/classification/deberta_v2")
class DebertaV2ForClassification(_DebertaV2ForClassification):
    def __init__(
        self,
        config_path: str,
        num_class: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            num_class=num_class,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/deberta_v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/deberta_v2")
        pretrained_name = config.getoption("pretrained_name", "default-deberta-v2")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        num_class = config.getoption("num_class", 1)

        config_path = (
            pretrained_deberta_v2_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_deberta_v2_infos
            else config_name_or_path
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, num_class, gradient_checkpointing)
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption("pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_deberta_v2_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_deberta_v2_infos
                and "weight" in pretrained_deberta_v2_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        tokens_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            tokens_ids=tokens_ids,
            attn_mask=attn_mask,
            seg_ids=seg_ids,
            pos_ids=pos_ids,
        )
        return ClassificationOutputs(outputs=outputs)
