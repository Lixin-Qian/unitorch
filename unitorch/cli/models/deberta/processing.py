# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.deberta import DebertaProcessor as _DebertaProcessor
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    BaseInputs,
    BaseOutputs,
    BaseTargets,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.deberta import pretrained_deberta_infos


class DebertaProcessor(_DebertaProcessor):
    def __init__(
        self,
        vocab_path,
        merge_path,
        max_seq_length: int = 128,
        source_type_id: int = 0,
        target_type_id: int = 0,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            max_seq_length=max_seq_length,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/deberta")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/deberta")
        pretrained_name = config.getoption("pretrained_name", "default-deberta")
        vocab_name_or_path = config.getoption("vocab_path", pretrained_name)
        vocab_path = (
            pretrained_deberta_infos[vocab_name_or_path]["vocab"]
            if vocab_name_or_path in pretrained_deberta_infos
            else vocab_name_or_path
        )
        vocab_path = cached_path(vocab_path)

        merge_name_or_path = config.getoption("merge_path", pretrained_name)
        merge_path = (
            pretrained_deberta_infos[merge_name_or_path]["merge"]
            if merge_name_or_path in pretrained_deberta_infos
            else merge_name_or_path
        )
        merge_path = cached_path(merge_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
        }

    @register_process("core/process/deberta_classification")
    def _processing_classification(
        self,
        text: str,
        text_pair: str = None,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().processing_classification(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return BaseInputs(
            tokens_ids=outputs.tokens_ids,
            attn_mask=outputs.attn_mask,
            seg_ids=outputs.seg_ids,
            pos_ids=outputs.pos_ids,
        )
