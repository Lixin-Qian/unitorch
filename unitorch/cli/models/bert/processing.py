# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import hf_cached_path
from unitorch.models.bert import BertProcessor as _BertProcessor
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
from unitorch.cli.models.bert import pretrained_bert_infos


class BertProcessor(_BertProcessor):
    def __init__(
        self,
        vocab_path,
        max_seq_length: int = 128,
        special_tokens_ids: Dict = dict(),
        do_lower_case: bool = True,
        do_basic_tokenize: bool = True,
        do_whole_word_mask: bool = True,
        masked_lm_prob: float = 0.15,
        max_predictions_per_seq: int = 20,
    ):
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            special_tokens_ids=special_tokens_ids,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            do_whole_word_mask=do_whole_word_mask,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq,
        )

    @classmethod
    @add_default_section_for_init("core/process/bert")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/bert")
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        vocab_name_or_path = config.getoption("vocab_path", pretrained_name)
        vocab_path = (
            pretrained_bert_infos[vocab_name_or_path]["vocab"]
            if vocab_name_or_path in pretrained_bert_infos
            else vocab_name_or_path
        )
        vocab_path = hf_cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("core/process/bert_classification")
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

    @register_process("core/process/bert_pretrain")
    def _processing_pretrain(
        self,
        text: str,
        text_pair: str,
        nsp_label: int,
        max_seq_length: Optional[int] = None,
        masked_lm_prob: Optional[float] = None,
        do_whole_word_mask: Optional[bool] = None,
        max_predictions_per_seq: Optional[int] = None,
    ):
        outputs = super().processing_pretrain(
            text=text,
            text_pair=text_pair,
            nsp_label=nsp_label,
            max_seq_length=max_seq_length,
            masked_lm_prob=masked_lm_prob,
            do_whole_word_mask=do_whole_word_mask,
            max_predictions_per_seq=max_predictions_per_seq,
        )
        return BaseInputs(
            tokens_ids=outputs.tokens_ids,
            attn_mask=outputs.attn_mask,
            seg_ids=outputs.seg_ids,
            pos_ids=outputs.pos_ids,
            nsp_label=outputs.nsp_label,
            mlm_label=outputs.mlm_label,
            mlm_label_mask=outputs.mlm_label_mask,
        )
