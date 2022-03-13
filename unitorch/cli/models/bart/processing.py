# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from unitorch.models.bart import BartProcessor as _BartProcessor
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
from unitorch.cli.models.bart import pretrained_bart_infos


class BartProcessor(_BartProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        special_tokens_ids: Dict = dict(),
        max_seq_length: int = 128,
        max_gen_seq_length: int = 48,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            special_tokens_ids=special_tokens_ids,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("core/process/bart")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/bart")
        pretrained_name = config.getoption("pretrained_name", "default-bart")
        vocab_name_or_path = config.getoption("vocab_path", pretrained_name)
        vocab_path = (
            pretrained_bart_infos[vocab_name_or_path]["vocab"]
            if vocab_name_or_path in pretrained_bart_infos
            else vocab_name_or_path
        )
        vocab_path = cached_path(vocab_path)

        merge_name_or_path = config.getoption("merge_path", pretrained_name)
        merge_path = (
            pretrained_bart_infos[merge_name_or_path]["merge"]
            if merge_name_or_path in pretrained_bart_infos
            else merge_name_or_path
        )
        merge_path = cached_path(merge_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
        }

    @register_process("core/process/bart_generation")
    def _processing_generation(
        self,
        text: str,
        text_pair: str = None,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        outputs = super().processing_generation(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return BaseInputs(
            tokens_ids_a=outputs.tokens_ids,
            tokens_mask_a=outputs.tokens_mask,
            tokens_ids_b=outputs.tokens_ids_pair,
            tokens_mask_b=outputs.tokens_mask_pair,
        ), GenerationTargets(
            refs=outputs.tokens_ids_target,
            masks=outputs.tokens_mask_target,
        )

    @register_process("core/process/bart_inference")
    def _processing_inference(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().processing_inference(
            text=text,
            max_seq_length=max_seq_length,
        )
        return BaseInputs(tokens_ids=outputs.tokens_ids)

    @register_process("core/process/bart_evaluation")
    def _processing_evaluation(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        outputs = super().processing_evaluation(
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.tokens_ids,
            masks=outputs.tokens_mask,
        )

    @register_process("core/postprocess/bart_detokenize")
    def _processing_decode(
        self,
        outputs: GenerationOutputs,
        skip_special_tokens: bool = True,
    ):
        decoded = super().processing_decode(sequences=outputs.sequences)
        _infos = outputs.to_dict()
        _infos.pop("sequences")
        _infos.pop("sequences_scores")
        return BaseOutputs(**_infos, sequences=decoded)
