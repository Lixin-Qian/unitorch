# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.infoxlm import InfoXLMProcessor as _InfoXLMProcessor
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
from unitorch.cli.models.infoxlm import pretrained_infoxlm_infos


class InfoXLMProcessor(_InfoXLMProcessor):
    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 30,
        special_tokens_ids: Dict = dict(),
        source_type_id: int = 0,
        target_type_id: int = 1,
    ):
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            special_tokens_ids=special_tokens_ids,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/infoxlm")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/infoxlm")
        pretrained_name = config.getoption("pretrained_name", "default-infoxlm")
        vocab_name_or_path = config.getoption("vocab_path", pretrained_name)
        vocab_path = (
            pretrained_infoxlm_infos[vocab_name_or_path]["vocab"]
            if vocab_name_or_path in pretrained_infoxlm_infos
            else vocab_name_or_path
        )
        vocab_path = cached_path(vocab_path)
        return {
            "vocab_path": vocab_path,
        }

    @register_process("core/process/infoxlm_generation")
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
            tokens_ids=outputs.tokens_ids,
            seg_ids=outputs.seg_ids,
            attn_mask=outputs.attn_mask,
            pos_ids=outputs.pos_ids,
        ), GenerationTargets(
            refs=outputs.tokens_ids_target,
            masks=outputs.tokens_mask_target,
        )

    @register_process("core/process/infoxlm_inference")
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

    @register_process("core/process/infoxlm_evaluation")
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

    @register_process("core/postprocess/infoxlm_detokenize")
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
