# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.models.unilm import UnilmForGeneration as _UnilmForGeneration
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator, GenerationOutputs
from unitorch.cli.models.unilm import pretrained_unilm_infos


@register_model("core/model/generation/unilm", generation_model_decorator)
class UnilmForGeneration(_UnilmForGeneration):
    def __init__(self, config_path):
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/unilm")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/generation/unilm")
        pretrained_name = config.getoption("pretrained_name", "default-unilm")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_unilm_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_unilm_infos
            else config_name_or_path
        )

        config_path = cached_path(config_path)

        inst = cls(config_path)
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption("pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_unilm_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_unilm_infos
                and "weight" in pretrained_unilm_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        tokens_ids=None,
        attn_mask=None,
        seg_ids=None,
        pos_ids=None,
        decoder_input_ids=None,
        decoder_pos_ids=None,
        decoder_seg_ids=None,
        decoder_attn_mask=None,
        decoder_mask_ids=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = super().forward(
            tokens_ids=tokens_ids,
            attn_mask=attn_mask,
            seg_ids=seg_ids,
            pos_ids=pos_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_pos_ids=decoder_pos_ids,
            decoder_seg_ids=decoder_seg_ids,
            decoder_attn_mask=decoder_attn_mask,
            decoder_mask_ids=decoder_mask_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.training:
            return GenerationOutputs(sequences=outputs)
        return outputs

    @add_default_section_for_function("core/model/generation/unilm")
    def generate(
        self,
        tokens_ids,
        num_beams=5,
        decoder_start_token_id=101,
        decoder_end_token_id=102,
        num_return_sequences=1,
        min_gen_seq_length=0,
        max_gen_seq_length=48,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        length_penalty=1.0,
        num_beam_groups=1,
        diversity_penalty=0.0,
        diverse_rate=0.0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    ):
        outputs = super().generate(
            tokens_ids,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            diverse_rate=diverse_rate,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return GenerationOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )
