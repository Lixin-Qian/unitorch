# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.file_utils import is_remote_url
from unitorch.models.vlp import (
    VLPForGeneration as _VLPForGeneration,
    VLPForClassification as _VLPForClassification,
)
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import (
    generation_model_decorator,
    GenerationOutputs,
    ClassificationOutputs,
)
from unitorch.cli.models.vlp import pretrained_vlp_infos


@register_model("core/model/generation/vlp", generation_model_decorator)
class VLPForGeneration(_VLPForGeneration):
    def __init__(
        self,
        vlp_config_path: str,
        detectron2_config_path: str,
        freeze_vision_model: bool = True,
        max_num_bbox: int = 100,
    ):
        super().__init__(
            vlp_config_path=vlp_config_path,
            detectron2_config_path=detectron2_config_path,
            freeze_vision_model=freeze_vision_model,
            max_num_bbox=max_num_bbox,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/vlp")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/generation/vlp")
        pretrained_name = config.getoption("pretrained_name", "default-vlp")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_vlp_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_vlp_infos
            else config_name_or_path
        )

        vlp_config_path = cached_path(config_path)

        config_name_or_path = config.getoption("detectron2_config_path", pretrained_name)
        config_path = (
            pretrained_vlp_infos[config_name_or_path]["detectron2_config"]
            if config_name_or_path in pretrained_vlp_infos
            else config_name_or_path
        )

        detectron2_config_path = cached_path(config_path)

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        max_num_bbox = config.getoption("max_num_bbox", 100)

        inst = cls(
            vlp_config_path=vlp_config_path,
            detectron2_config_path=detectron2_config_path,
            freeze_vision_model=freeze_vision_model,
            max_num_bbox=max_num_bbox,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption("detectron2_pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_vlp_infos[pretrained_name_or_path]["detectron2_weight"]
                if pretrained_name_or_path in pretrained_vlp_infos
                and "detectron2_weight" in pretrained_vlp_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )

            inst.vision_model.from_pretrained(weight_path)

            pretrained_name_or_path = config.getoption("pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_vlp_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_vlp_infos
                and "weight" in pretrained_vlp_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )

            inst.from_pretrained(weight_path)

        return inst

    def from_pretrained(self, pretrained_weight_path):
        if not (is_remote_url(pretrained_weight_path) or os.path.exists(pretrained_weight_path)):
            return
        weight_path = cached_path(pretrained_weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "vis_embed" in key:
                new_key = key.replace("vis_embed", "vision_embedding")
            if "vis_pe_embed" in key:
                new_key = key.replace("vis_pe_embed", "vision_position_embedding")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        self.load_state_dict(state_dict, False)
        logging.info(f"{type(self).__name__} model load weight from pretrain {weight_path}")

    @autocast()
    def forward(
        self,
        tokens_ids=None,
        attn_mask=None,
        seg_ids=None,
        pos_ids=None,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_pos_ids=None,
        decoder_seg_ids=None,
        decoder_attn_mask=None,
        decoder_mask_ids=None,
        decoder_pixel_mask=None,
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
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_pos_ids=decoder_pos_ids,
            decoder_seg_ids=decoder_seg_ids,
            decoder_attn_mask=decoder_attn_mask,
            decoder_mask_ids=decoder_mask_ids,
            decoder_pixel_mask=decoder_pixel_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.training:
            return GenerationOutputs(sequences=outputs)
        return outputs

    @add_default_section_for_function("core/model/generation/vlp")
    def generate(
        self,
        pixel_values,
        tokens_ids=None,
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
            pixel_values=pixel_values,
            tokens_ids=tokens_ids,
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


@register_model("core/model/caption/vlp", generation_model_decorator)
class VLPForCaption(_VLPForGeneration):
    def __init__(
        self,
        vlp_config_path: str,
        detectron2_config_path: str,
        freeze_vision_model: bool = True,
        max_num_bbox: int = 100,
    ):
        super().__init__(
            vlp_config_path=vlp_config_path,
            detectron2_config_path=detectron2_config_path,
            freeze_vision_model=freeze_vision_model,
            max_num_bbox=max_num_bbox,
        )

    @classmethod
    @add_default_section_for_init("core/model/caption/vlp")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/caption/vlp")
        pretrained_name = config.getoption("pretrained_name", "default-vlp")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_vlp_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_vlp_infos
            else config_name_or_path
        )

        vlp_config_path = cached_path(config_path)

        config_name_or_path = config.getoption("detectron2_config_path", pretrained_name)
        config_path = (
            pretrained_vlp_infos[config_name_or_path]["detectron2_config"]
            if config_name_or_path in pretrained_vlp_infos
            else config_name_or_path
        )

        detectron2_config_path = cached_path(config_path)

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        max_num_bbox = config.getoption("max_num_bbox", 100)

        inst = cls(
            vlp_config_path=vlp_config_path,
            detectron2_config_path=detectron2_config_path,
            freeze_vision_model=freeze_vision_model,
            max_num_bbox=max_num_bbox,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption("detectron2_pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_vlp_infos[pretrained_name_or_path]["detectron2_weight"]
                if pretrained_name_or_path in pretrained_vlp_infos
                and "detectron2_weight" in pretrained_vlp_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )

            inst.vision_model.from_pretrained(weight_path)

            pretrained_name_or_path = config.getoption("pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_vlp_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_vlp_infos
                and "weight" in pretrained_vlp_infos[pretrained_name_or_path]
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
        pixel_values=None,
        decoder_input_ids=None,
        decoder_pos_ids=None,
        decoder_seg_ids=None,
        decoder_attn_mask=None,
        decoder_mask_ids=None,
        decoder_pixel_mask=None,
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
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_pos_ids=decoder_pos_ids,
            decoder_seg_ids=decoder_seg_ids,
            decoder_attn_mask=decoder_attn_mask,
            decoder_mask_ids=decoder_mask_ids,
            decoder_pixel_mask=decoder_pixel_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.training:
            return GenerationOutputs(sequences=outputs)
        return outputs

    @add_default_section_for_function("core/model/caption/vlp")
    def generate(
        self,
        pixel_values,
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
            pixel_values=pixel_values,
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


@register_model("core/model/classification/vlp")
class VLPForClassification(_VLPForClassification):
    def __init__(
        self,
        vlp_config_path: str,
        detectron2_config_path: str,
        freeze_vision_model: bool = True,
        freeze_base_model: bool = True,
        max_num_bbox: int = 100,
        num_class: int = 1,
    ):
        super().__init__(
            vlp_config_path=vlp_config_path,
            detectron2_config_path=detectron2_config_path,
            freeze_vision_model=freeze_vision_model,
            freeze_base_model=freeze_base_model,
            max_num_bbox=max_num_bbox,
            num_class=num_class,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/vlp")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/vlp")
        pretrained_name = config.getoption("pretrained_name", "default-vlp")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_vlp_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_vlp_infos
            else config_name_or_path
        )

        vlp_config_path = cached_path(config_path)

        config_name_or_path = config.getoption("detectron2_config_path", pretrained_name)
        config_path = (
            pretrained_vlp_infos[config_name_or_path]["detectron2_config"]
            if config_name_or_path in pretrained_vlp_infos
            else config_name_or_path
        )

        detectron2_config_path = cached_path(config_path)

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        freeze_base_model = config.getoption("freeze_base_model", True)
        max_num_bbox = config.getoption("max_num_bbox", 100)
        num_class = config.getoption("num_class", 1)

        inst = cls(
            vlp_config_path=vlp_config_path,
            detectron2_config_path=detectron2_config_path,
            freeze_vision_model=freeze_vision_model,
            freeze_base_model=freeze_base_model,
            max_num_bbox=max_num_bbox,
            num_class=num_class,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption("detectron2_pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_vlp_infos[pretrained_name_or_path]["detectron2_weight"]
                if pretrained_name_or_path in pretrained_vlp_infos
                and "detectron2_weight" in pretrained_vlp_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )

            inst.vision_model.from_pretrained(weight_path)

            pretrained_name_or_path = config.getoption("pretrained_weight_path", pretrained_name)
            weight_path = (
                pretrained_vlp_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_vlp_infos
                and "weight" in pretrained_vlp_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )

            inst.from_pretrained(weight_path)

        return inst

    def from_pretrained(self, pretrained_weight_path):
        if not (is_remote_url(pretrained_weight_path) or os.path.exists(pretrained_weight_path)):
            return
        weight_path = cached_path(pretrained_weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "vis_embed" in key:
                new_key = key.replace("vis_embed", "vision_embedding")
            if "vis_pe_embed" in key:
                new_key = key.replace("vis_pe_embed", "vision_position_embedding")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        self.load_state_dict(state_dict, False)
        logging.info(f"{type(self).__name__} model load weight from pretrain {weight_path}")

    @autocast()
    def forward(
        self,
        tokens_ids=None,
        attn_mask=None,
        seg_ids=None,
        pos_ids=None,
        pixel_values=None,
    ):
        outputs = super().forward(
            tokens_ids=tokens_ids,
            attn_mask=attn_mask,
            seg_ids=seg_ids,
            pos_ids=pos_ids,
            pixel_values=pixel_values,
        )
        return ClassificationOutputs(outputs=outputs)
