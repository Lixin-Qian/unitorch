# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.vlp import VLPProcessor as _VLPProcessor
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
from unitorch.cli.models.vlp import pretrained_vlp_infos


class VLPProcessor(_VLPProcessor):
    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 30,
        do_lower_case=False,
        do_basic_tokenize=False,
        special_tokens_ids: Dict = dict(),
        source_type_id: int = 0,
        target_type_id: int = 1,
        pixel_mean: List[float] = [103.53, 116.28, 123.675],
        pixel_std: List[float] = [1.0, 1.0, 1.0],
        resize_shape: List[int] = [224, 224],
    ):
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            special_tokens_ids=special_tokens_ids,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            resize_shape=resize_shape,
        )

    @classmethod
    @add_default_section_for_init("core/process/vlp")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/vlp")
        pretrained_name = config.getoption("pretrained_name", "default-vlp")
        vocab_name_or_path = config.getoption("vocab_path", pretrained_name)
        vocab_path = (
            pretrained_vlp_infos[vocab_name_or_path]["vocab"]
            if vocab_name_or_path in pretrained_vlp_infos
            else vocab_name_or_path
        )
        vocab_path = cached_path(vocab_path)
        return {
            "vocab_path": vocab_path,
        }

    def _request_url(self, url):
        while True:
            try:
                doc = requests.get(url, timeout=600)
                return doc
            except:
                time.sleep(random() * 2)

    @register_process("core/process/vlp_generation")
    def _processing_generation(
        self,
        image: Union[Image.Image, str],
        text: str,
        text_pair: str = None,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
        image_type: Optional[str] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_generation(
            image=image,
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
            pixel_values=outputs.image,
        ), GenerationTargets(
            refs=outputs.tokens_ids_target,
            masks=outputs.tokens_mask_target,
        )

    @register_process("core/process/vlp_inference")
    def _processing_inference(
        self,
        image: Union[Image.Image, str],
        text: str,
        max_seq_length: Optional[int] = None,
        image_type: Optional[str] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_inference(
            image=image,
            text=text,
            max_seq_length=max_seq_length,
        )
        return BaseInputs(
            tokens_ids=outputs.tokens_ids,
            pixel_values=outputs.image,
        )

    @register_process("core/process/vlp_caption")
    def _processing_caption(
        self,
        image: Union[Image.Image, str],
        text: str,
        max_gen_seq_length: Optional[int] = None,
        image_type: Optional[str] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_caption(
            image=image,
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return BaseInputs(
            tokens_ids=outputs.tokens_ids,
            seg_ids=outputs.seg_ids,
            attn_mask=outputs.attn_mask,
            pos_ids=outputs.pos_ids,
            pixel_values=outputs.image,
        ), GenerationTargets(
            refs=outputs.tokens_ids_target,
            masks=outputs.tokens_mask_target,
        )

    @register_process("core/process/vlp_image")
    def _processing_image(
        self,
        image: Union[Image.Image, str],
        image_type: Optional[str] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_image(
            image=image,
        )
        return BaseInputs(pixel_values=outputs.image)

    @register_process("core/process/vlp_classification")
    def _processing_classification(
        self,
        image: Union[Image.Image, str],
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        image_type: Optional[str] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().processing_classification(
            image=image,
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return BaseInputs(
            tokens_ids=outputs.tokens_ids,
            seg_ids=outputs.seg_ids,
            attn_mask=outputs.attn_mask,
            pos_ids=outputs.pos_ids,
            pixel_values=outputs.image,
        )

    @register_process("core/process/vlp_evaluation")
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

    @register_process("core/postprocess/vlp_detokenize")
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
