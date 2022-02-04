# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch import hf_cached_path
from unitorch.models.clip import CLIPProcessor as _CLIPProcessor
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
from unitorch.cli.models.clip import pretrained_clip_infos


class CLIPProcessor(_CLIPProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        max_seq_length: int = 128,
        position_start_id: int = 0,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/clip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/clip")
        pretrained_name = config.getoption("pretrained_name", "default-clip")
        vocab_name_or_path = config.getoption("vocab_path", pretrained_name)
        vocab_path = (
            pretrained_clip_infos[vocab_name_or_path]["vocab"]
            if vocab_name_or_path in pretrained_clip_infos
            else vocab_name_or_path
        )
        vocab_path = hf_cached_path(vocab_path)

        merge_name_or_path = config.getoption("merge_path", pretrained_name)
        merge_path = (
            pretrained_clip_infos[merge_name_or_path]["merge"]
            if merge_name_or_path in pretrained_clip_infos
            else merge_name_or_path
        )
        merge_path = hf_cached_path(merge_path)

        vision_config_name_or_path = config.getoption(
            "vision_config_path", pretrained_name
        )
        vision_config_path = (
            pretrained_clip_infos[vision_config_name_or_path]["vision_config"]
            if vision_config_name_or_path in pretrained_clip_infos
            else vision_config_name_or_path
        )

        vision_config_path = hf_cached_path(vision_config_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "vision_config_path": vision_config_path,
        }

    def _read_image(self, image_path):
        return Image.open(image_path)

    @register_process("core/process/clip_classification")
    def _processing_classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: int = None,
    ):
        if isinstance(image, str):
            image = self._read_image(image)

        outputs = super().processing_classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        )
        return BaseInputs(
            input_ids=outputs.tokens_ids,
            attention_mask=outputs.attn_mask,
            position_ids=outputs.pos_ids,
            pixel_values=outputs.image,
        )

    @register_process("core/process/clip_text_classification")
    def _processing_text_classifictaion(
        self,
        text: str,
        max_seq_length: int = None,
    ):
        outputs = super().processing_text_classifictaion(
            text=text,
            max_seq_length=max_seq_length,
        )
        return BaseInputs(
            input_ids=outputs.tokens_ids,
            attention_mask=outputs.attn_mask,
            position_ids=outputs.pos_ids,
        )

    @register_process("core/process/clip_image_classification")
    def _processing_image_classifictaion(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = self._read_image(image)
        outputs = super().processing_image_classifictaion(image=image)
        return BaseInputs(pixel_values=outputs.image)
