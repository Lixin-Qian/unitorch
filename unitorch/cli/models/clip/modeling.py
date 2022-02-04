# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch import hf_cached_path
from unitorch.models.clip import (
    CLIPForPretrain as _CLIPForPretrain,
    CLIPForClassification as _CLIPForClassification,
    CLIPForTextClassification as _CLIPForTextClassification,
    CLIPForImageClassification as _CLIPForImageClassification,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.clip import pretrained_clip_infos


@register_model("core/model/pretrain/clip")
class CLIPForPretrain(_CLIPForPretrain):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        freeze_base_model: bool = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/pretrain/clip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/pretrain/clip")
        pretrained_name = config.getoption("pretrained_name", "default-clip")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_clip_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_clip_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_clip_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_clip_infos
                and "weight" in pretrained_clip_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return LossOutputs(loss=outputs)


@register_model("core/model/classification/clip")
class CLIPForClassification(_CLIPForClassification):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_class: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_class=num_class,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/clip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/clip")
        pretrained_name = config.getoption("pretrained_name", "default-clip")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_clip_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_clip_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_class = config.getoption("num_class", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_class=num_class,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_clip_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_clip_infos
                and "weight" in pretrained_clip_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/text_clip")
class CLIPForTextClassification(_CLIPForTextClassification):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_class: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_class=num_class,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/text_clip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/text_clip")
        pretrained_name = config.getoption("pretrained_name", "default-clip")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_clip_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_clip_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_class = config.getoption("num_class", 1)
        freeze_base_model = config.getoption("freeze_base_Truemodel", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_class=num_class,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_clip_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_clip_infos
                and "weight" in pretrained_clip_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/image_clip")
class CLIPForImageClassification(_CLIPForImageClassification):
    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_class: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_class=num_class,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/image_clip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/image_clip")
        pretrained_name = config.getoption("pretrained_name", "default-clip")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_clip_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_clip_infos
            else config_name_or_path
        )

        config_path = hf_cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_class = config.getoption("num_class", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_class=num_class,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        if pretrained_name is not None:
            pretrained_name_or_path = config.getoption(
                "pretrained_weight_path", pretrained_name
            )
            weight_path = (
                pretrained_clip_infos[pretrained_name_or_path]["weight"]
                if pretrained_name_or_path in pretrained_clip_infos
                and "weight" in pretrained_clip_infos[pretrained_name_or_path]
                else pretrained_name_or_path
            )
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        pixel_values=None,
    ):
        outputs = super().forward(pixel_values=pixel_values)
        return ClassificationOutputs(outputs=outputs)
