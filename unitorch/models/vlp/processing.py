# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torchvision.transforms import Resize, Normalize
from detectron2.data.detection_utils import convert_PIL_to_numpy

from unitorch.functions import pop_first_non_none_value
from unitorch.models.bert import get_bert_tokenizer
from unitorch.models import (
    HuggingfaceGenerationProcessor,
    GenericOutputs,
    _truncate_seq_pair,
)


class VLPProcessor(HuggingfaceGenerationProcessor):
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
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [1.0, 1.0, 1.0],
        resize_shape: List[int] = [224, 224],
    ):
        tokenizer = get_bert_tokenizer(
            vocab_path,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            special_tokens_ids=special_tokens_ids,
        )
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        self.mask_token = self.tokenizer.mask_token
        self._tril_matrix = torch.tril(torch.ones((1024, 1024), dtype=torch.long))
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id

        self.pixel_mean = torch.tensor(pixel_mean)
        self.pixel_std = torch.tensor(pixel_std)

        self.process_resize = Resize(resize_shape)
        self.process_norm = Normalize(self.pixel_mean, self.pixel_std)

    def image_transform(self, image: Image.Image):
        image = self.process_resize(image)
        image = convert_PIL_to_numpy(image, "BGR").copy()
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = self.process_norm(image)
        return image

    def processing_generation(
        self,
        image: Image.Image,
        text: str,
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        image = self.image_transform(image)
        max_seq_length = pop_first_non_none_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_first_non_none_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )
        max_seq_length = max_seq_length + max_gen_seq_length

        tokens_a = self.tokenizer.tokenize(str(text))
        tokens_b = self.tokenizer.tokenize(str(text_pair))
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
        tokens = (
            [self.bos_token, self.eos_token]
            + tokens_a
            + [self.eos_token]
            + tokens_b
            + [self.eos_token]
        )

        tokens_b = tokens_b + [self.eos_token]

        tokens_t = tokens_b[:max_gen_seq_length] + [self.pad_token] * (
            max_gen_seq_length - len(tokens_b)
        )
        tokens_mask_t = [1] * len(tokens_b[:max_gen_seq_length]) + [0] * (
            max_gen_seq_length - len(tokens_b)
        )
        tokens_ids_t = self.tokenizer.convert_tokens_to_ids(tokens_t)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokens_mask = torch.zeros(
            max_seq_length + max_gen_seq_length,
            max_seq_length + max_gen_seq_length,
            dtype=torch.long,
        )
        tokens_mask[:, : len(tokens_a) + 2].fill_(1)
        second_st, second_end = len(tokens_a) + 2, len(tokens_a) + len(tokens_b) + 4
        tokens_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[: second_end - second_st, : second_end - second_st]
        )

        segment_ids = (
            [self.source_type_id, self.source_type_id]
            + [self.source_type_id] * len(tokens_a)
            + [self.source_type_id]
            + [self.target_type_id] * len(tokens_b)
        )
        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += [self.pad_token_id] * len(padding)
        segment_ids += padding
        position_ids = list(range(len(tokens_ids))) + list(
            range(len(tokens_a) + 3, len(tokens_a) + 3 + max_gen_seq_length)
        )

        tokens_ids += self.tokenizer.convert_tokens_to_ids(
            [self.mask_token] * max_gen_seq_length
        )
        segment_ids += [self.target_type_id] * max_gen_seq_length
        tokens_ids_t = [0] * max_seq_length + tokens_ids_t
        tokens_mask_t = [0] * max_seq_length + tokens_mask_t
        mask_st, mask_end = max_seq_length, max_seq_length + max_gen_seq_length
        tokens_mask[mask_st:mask_end, second_st:second_end].copy_(
            self._tril_matrix[: mask_end - mask_st, : second_end - second_st]
        )
        tokens_mask[mask_st:mask_end, mask_st:mask_end].copy_(
            torch.eye(mask_end - mask_st)
        )

        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            seg_ids=torch.tensor(segment_ids, dtype=torch.long),
            attn_mask=torch.tensor(tokens_mask, dtype=torch.long),
            pos_ids=torch.tensor(position_ids, dtype=torch.long),
            image=image,
            tokens_ids_target=torch.tensor(tokens_ids_t, dtype=torch.long),
            tokens_mask_target=torch.tensor(tokens_mask_t, dtype=torch.long),
        )

    def processing_inference(
        self,
        image: Image.Image,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        image = self.image_transform(image)
        max_seq_length = pop_first_non_none_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_seq_length - 2]
        tokens = [self.eos_token] + tokens + [self.eos_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = tokens_ids[:max_seq_length]
        tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += [self.pad_token_id] * len(padding)
        tokens_mask += padding

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        return GenericOutputs(
            image=image,
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            tokens_mask=torch.tensor(tokens_mask, dtype=torch.long),
        )

    def processing_caption(
        self,
        image: Image.Image,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        image = self.image_transform(image)
        max_gen_seq_length = pop_first_non_none_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        _tokens = self.tokenizer.tokenize(str(text))[: max_gen_seq_length - 3]
        tokens = [self.bos_token, self.eos_token] + _tokens + [self.eos_token]
        _tokens += [self.eos_token]

        tokens_t = _tokens + [self.pad_token] * (max_gen_seq_length - len(_tokens))
        tokens_mask_t = [1] * len(_tokens) + [0] * (max_gen_seq_length - len(_tokens))
        tokens_ids_t = self.tokenizer.convert_tokens_to_ids(tokens_t)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokens_mask = torch.zeros(
            max_gen_seq_length + max_gen_seq_length,
            max_gen_seq_length + max_gen_seq_length,
            dtype=torch.long,
        )
        tokens_mask[:, :1].fill_(1)
        second_st, second_end = 1, len(_tokens) + 3
        tokens_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[: second_end - second_st, : second_end - second_st]
        )

        segment_ids = [self.source_type_id, self.source_type_id] + [
            self.target_type_id
        ] * len(_tokens)
        padding = [0] * (max_gen_seq_length - len(tokens_ids))
        tokens_ids += [self.pad_token_id] * len(padding)
        segment_ids += padding
        position_ids = list(range(len(tokens_ids))) + list(
            range(2, 2 + max_gen_seq_length)
        )

        tokens_ids += self.tokenizer.convert_tokens_to_ids(
            [self.mask_token] * max_gen_seq_length
        )
        segment_ids += [self.target_type_id] * max_gen_seq_length
        tokens_ids_t = [0] * max_gen_seq_length + tokens_ids_t
        tokens_mask_t = [0] * max_gen_seq_length + tokens_mask_t
        mask_st, mask_end = max_gen_seq_length, max_gen_seq_length + max_gen_seq_length
        tokens_mask[mask_st:mask_end, second_st:second_end].copy_(
            self._tril_matrix[: mask_end - mask_st, : second_end - second_st]
        )
        tokens_mask[mask_st:mask_end, mask_st:mask_end].copy_(
            torch.eye(mask_end - mask_st)
        )

        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            seg_ids=torch.tensor(segment_ids, dtype=torch.long),
            attn_mask=torch.tensor(tokens_mask, dtype=torch.long),
            pos_ids=torch.tensor(position_ids, dtype=torch.long),
            image=image,
            tokens_ids_target=torch.tensor(tokens_ids_t, dtype=torch.long),
            tokens_mask_target=torch.tensor(tokens_mask_t, dtype=torch.long),
        )

    def processing_image(
        self,
        image: Image.Image,
    ):
        image = self.image_transform(image)
        return GenericOutputs(
            image=image,
        )

    def processing_classification(
        self,
        image: Image.Image,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        image = self.image_transform(image)
        max_seq_length = int(
            pop_first_non_none_value(
                max_seq_length,
                self.max_seq_length,
            )
        )

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 3]
            tokens = [self.bos_token, self.sep_token] + tokens + [self.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [self.source_type_id] * len(tokens_ids)
            tokens_mask = [1] * len(tokens_ids)
        else:
            tokens_b = self.tokenizer.tokenize(str(text_pair))
            _truncate_seq_pair(tokens, tokens_b, max_seq_length - 4)
            segment_ids = (
                [self.source_type_id, self.source_type_id]
                + [self.source_type_id] * len(tokens)
                + [self.source_type_id]
                + [self.target_type_id] * len(tokens_b)
                + [self.target_type_id]
            )
            tokens = (
                [self.bos_token, self.sep_token]
                + tokens
                + [self.sep_token]
                + tokens_b
                + [self.sep_token]
            )
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += len(padding) * [self.pad_token_id]
        tokens_mask += padding
        segment_ids += len(padding) * [self.target_type_id]

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return GenericOutputs(
            image=image,
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            seg_ids=torch.tensor(segment_ids, dtype=torch.long),
            attn_mask=torch.tensor(tokens_mask, dtype=torch.long),
            pos_ids=torch.tensor(
                list(
                    range(
                        max_seq_length,
                    )
                ),
                dtype=torch.long,
            ),
        )
