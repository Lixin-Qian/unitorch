# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch


from unitorch.functions import pop_first_non_none_value
from unitorch.models.bert import get_bert_tokenizer
from unitorch.models import (
    HuggingfaceGenerationProcessor,
    GenericOutputs,
    _truncate_seq_pair,
)


class UnilmProcessor(HuggingfaceGenerationProcessor):
    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 30,
        do_lower_case=True,
        do_basic_tokenize=True,
        special_tokens_ids: Dict = dict(),
        source_type_id: int = 0,
        target_type_id: int = 1,
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

    def processing_generation(
        self,
        text: str,
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
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
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = (
            [self.bos_token] + tokens_a + [self.eos_token] + tokens_b + [self.eos_token]
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
        tokens_mask[:, : len(tokens_a) + 1].fill_(1)
        second_st, second_end = len(tokens_a) + 1, len(tokens_a) + len(tokens_b) + 3
        tokens_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[: second_end - second_st, : second_end - second_st]
        )

        segment_ids = (
            [self.source_type_id]
            + [self.source_type_id] * len(tokens_a)
            + [self.source_type_id]
            + [self.target_type_id] * len(tokens_b)
        )
        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += [self.pad_token_id] * len(padding)
        segment_ids += padding
        position_ids = list(range(len(tokens_ids))) + list(
            range(len(tokens_a) + 2, len(tokens_a) + 2 + max_gen_seq_length)
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
            tokens_ids_target=torch.tensor(tokens_ids_t, dtype=torch.long),
            tokens_mask_target=torch.tensor(tokens_mask_t, dtype=torch.long),
        )
