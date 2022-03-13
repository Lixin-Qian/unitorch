# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import MBartTokenizer

from unitorch.functions import pop_first_non_none_value
from unitorch.models import HuggingfaceGenerationProcessor


def _get_mbart_tokenizer(
    vocab_path,
    special_tokens_ids=dict(),
):
    assert os.path.exists(vocab_path)
    tokenizer = MBartTokenizer(vocab_path)
    for token, _id in special_tokens_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class MBartProcessor(HuggingfaceGenerationProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
        special_tokens_ids: Dict = dict(),
    ):
        """
        Args:
            vocab_path: vocab file path in mbart tokenizer
            max_seq_length: max sequence length encode text
            max_gen_seq_length: max sequence length decode text
            special_tokens_ids: special tokens dict in mbart tokenizer
        """
        tokenizer = _get_mbart_tokenizer(
            vocab_path,
            special_tokens_ids=special_tokens_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
