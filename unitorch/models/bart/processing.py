# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers import BartTokenizer
from unitorch.models import HuggingfaceGenerationProcessor


def _get_bart_tokenizer(
    vocab_path,
    merge_path,
    special_tokens_ids=dict(),
):
    assert os.path.exists(vocab_path) and os.path.exists(merge_path)
    tokenizer = BartTokenizer(vocab_path, merge_path)
    for token, _id in special_tokens_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class BartProcessor(HuggingfaceGenerationProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        special_tokens_ids: Optional[Dict] = dict(),
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Args:
            vocab_path: vocab file path in bart tokenizer
            merge_path: merge file path in bart tokenizer
            special_tokens_ids: special tokens dict in bart tokenizer
            max_seq_length: max sequence length encode text
            max_gen_seq_length: max sequence length decode text
        """
        tokenizer = _get_bart_tokenizer(
            vocab_path,
            merge_path,
            special_tokens_ids=special_tokens_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
