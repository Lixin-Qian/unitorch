# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from functools import partial

import torch
from transformers import ProphetNetTokenizer
from unitorch.models import HuggingfaceGenerationProcessor


def get_prophetnet_tokenizer(vocab_path, special_tokens_ids=dict()):
    assert os.path.exists(vocab_path)
    tokenizer = ProphetNetTokenizer(vocab_path)
    for token, _id in special_tokens_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class ProphetNetProcessor(HuggingfaceGenerationProcessor):
    def __init__(
        self,
        vocab_path=None,
        special_tokens_ids=dict(),
        max_seq_length=128,
        max_gen_seq_length=48,
    ):
        tokenizer = get_prophetnet_tokenizer(
            vocab_path,
            special_tokens_ids=special_tokens_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
