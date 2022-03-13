# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from random import randint, shuffle, choice

import numpy as np
import torch
from transformers import BertTokenizer

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.functions import pop_first_non_none_value
from unitorch.models import _truncate_seq_pair, HuggingfaceClassificationProcessor


def _get_random_word(vocab_words):
    i = randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def _get_random_mask_indexes(
    tokens,
    masked_lm_prob=0.15,
    do_whole_word_mask=True,
    max_predictions_per_seq=20,
    special_tokens=[],
):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in special_tokens:
            continue
        if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")) and cand_indexes[-1][
            -1
        ] == i - 1:
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(covered_indexes) >= num_to_predict:
            break
        if len(covered_indexes) + len(index_set) > num_to_predict or any(i in covered_indexes for i in index_set):
            continue
        covered_indexes.update(index_set)
    return covered_indexes


def get_bert_tokenizer(
    vocab_path,
    do_lower_case: bool = True,
    do_basic_tokenize: bool = True,
    special_tokens_ids: Dict = dict(),
):
    """
    Args:
        vocab_path: vocab file path in bert tokenizer
        do_lower_case: if do lower case to input text
        do_basic_tokenize: if do basic tokenize to input text
        special_tokens_ids: special tokens dict in bert tokenizer
    Returns: return bert tokenizer
    """
    assert os.path.exists(vocab_path)
    tokenizer = BertTokenizer(
        vocab_path,
        do_lower_case=do_lower_case,
        do_basic_tokenize=do_basic_tokenize,
    )
    for token, _id in special_tokens_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class BertProcessor(HuggingfaceClassificationProcessor):
    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        special_tokens_ids: Optional[Dict] = dict(),
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        do_whole_word_mask: Optional[bool] = True,
        masked_lm_prob: Optional[float] = 0.15,
        max_predictions_per_seq: Optional[int] = 20,
    ):
        """
        Args:
            vocab_path: vocab file path in bert tokenizer
            max_seq_length: max sequence length text
            special_tokens_ids: special tokens dict in bert tokenizer
            do_lower_case: if do lower case to input text
            do_basic_tokenize: if do basic tokenize to input text
            do_whole_word_mask: if mask whole word in mlm task
            masked_lm_prob: mask prob in mlm task
            max_predictions_per_seq: max tokens to predict in mlm task
        """
        tokenizer = get_bert_tokenizer(
            vocab_path,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            special_tokens_ids=special_tokens_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        self.do_whole_word_mask = do_whole_word_mask
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(self.tokenizer.vocab.keys())

    def processing_pretrain(
        self,
        text: str,
        text_pair: str,
        nsp_label: int,
        max_seq_length: Optional[int] = None,
        masked_lm_prob: Optional[float] = None,
        do_whole_word_mask: Optional[bool] = None,
        max_predictions_per_seq: Optional[int] = None,
    ):
        """
        Args:
            text: text_a to bert pretrain
            text_pair: text_b to bert pretrain
            nsp_label: nsp label to bert pretrain
            max_seq_length: max sequence length text
            masked_lm_prob: mask prob in mlm task
            do_whole_word_mask: if mask whole word in mlm task
            max_predictions_per_seq: max tokens to predict in mlm task
        """
        max_seq_length = int(
            pop_first_non_none_value(
                max_seq_length,
                self.max_seq_length,
            )
        )

        masked_lm_prob = float(
            pop_first_non_none_value(
                masked_lm_prob,
                self.masked_lm_prob,
            )
        )

        do_whole_word_mask = bool(
            pop_first_non_none_value(
                do_whole_word_mask,
                self.do_whole_word_mask,
            )
        )

        max_predictions_per_seq = int(
            pop_first_non_none_value(
                max_predictions_per_seq,
                self.max_predictions_per_seq,
            )
        )

        tokens_a = self.tokenizer.tokenize(str(text))
        tokens_b = self.tokenizer.tokenize(str(text_pair))
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = [self.cls_token] + tokens_a + [self.sep_token] + tokens_b + [self.sep_token]

        covered_indexes = _get_random_mask_indexes(
            tokens,
            masked_lm_prob,
            do_whole_word_mask,
            max_predictions_per_seq,
            special_tokens=[self.cls_token, self.sep_token],
        )
        label = [tokens[pos] if pos in covered_indexes else self.pad_token for pos in range(max_seq_length)]
        label_mask = [1 if pos in covered_indexes else 0 for pos in range(max_seq_length)]
        label = self.tokenizer.convert_tokens_to_ids(label)

        for index in covered_indexes:
            masked_token = None
            if random.random() < 0.8:
                masked_token = self.masked_token
            else:
                masked_token = tokens[index] if random.random() < 0.5 else _get_random_word(self.vocab_words)
            tokens[index] = masked_token

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] + [0] * len(tokens_a) + [0] + [1] * len(tokens_b) + [1]
        tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += len(padding) * [self.pad_token_id]
        tokens_mask += padding
        segment_ids += len(padding) * [1]

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return dict(
            {
                "tokens_ids": torch.tensor(tokens_ids, dtype=torch.long),
                "seg_ids": torch.tensor(segment_ids, dtype=torch.long),
                "attn_mask": torch.tensor(tokens_mask, dtype=torch.long),
                "pos_ids": torch.tensor(list(range(max_seq_length)), dtype=torch.long),
                "nsp_label": torch.tensor(int(nsp_label), dtype=torch.long),
                "mlm_label": torch.tensor(label, dtype=torch.long),
                "mlm_label_mask": torch.tensor(label_mask, dtype=torch.long),
            }
        )
