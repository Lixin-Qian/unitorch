# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import PreTrainedTokenizer
from unitorch.functions import pop_first_non_none_value
from unitorch.models import GenericOutputs, _truncate_seq_pair


class HuggingfaceGenerationProcessor(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Args:
            tokenizer: a huggingface tokenizer
            max_seq_length: max sequence length to encode text
            max_gen_seq_length: max sequence length to decode text
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_gen_seq_length = max_gen_seq_length
        self.pad_token = self.tokenizer.pad_token
        self.bos_token = self.tokenizer.bos_token
        self.sep_token = self.tokenizer.sep_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer.get_vocab())

    def processing_generation(
        self,
        text: str,
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: encode text
            text_pair: decode text
            max_seq_length: max sequence length to encode text
            max_gen_seq_length: max sequence length to decode text
        """
        max_seq_length = pop_first_non_none_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_first_non_none_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )
        tokens_a = self.tokenizer.tokenize(str(text))
        tokens_a = [self.bos_token] + tokens_a[: max_seq_length - 2] + [self.sep_token]
        tokens_ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)

        tokens_mask_a = [1] * len(tokens_ids_a)
        padding_a = [0] * (max_seq_length - len(tokens_ids_a))
        tokens_ids_a += [self.pad_token_id] * len(padding_a)
        tokens_mask_a += padding_a

        assert len(tokens_ids_a) == max_seq_length
        tokens_b = self.tokenizer.tokenize(str(text_pair))
        tokens_b = [self.sep_token] + tokens_b[: max_gen_seq_length - 2] + [self.eos_token]
        tokens_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
        tokens_b_len = len(tokens_ids_b)

        _tokens_ids_b = tokens_ids_b[: tokens_b_len - 1]
        _tokens_ids_t = tokens_ids_b[1:tokens_b_len]
        _tokens_mask_t = [1] * len(_tokens_ids_t)

        padding_b = [0] * (max_gen_seq_length - len(_tokens_ids_b))
        _tokens_ids_b += [self.pad_token_id] * len(padding_b)
        _tokens_mask_b = [1] * (tokens_b_len - 1) + padding_b
        _tokens_ids_t += [self.pad_token_id] * len(padding_b)
        _tokens_mask_t += padding_b
        assert len(_tokens_ids_b) == max_gen_seq_length
        assert len(_tokens_ids_t) == max_gen_seq_length
        assert len(_tokens_mask_t) == max_gen_seq_length

        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids_a, dtype=torch.long),
            tokens_mask=torch.tensor(tokens_mask_a, dtype=torch.long),
            tokens_ids_pair=torch.tensor(_tokens_ids_b, dtype=torch.long),
            tokens_mask_pair=torch.tensor(_tokens_mask_b, dtype=torch.long),
            tokens_ids_target=torch.tensor(_tokens_ids_t, dtype=torch.long),
            tokens_mask_target=torch.tensor(_tokens_mask_t, dtype=torch.long),
        )

    def processing_inference(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: encode text
            max_seq_length: max sequence length to encode text
        """
        max_seq_length = pop_first_non_none_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_seq_length - 2]
        tokens = [self.bos_token] + tokens + [self.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = tokens_ids[:max_seq_length]
        tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += [self.pad_token_id] * len(padding)
        tokens_mask += padding

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            tokens_mask=torch.tensor(tokens_mask, dtype=torch.long),
        )

    def processing_evaluation(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: decode text
            max_gen_seq_length: max sequence length to decode text
        """
        max_gen_seq_length = pop_first_non_none_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_gen_seq_length - 2]
        tokens = [self.sep_token] + tokens + [self.eos_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = tokens_ids[1:max_gen_seq_length]
        tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_gen_seq_length - len(tokens_ids))
        tokens_ids += [self.pad_token_id] * len(padding)
        tokens_mask += padding

        assert len(tokens_ids) == max_gen_seq_length
        assert len(tokens_mask) == max_gen_seq_length
        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            tokens_mask=torch.tensor(tokens_mask, dtype=torch.long),
        )

    def processing_decode(
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = True,
    ):
        """
        Args:
            sequences: generation model output tensor 2-dim or 3-dim
            skip_special_tokens: if skip special tokens
        """
        if sequences.dim() == 3:
            _, num_return_sequences, sequences_length = sequences.size()
            sequences = sequences.reshape(-1, sequences_length).clamp_max(self.vocab_size)
            sequences = sequences.clamp_min(0)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            decode_tokens = self.tokenizer.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
            )
            decode_tokens = [
                decode_tokens[i : i + num_return_sequences] for i in range(0, len(decode_tokens), num_return_sequences)
            ]
        elif sequences.dim() == 2:
            sequences = sequences.clamp_min(0).clamp_max(self.vocab_size)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            decode_tokens = self.tokenizer.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            raise ValueError(f"can't decode the tensor with shape {sequences.shape}")

        return decode_tokens


class HuggingfaceClassificationProcessor(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 1,
        position_start_id: Optional[int] = 0,
    ):
        """
        Args:
            tokenizer: a huggingface tokenizer
            max_seq_length: max sequence length to encode text
            source_type_id: token type id to text_a
            target_type_id: token type id to text_b
            position_start_id: start id of position
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token = self.tokenizer.pad_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.position_start_id = position_start_id

    def processing_classification(
        self,
        text: str,
        text_pair: str = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: encode text
            text_pair: decode text
            max_seq_length: max sequence length to encode text
        """
        max_seq_length = int(
            pop_first_non_none_value(
                max_seq_length,
                self.max_seq_length,
            )
        )

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 2]
            tokens = [self.cls_token] + tokens + [self.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [self.source_type_id] * len(tokens_ids)
            tokens_mask = [1] * len(tokens_ids)
        else:
            tokens_b = self.tokenizer.tokenize(str(text_pair))
            _truncate_seq_pair(tokens, tokens_b, max_seq_length - 3)
            segment_ids = (
                [self.source_type_id]
                + [self.source_type_id] * len(tokens)
                + [self.source_type_id]
                + [self.target_type_id] * len(tokens_b)
                + [self.target_type_id]
            )
            tokens = [self.cls_token] + tokens + [self.sep_token] + tokens_b + [self.sep_token]
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
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            seg_ids=torch.tensor(segment_ids, dtype=torch.long),
            attn_mask=torch.tensor(tokens_mask, dtype=torch.long),
            pos_ids=torch.tensor(
                list(
                    range(
                        self.position_start_id,
                        self.position_start_id + max_seq_length,
                    )
                ),
                dtype=torch.long,
            ),
        )


class GeneralizedProcessor(object):
    def __init__(
        self,
        num_class: Optional[int] = None,
        sep: Optional[str] = ",",
        max_seq_length: Optional[int] = 128,
        map_dict: Optional[Dict] = dict(),
    ):
        """
        Args:
            num_class: num class to classification
            sep: delimiter to input text
            max_seq_length: max sequence length to label sequence
            map_dict: label mapping to input text
        """
        self.num_class = num_class
        self.sep = sep
        self.max_seq_length = max_seq_length
        self.map_dict = map_dict

    def parse_digit(
        self,
        digit: Union[int, float, str],
        dtype: Optional[str] = "int",
    ):
        """
        Args:
            digit: input text/int/float to convert
            dtype: target data type
        Returns: a int/float number
        """
        if isinstance(digit, str):
            assert digit.isdigit()

        if dtype == "int":
            return int(digit)
        return float(digit)

    def processing_digit(
        self,
        digit: Union[int, float, str],
        dtype: Optional[str] = "int",
    ):
        """
        Args:
            digit: input text/int/float to convert
            dtype: target data type
        Returns: a tensor
        """
        if dtype == "int":
            return torch.tensor(self.parse_digit(digit, dtype="int"))
        return torch.tensor(self.parse_digit(digit, dtype="float"))

    def processing_target(
        self,
        text: Union[int, float, str],
        dtype: Optional[str] = "int",
    ):
        """
        Args:
            text: input text to convert
            dtype: target data type
        Returns: a tensor after replacement with map_dict
        """
        if text in self.map_dict:
            text = self.map_dict[text]

        return self.processing_digit(text, dtype=dtype)

    def processing_features(
        self,
        features: Union[List, str],
        sep: Optional[str] = None,
        dtype: Optional[str] = "int",
        shape: Optional[tuple] = None,
    ):
        """
        Args:
            features: input feature list or string to process
            sep: delimiter to split features string
            dtype: target data type
            shape: reshape the process results
        Returns: a tensor after replacement with map_dict
        """
        if isinstance(features, str):
            features = features.split(sep=sep)
        if dtype == "int":
            features = torch.tensor(list(map(int, features)))
        else:
            features = torch.tensor(list(map(float, features)))

        if shape is not None:
            features = features.reshape(shape)

        return features

    def processing_sequence(
        self,
        text: Union[List, str],
        sep: Optional[str] = None,
        dtype: Optional[str] = "int",
    ):
        """
        Args:
            text: input list or string to process
            sep: delimiter to split text
            dtype: target data type
        Returns: a tensor after replacement with map_dict
        """
        if isinstance(text, str):
            sep = pop_first_non_none_value(sep, self.sep)
            tensor = [self.parse_digit(self.map_dict.get(t, t), dtype=dtype) for t in text.split(sep)]
        else:
            tensor = [self.parse_digit(t, dtype=dtype) for t in text]
        return torch.tensor(tensor)

    def processing_multi_target(
        self,
        text: Union[List, str],
        sep: Optional[str] = None,
    ):
        """
        Args:
            text: input list or string to process
            sep: delimiter to split text
        Returns: a tensor after replacement with map_dict
        """
        outputs = torch.zeros(self.num_class)
        if isinstance(text, str):
            sep = pop_first_non_none_value(sep, self.sep)
            indexes = [self.parse_digit(self.map_dict.get(t, t), dtype="int") for t in text.split(sep)]
        else:
            indexes = [self.parse_digit(t, dtype="int") for t in text]
        outputs[indexes] = 1
        return outputs
