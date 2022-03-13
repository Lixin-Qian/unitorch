# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from unitorch.functions import pop_first_non_none_value
from unitorch.models import GenericOutputs


class CLIPProcessor(object):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        """
        Args:
            vocab_path: vocab file path in bart tokenizer
            merge_path: merge file path in bart tokenizer
            vision_config_path: vision config path to clip processor
            max_seq_length: max sequence length encode text
            position_start_id: start id of position
        """
        self.tokenizer = CLIPTokenizer(
            vocab_file=vocab_path,
            merges_file=merge_path,
        )
        self.vision_processor = CLIPFeatureExtractor.from_json_file(vision_config_path)

        self.max_seq_length = max_seq_length
        self.position_start_id = position_start_id

        self.pad_token = self.tokenizer.pad_token
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.source_type_id = 0

        self.size = self.vision_processor.size
        self.resample = self.vision_processor.resample
        self.crop_size = self.vision_processor.crop_size
        self.image_mean = self.vision_processor.image_mean
        self.image_std = self.vision_processor.image_std

    def processing_classification(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: input text
            image: input image
            max_seq_length: max sequence length to input text
        """
        max_seq_length = int(
            pop_first_non_none_value(
                max_seq_length,
                self.max_seq_length,
            )
        )

        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_seq_length - 2]
        tokens = [self.bos_token] + tokens + [self.eos_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [self.source_type_id] * len(tokens_ids)
        tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += len(padding) * [self.pad_token_id]
        tokens_mask += padding

        image = self.vision_processor.resize(
            image=image,
            size=self.size,
            resample=self.resample,
        )
        image = self.vision_processor.center_crop(
            image,
            self.crop_size,
        )
        image = self.vision_processor.normalize(
            image=image,
            mean=self.image_mean,
            std=self.image_std,
        )

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
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
            image=torch.tensor(image),
        )

    def processing_text_classifictaion(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: input text
            max_seq_length: max sequence length to input text
        """
        max_seq_length = int(
            pop_first_non_none_value(
                max_seq_length,
                self.max_seq_length,
            )
        )

        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_seq_length - 2]
        tokens = [self.bos_token] + tokens + [self.eos_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [self.source_type_id] * len(tokens_ids)
        tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += len(padding) * [self.pad_token_id]
        tokens_mask += padding

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
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

    def processing_image_classifictaion(
        self,
        image: Image.Image,
    ):
        """
        Args:
            image: input image
        """
        image = self.vision_processor.resize(
            image=image,
            size=self.size,
            resample=self.resample,
        )
        image = self.vision_processor.center_crop(
            image,
            self.crop_size,
        )
        image = self.vision_processor.normalize(
            image=image,
            mean=self.image_mean,
            std=self.image_std,
        )

        return GenericOutputs(
            image=torch.tensor(image),
        )
