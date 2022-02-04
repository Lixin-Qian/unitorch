# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.file_utils import is_remote_url
from unitorch import hf_cached_path


# generic model & outputs
class GenericModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def from_checkpoint(
        self,
        ckpt_dir="./cache",
        weight_name="pytorch_model.bin",
        **kwargs,
    ):
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict)
        logging.info(
            f"{type(self).__name__} model load weight from checkpoint {weight_path}"
        )

    def save_checkpoint(
        self,
        ckpt_dir="./cache",
        weight_name="pytorch_model.bin",
        **kwargs,
    ):
        state_dict = self.state_dict()
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(state_dict, weight_path)
        logging.info(f"{type(self).__name__} model save checkpoint to {weight_path}")

    def from_pretrained(
        self,
        weight_path=None,
        replace_keys: Dict = dict(),
        **kwargs,
    ):
        if "state_dict" in kwargs:
            state_dict = kwargs.pop("state_dict")
            _self_state_dict = self.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in _self_state_dict and v.shape == _self_state_dict[k].shape
            }
            self.load_state_dict(state_dict, False)
            return

        if weight_path is None:
            return

        if not (is_remote_url(weight_path) or os.path.exists(weight_path)):
            return

        weight_path = hf_cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")

        if "gamma" not in replace_keys:
            replace_keys["gamma"] = "weight"
        if "beta" not in replace_keys:
            replace_keys["beta"] = "bias"

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            for rkey, nkey in replace_keys.items():
                if rkey not in key:
                    continue
                if new_key is None:
                    new_key = key.replace(rkey, nkey)
                else:
                    new_key = new_key.replace(rkey, nkey)

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        self.load_state_dict(state_dict, False)
        logging.info(
            f"{type(self).__name__} model load weight from pretrain {weight_path}"
        )


class GenericOutputs(object):
    def __init__(
        self,
        attrs: Dict = dict(),
        **kwargs,
    ):
        for k, v in {**attrs, **kwargs}.items():
            setattr(self, k, v)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


from unitorch.models.processing_utils import (
    HuggingfaceGenerationProcessor,
    HuggingfaceClassificationProcessor,
)

# import models
import unitorch.models.bart
import unitorch.models.bert
import unitorch.models.deberta
import unitorch.models.detectron2
import unitorch.models.mass
import unitorch.models.mbart
import unitorch.models.prophetnet
import unitorch.models.roberta
import unitorch.models.unilm
import unitorch.models.infoxlm
import unitorch.models.vlp
import unitorch.models.xprophetnet
import unitorch.models.vit
import unitorch.models.vit_mae
import unitorch.models.swin
import unitorch.models.detr
