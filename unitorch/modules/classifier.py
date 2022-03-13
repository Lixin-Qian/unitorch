# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers.activations import quick_gelu
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


class reslayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        downscale_dim: int,
        output_dim: int,
        use_bn: Optional[bool] = True,
    ):
        """
        Net Structure: ` in -> fc1 -> bn -> relu -> fc2 + in -> relu`
        Args:
            - feature_dim: the input feature dim
            - downscale_dim: the downscale dim
            - output_dim: the output dim
            - use_bn: if add bn between fc1 & relu
        """
        super().__init__()
        assert feature_dim == output_dim
        self.fc1 = nn.Linear(feature_dim, downscale_dim)
        self.fc2 = nn.Linear(downscale_dim, output_dim)
        self.bn = nn.BatchNorm1d(downscale_dim) if use_bn else None

        # init weight
        self.fc1.weight.data.normal_(mean=0.0, std=0.02)
        self.fc1.bias.data.zero_()

        self.fc2.weight.data.normal_(mean=0.0, std=0.02)
        self.fc2.bias.data.zero_()

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: the input 2d tensor
        """
        output = self.fc1(input)
        if self.bn is not None:
            output = self.bn(output)
        output = torch.relu(output)
        output = self.fc2(output) + input
        return torch.relu(output)


class mlplayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        downscale_dim: int,
        output_dim: int,
        add_pre_layer_norm: Optional[bool] = True,
        add_post_layer_norm: Optional[bool] = False,
    ):
        """
        Net Structure: ` in -> pre_layer_norm -> fc1 -> gelu -> fc2 + in -> post_layer_norm`
        Args:
            - feature_dim: the input feature dim
            - downscale_dim: the downscale dim
            - output_dim: the output dim
            - add_pre_layer_norm: if use pre_layer_norm between in & fc1
            - add_post_layer_norm: if use post_layer_norm after fc2
        """
        super().__init__()
        assert feature_dim == output_dim
        self.fc1 = nn.Linear(feature_dim, downscale_dim)
        self.fc2 = nn.Linear(downscale_dim, output_dim)

        # init weight
        self.fc1.weight.data.normal_(mean=0.0, std=0.02)
        self.fc1.bias.data.zero_()

        self.fc2.weight.data.normal_(mean=0.0, std=0.02)
        self.fc2.bias.data.zero_()

        self.add_pre_layer_norm = add_pre_layer_norm
        self.pre_layer_norm = nn.LayerNorm(output_dim) if add_pre_layer_norm else None

        self.add_post_layer_norm = add_post_layer_norm
        self.post_layer_norm = nn.LayerNorm(output_dim) if add_post_layer_norm else None

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: the input 2d tensor
        """
        if self.add_pre_layer_norm:
            output = self.pre_layer_norm(input)
        else:
            output = input
        output = self.fc1(output)
        output = quick_gelu(output)
        output = self.fc2(output) + input
        if self.add_post_layer_norm:
            output = self.post_layer_norm(output)
        return output
