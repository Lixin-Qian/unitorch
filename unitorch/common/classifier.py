# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers.activations import quick_gelu


class reslayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        downscale_dim: int,
        output_dim: int,
        use_bn: bool = True,
    ):
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
        add_pre_layer_norm: bool = True,
        add_post_layer_norm: bool = False,
    ):
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
