# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# some functions
def convert_tensor_to_2D_strings(inputs: torch.Tensor):
    return [list(map(str, t1)) for t1 in inputs]


def convert_tensor_to_3D_strings(inputs: torch.Tensor):
    return [[list(map(str, t2)) for t2 in t1] for t1 in inputs]


def remove_2D_strings_ignore_tokens(inputs, ignore_tokens):
    if ignore_tokens is None:
        return inputs
    return [list(filter(lambda x: x not in ignore_tokens, t1)) for t1 in inputs]


def remove_3D_strings_ignore_tokens(inputs, ignore_tokens):
    if ignore_tokens is None:
        return inputs
    return [
        [list(filter(lambda x: x not in ignore_tokens, t2)) for t2 in t1]
        for t1 in inputs
    ]


from unitorch.score.bleu import bleu_score
from unitorch.score.rouge import rouge1_score, rouge2_score, rougel_score
from unitorch.score.voc_map import voc_map_score
