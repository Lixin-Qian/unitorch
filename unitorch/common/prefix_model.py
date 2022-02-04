# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import torch
import math
import json
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict
from torch import Tensor, device
from functools import partial

from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertIntermediate as PrefixIntermediate,
    BertOutput as PrefixOutput,
    BertPooler as PrefixPooler,
    BertPreTrainedModel as PrefixPreTrainedModel,
    BertSelfOutput as PrefixSelfOutput,
)


def _reorder_buffer(
    attn_cache,
    beam_idx,
):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None and "prefix_" not in k:
            attn_cache[k] = input_buffer_k.index_select(0, beam_idx)
    return attn_cache


def _reorder_buffer_v2(
    attn_cache,
    batch_idx,
    beam_idx,
):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            if "prefix_" in k:
                attn_cache[k] = (
                    input_buffer_k
                    if batch_idx is None
                    else input_buffer_k.index_select(0, batch_idx)
                )
            else:
                attn_cache[k] = (
                    input_buffer_k
                    if beam_idx is None
                    else input_buffer_k.index_select(0, beam_idx)
                )
    return attn_cache


def _relative_position_bucket(
    relative_position,
    bidirectional=True,
    num_buckets=32,
    max_distance=128,
):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    result = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        result += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    result += torch.where(is_small, n, val_if_large)
    return result


class PrefixConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=28996,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=6,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        source_type_id=0,
        target_type_id=1,
        bos_token_id=101,
        mask_token_id=103,
        eos_token_id=102,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.use_cache = True

        if isinstance(vocab_size, str) or (
            sys.version_info[0] == 2 and isinstance(vocab_size, unicode)
        ):
            with open(vocab_size, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
                for key, value in json_config.items():
                    self.__dict__[key] = value
        elif isinstance(vocab_size, int):
            self.vocab_size = vocab_size
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                " or the path to a pretrained model config file (str)"
            )


class PrefixEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=self.position_ids.device,
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
        else:
            token_type_embeddings = torch.zeros_like(inputs_embeds).to(inputs_embeds)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PrefixSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        history_states=None,
        rel_pos=None,
    ):
        """
        Args:
            - hidden_states: last layer output or embedding output
            - attention_mask: mask for tokens per batch
            - history_states: a optimized cache dict for beam search inference
        Shape:
            - hidden_states: (N, S + L, E) for training, (N, 2, E) for cached beam search inference
            - history_states is a optimized cache dict for beam search inference
                - past_prefix_key_layer: (N, S, E)
                - past_prefix_value_layer: (N, S, E)
                - past_key_layer: (N * B, L, E)
                - past_value_layer: (N * B, L, E)
            Note: S is the source sequence length, L is the target sequence length, N is the batch size, E is the embedding dimension, B is the beam size
        """
        new_query_layer = self.query(hidden_states)
        new_key_layer = self.key(hidden_states)
        new_value_layer = self.value(hidden_states)

        past_prefix_key_layer = history_states.get("past_prefix_key_layer")
        past_prefix_value_layer = history_states.get("past_prefix_value_layer")
        past_key_layer = history_states.get("past_key_layer")
        past_value_layer = history_states.get("past_value_layer")

        query_layer = self.transpose_for_scores(new_query_layer)
        key_layer = self.transpose_for_scores(new_key_layer)
        value_layer = self.transpose_for_scores(new_value_layer)
        if past_prefix_key_layer is not None:
            prefix_size = past_prefix_key_layer.size()
            prefix_attention_scores = torch.einsum(
                "bxhtd,bhsd->bxhts",
                query_layer.view(prefix_size[0], -1, *query_layer.size()[1:]),
                past_prefix_key_layer,
            )
            prefix_attention_scores = prefix_attention_scores.reshape(
                -1, *prefix_attention_scores.size()[2:]
            )
            if past_key_layer is not None:
                key_layer = torch.cat((past_key_layer, key_layer), dim=2)
                value_layer = torch.cat((past_value_layer, value_layer), dim=2)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            prefix_attention_scores = prefix_attention_scores / math.sqrt(
                self.attention_head_size
            )
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_scores = torch.cat(
                (prefix_attention_scores, attention_scores), dim=-1
            )
            if rel_pos is not None:
                attention_scores = attention_scores + rel_pos

            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            prefix_attention_probs = attention_probs[:, :, :, : prefix_size[2]]
            attention_probs = attention_probs[:, :, :, prefix_size[2] :]
            prefix_attention_probs = prefix_attention_probs.to(
                past_prefix_value_layer.dtype
            )
            prefix_context_layer = torch.einsum(
                "bxhtd,bhds->bxhts",
                prefix_attention_probs.view(
                    prefix_size[0], -1, *prefix_attention_probs.size()[1:]
                ),
                past_prefix_value_layer,
            )
            prefix_context_layer = prefix_context_layer.reshape(
                -1, *prefix_context_layer.size()[2:]
            )
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = prefix_context_layer + context_layer

        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if rel_pos is not None:
                attention_scores = attention_scores + rel_pos

            attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)

        if history_states is None or len(history_states) == 0:
            history_states.update(
                dict(
                    {
                        "past_prefix_key_layer": key_layer,
                        "past_prefix_value_layer": value_layer,
                    }
                )
            )
        else:
            history_states.update(
                dict(
                    {
                        "past_prefix_key_layer": past_prefix_key_layer,
                        "past_prefix_value_layer": past_prefix_value_layer,
                        "past_key_layer": key_layer[:, :, :-1, :],
                        "past_value_layer": value_layer[:, :, :-1, :],
                    }
                )
            )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class PrefixAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = PrefixSelfAttention(config)
        self.output = PrefixSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = (
            set(heads) - self.pruned_heads
        )  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        input_tensor,
        attention_mask,
        head_mask=None,
        history_states=None,
        rel_pos=None,
    ):
        self_outputs = self.self(
            input_tensor,
            attention_mask,
            head_mask,
            history_states=history_states,
            rel_pos=rel_pos,
        )
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class PrefixLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PrefixAttention(config)
        self.intermediate = PrefixIntermediate(config)
        self.output = PrefixOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        history_states=None,
        rel_pos=None,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            history_states=history_states,
            rel_pos=rel_pos,
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class PrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [PrefixLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        history_states=None,
        rel_pos=None,
    ):
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                history_states=history_states[i],
                rel_pos=rel_pos,
            )

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions), (all encoder layers)


class PrefixTextModel(PrefixPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        self.embeddings = PrefixEmbeddings(config)
        self.encoder = PrefixEncoder(config)
        self.pooler = PrefixPooler(config) if add_pooling_layer else None

        self.enable_relative_position = hasattr(config, "rel_pos_bins")
        if self.enable_relative_position:
            self.rel_pos_bias = nn.Linear(
                config.rel_pos_bins, config.num_attention_heads, bias=False
            )

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        history_states=None,
        rel_pos=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if history_states is None:
            history_states = [
                dict().copy() for _ in range(self.config.num_hidden_layers)
            ]
        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )

        if self.enable_relative_position and rel_pos is not None:
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins).to(
                self.rel_pos_bias.weight
            )
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        elif self.enable_relative_position:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = _relative_position_bucket(
                rel_pos_mat, num_buckets=32, max_distance=128
            )
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins).to(
                self.rel_pos_bias.weight
            )
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            history_states=history_states,
            rel_pos=rel_pos,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        outputs = (
            (
                sequence_output,
                pooled_output,
            )
            + encoder_outputs[1:]
            + (history_states,)
        )  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), (history_states)


class PrefixPixelModel(PrefixPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        self.embeddings = PrefixEmbeddings(config)
        self.encoder = PrefixEncoder(config)
        self.pooler = PrefixPooler(config) if add_pooling_layer else None

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_embedding=None,
        head_mask=None,
        history_states=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        if pixel_embedding is not None:
            pixel_seq_length = pixel_embedding.size(1)
            embedding_output = torch.cat([pixel_embedding, embedding_output], axis=1)
        else:
            pixel_seq_length = 0

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if history_states is None:
            history_states = [
                dict().copy() for _ in range(self.config.num_hidden_layers)
            ]

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            history_states=history_states,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = (
            self.pooler(sequence_output[:, pixel_seq_length:])
            if self.pooler is not None
            else None
        )

        outputs = (
            (
                sequence_output,
                pooled_output,
            )
            + encoder_outputs[1:]
            + (history_states,)
        )  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), (history_states)
