# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.file_utils import is_remote_url
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from detectron2.modeling.roi_heads import StandardROIHeads
from unitorch.modules.prefix_model import (
    PrefixConfig,
    PrefixPixelModel,
    _reorder_buffer,
    _reorder_buffer_v2,
)
from unitorch import hf_cached_path
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.detectron2 import GeneralizedRCNN


class _VLPConfig(PrefixConfig):
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
        region_feature_size=2048,
        region_position_size=1607,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            bos_token_id=bos_token_id,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.region_feature_size = region_feature_size
        self.region_position_size = region_position_size


class _VLPRCNN(GeneralizedRCNN):
    def __init__(self, detectron2_config_path):
        super().__init__(detectron2_config_path)

    def _get_box_features(self, features, proposals):
        assert isinstance(self.roi_heads, StandardROIHeads)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.box_in_features], proposal_boxes)
        fc1_features = self.roi_heads.box_head.flatten(box_features)
        fc1_features = self.roi_heads.box_head.fc1(fc1_features)
        fc1_features = self.roi_heads.box_head.fc_relu1(fc1_features)
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)
        _, prediction_indexes = self.roi_heads.box_predictor.inference(predictions, proposals)
        num_proposals = [0] + [len(prop) for prop in proposals]
        num_proposals = list(accumulate(num_proposals))

        if len(predictions) > 1:
            predictions = predictions[0]

        cls_prob = [
            predictions[num_proposals[i] : num_proposals[i + 1]][prediction_indexes[i]].to(self.dtype)
            for i in range(len(prediction_indexes))
        ]
        obj_feat = [
            fc1_features[num_proposals[i] : num_proposals[i + 1]][prediction_indexes[i]].to(self.dtype)
            for i in range(len(prediction_indexes))
        ]
        return GenericOutputs(cls_prob=cls_prob, obj_feat=obj_feat)

    def prepare_features(self, images):
        features = self.detect(images, return_features=True)

        bboxes = features.bboxes
        scores = features.scores
        classes = features.classes
        obj_feat = features.features.obj_feat
        cls_prob = features.features.cls_prob

        vis_pe = [
            torch.cat([b, c.unsqueeze(-1), s.unsqueeze(-1), p[:, 1:]], axis=1).to(self.dtype)
            for b, s, c, p in zip(bboxes, classes, scores, cls_prob)
        ]

        return GenericOutputs(vis_feat=obj_feat, vis_pe=vis_pe)


class VLPForGeneration(GenericModel, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(
        self,
        vlp_config_path: str,
        detectron2_config_path: str,
        freeze_vision_model: Optional[bool] = True,
        max_num_bbox: Optional[int] = 100,
    ):
        """
        Args:
            vlp_config_path: config file path to text part of vlp model
            detectron2_config_path: config file path to image part of vlp model (faster-rcnn based on detectron2)
            freeze_vision_model: if to freeze image part of model
            max_num_bbox: max num bbox returns from faster-rcnn
        """
        super().__init__()
        self.config = _VLPConfig.from_json_file(vlp_config_path)
        self.config.gradient_checkpointing = False
        self.freeze_vision_model = freeze_vision_model
        self.max_num_bbox = max_num_bbox

        self.bert = PrefixPixelModel(self.config)
        self.cls = BertOnlyMLMHead(self.config)

        self.vision_model = _VLPRCNN(detectron2_config_path)
        self.vision_embedding = nn.Sequential(
            nn.Linear(self.config.region_feature_size, self.config.region_feature_size),
            nn.ReLU(),
            nn.Linear(self.config.region_feature_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob),
        )

        self.vision_position_embedding = nn.Sequential(
            nn.Linear(self.config.region_position_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob),
        )

        self.vision_type_embedding = self.bert.embeddings.token_type_embeddings
        self.vision_embedding_layer_norm = self.bert.embeddings.LayerNorm
        self.vision_embedding_dropout = self.bert.embeddings.dropout

        self.init_weights()

        self.hist_index = int(self.config.output_hidden_states) + int(self.config.output_attentions) + 2
        self.bert.embeddings.word_embeddings.weight = self.cls.predictions.decoder.weight

        if freeze_vision_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_vision_model:
            self.vision_model.train(False)
        return self

    def from_pretrained(self, weight_path):
        """
        Load model's pretrained weight
        Args:
            weight_path: the path of pretrained weight of mbart
        """
        if not (is_remote_url(weight_path) or os.path.exists(weight_path)):
            return
        weight_path = hf_cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "vis_embed" in key:
                new_key = key.replace("vis_embed", "vision_embedding")
            if "vis_pe_embed" in key:
                new_key = key.replace("vis_pe_embed", "vision_position_embedding")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        self.load_state_dict(state_dict, False)
        logging.info(f"{type(self).__name__} model load weight from pretrain {weight_path}")

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(self.parameters()).device

    def _prepare_pixel_embedding_mask(self, pixel_values):
        outputs = self.vision_model.prepare_features(pixel_values)
        vis_feat, vis_pe = outputs.vis_feat, outputs.vis_pe
        _vis_group = []
        for f, p in zip(vis_feat, vis_pe):
            f = f[: self.max_num_bbox]
            p = p[: self.max_num_bbox]

            _f = F.pad(f, [0, 0, 0, self.max_num_bbox - f.size(0)], "constant", 0)
            _p = F.pad(p, [0, 0, 0, self.max_num_bbox - p.size(0)], "constant", 0)
            _m = torch.tensor([1] * p.size(0) + [0] * (self.max_num_bbox - p.size(0))).to(p.device)

            _vis_group.append([_f, _p, _m])

        vis_feat, vis_pe, vis_mask = zip(*_vis_group)
        vis_feat, vis_pe, vis_mask = (
            torch.stack(vis_feat, dim=0),
            torch.stack(vis_pe, dim=0),
            torch.stack(vis_mask, dim=0),
        )
        vis_feat = self.vision_embedding(vis_feat)
        vis_pe = self.vision_position_embedding(vis_pe).to(vis_feat)
        vis_seg = self.vision_type_embedding(
            self.config.source_type_id * torch.ones(vis_feat.size()[:2]).to(vis_feat.device).long()
        ).to(vis_feat)

        embeddings = self.vision_embedding_layer_norm(vis_feat + vis_pe + vis_seg)

        embeddings_output = self.vision_embedding_dropout(embeddings)
        embeddings_mask = vis_mask
        return GenericOutputs(
            embeddings_output=embeddings_output,
            embeddings_mask=embeddings_mask,
        )

    def _prepare_pixel_attention_mask(self, pixel_mask, text_mask):
        pixel_length = pixel_mask.size(-1)
        text_length = text_mask.size(-1)
        new_seq_len = pixel_length + text_length
        attn_mask = torch.zeros(pixel_mask.size(0), new_seq_len, new_seq_len).to(text_mask)

        assert pixel_mask.dim() == 2
        attn1 = attn_mask[:, :, :pixel_length] + pixel_mask[:, None, :]
        attn2 = attn_mask[:, :pixel_length, pixel_length:] + text_mask[:, 0, :][:, None, :]
        attn_mask[:, :, :pixel_length].copy_(attn1)
        attn_mask[:, :pixel_length, pixel_length:].copy_(attn2)
        attn_mask[:, pixel_length:, pixel_length:].copy_(text_mask)

        return attn_mask

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        **kwargs,
    ):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        if past is None:
            active_batch_size, _ = decoder_input_ids.size()
            prefix_token, prefix_seg, prefix_pos, prefix_mask, pixel_values = (
                self.prefix_state["prefix_token"],
                self.prefix_state["prefix_seg"],
                self.prefix_state["prefix_pos"],
                self.prefix_state["prefix_mask"],
                self.prefix_state["pixel_values"],
            )
            prefix_len = self.prefix_state["prefix_len"]
            pixel_embedding_mask = self._prepare_pixel_embedding_mask(pixel_values)
            pixel_embedding = pixel_embedding_mask.embeddings_output
            pixel_mask = pixel_embedding_mask.embeddings_mask
            pixel_len = pixel_mask.size(-1)
            prefix_mask = self._prepare_pixel_attention_mask(pixel_mask, prefix_mask)
            outputs = self.bert(
                prefix_token[:, :prefix_len],
                prefix_seg[:, :prefix_len],
                prefix_mask[:, : prefix_len + pixel_len, : prefix_len + pixel_len],
                prefix_pos[:, :prefix_len],
                pixel_embedding,
            )
            token_pos = prefix_pos.repeat(1, self.num_beams).view(active_batch_size, prefix_pos.size(1))
            token_pos = token_pos[:, prefix_len:]
            token_mask = (
                prefix_mask.unsqueeze(1)
                .repeat(1, self.num_beams, 1, 1)
                .view(active_batch_size, prefix_mask.size(1), prefix_mask.size(1))
            )
            token_mask = token_mask[:, prefix_len + pixel_len :, :]
            history_states = outputs[self.hist_index]
            decoder_mask_token = torch.ones(active_batch_size, 1).to(decoder_input_ids) * self.config.mask_token_id
            decoder_seg_ids = torch.ones(active_batch_size, 2).to(decoder_input_ids) * self.config.target_type_id
            pixel_mask = pixel_mask.repeat(1, 2 * self.num_beams).view(active_batch_size, 2, pixel_mask.size(1))
        else:
            (token_pos, token_mask, decoder_mask_token, decoder_seg_ids, pixel_mask, history_states,) = (
                past[0],
                past[1],
                past[2],
                past[3],
                past[4],
                past[5:],
            )
        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_mask_ids": decoder_mask_token,
            "decoder_attn_mask": token_mask,
            "decoder_seg_ids": decoder_seg_ids,
            "decoder_pos_ids": token_pos,
            "decoder_pixel_mask": pixel_mask,
            "past_key_values": history_states,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        For beam search in huggingface generation mixin
        """
        (pos_ids, token_mask, decoder_mask_token, decoder_seg, pixel_mask, history_states,) = (
            past[0],
            past[1],
            past[2],
            past[3],
            past[4],
            past[5:],
        )
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(_reorder_buffer(layer_past, beam_idx))
        newpast = [
            pos_ids,
            token_mask,
            decoder_mask_token,
            decoder_seg,
            pixel_mask,
        ] + reordered_past
        return newpast

    @staticmethod
    def _reorder_cache_v2(past, batch_idx, beam_idx):
        """
        For faster inference by optimized beam search in generation mixin v2
        """
        (pos_ids, token_mask, decoder_mask_token, decoder_seg, pixel_mask, history_states,) = (
            past[0],
            past[1],
            past[2],
            past[3],
            past[4],
            past[5:],
        )
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(_reorder_buffer_v2(layer_past, batch_idx, beam_idx))
        pos_ids = pos_ids[beam_idx]
        token_mask = token_mask[beam_idx]
        decoder_mask_token = decoder_mask_token[beam_idx]
        decoder_seg = decoder_seg[beam_idx]
        pixel_mask = pixel_mask[beam_idx]
        newpast = [
            pos_ids,
            token_mask,
            decoder_mask_token,
            decoder_seg,
            pixel_mask,
        ] + reordered_past
        return newpast

    def forward(
        self,
        tokens_ids=None,
        attn_mask=None,
        seg_ids=None,
        pos_ids=None,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_pos_ids=None,
        decoder_seg_ids=None,
        decoder_attn_mask=None,
        decoder_mask_ids=None,
        decoder_pixel_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Args:
            tokens_ids: tokens of encode text & decode
            attn_mask: attention mask of tokens
            seg_ids: token type ids
            pos_ids: position ids
            pixel_values: pixels of images
            others: used in beam search
        Returns: forward logits
        """
        if self.training:
            pixel_embedding_mask = self._prepare_pixel_embedding_mask(pixel_values)
            pixel_embedding = pixel_embedding_mask.embeddings_output
            pixel_mask = pixel_embedding_mask.embeddings_mask

            attn_mask = self._prepare_pixel_attention_mask(pixel_mask, attn_mask)
            outputs = self.bert(
                tokens_ids,
                seg_ids,
                attn_mask,
                pos_ids,
                pixel_embedding,
            )
            logits = self.cls(outputs[0])
            logits = logits[:, self.max_num_bbox :]
            return logits
        decoder_token = torch.cat([decoder_input_ids, decoder_mask_ids], dim=1)
        decoder_len = decoder_token.size(1)
        decoder_token = decoder_token[:, -2:]
        pixel_length = decoder_pixel_mask.size(-1)
        decoder_mask = decoder_attn_mask[
            :,
            decoder_len - 2 : decoder_len,
            : self.prefix_state["prefix_len"] + decoder_len + pixel_length,
        ]
        # decoder_mask = torch.cat([decoder_pixel_mask, decoder_mask], dim=-1)
        decoder_pos = decoder_pos_ids[:, decoder_len - 2 : decoder_len]
        outputs = self.bert(
            decoder_token,
            decoder_seg_ids,
            decoder_mask,
            decoder_pos,
            history_states=past_key_values,
        )
        logits = self.cls(outputs[0])
        state4cache = [
            decoder_pos_ids,
            decoder_attn_mask,
            decoder_mask_ids,
            decoder_seg_ids,
            decoder_pixel_mask,
        ] + outputs[self.hist_index]
        return Seq2SeqLMOutput(logits=logits, past_key_values=state4cache)

    def generate(
        self,
        pixel_values,
        tokens_ids=None,
        num_beams=5,
        decoder_start_token_id=101,
        decoder_end_token_id=102,
        num_return_sequences=1,
        min_gen_seq_length=0,
        max_gen_seq_length=48,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        length_penalty=1.0,
        num_beam_groups=1,
        diversity_penalty=0.0,
        diverse_rate=0.0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    ):
        """
        Args:
            tokens_ids: tokens of encode text
            pixel_values: pixels of images
        """
        self.num_beams = num_beams
        if decoder_start_token_id is not None:
            self.config.bos_token_id = decoder_start_token_id

        prefix_token = torch.ones(len(pixel_values), 2).to(
            device=pixel_values.device if isinstance(pixel_values, torch.Tensor) else pixel_values[0].device
        )
        prefix_token *= torch.tensor([self.config.bos_token_id, self.config.eos_token_id]).to(prefix_token)
        prefix_token = prefix_token.long()

        if tokens_ids is not None:
            prefix_token = torch.cat([prefix_token[:, :-1], tokens_ids], dim=-1)

        prefix_mask1 = prefix_token.ne(self.config.pad_token_id).long()
        batch_size, prefix_len = prefix_token.size()
        total_seq_length = max_gen_seq_length + prefix_len + 1
        prefix_mask = prefix_mask1[:, None, :].repeat(1, total_seq_length, 1)
        new_mask = torch.zeros(batch_size, total_seq_length, max_gen_seq_length + 1).to(prefix_mask)
        tri_mask = torch.ones(batch_size, total_seq_length, max_gen_seq_length + 1).to(prefix_mask)
        new_mask[:, prefix_len:, :] = torch.tril(tri_mask[:, prefix_len:, :])
        new_mask[:, :, 0] = 0
        prefix_mask = torch.cat((prefix_mask, new_mask), dim=-1)
        prefix_seg = torch.tensor([self.config.source_type_id] * prefix_len).to(prefix_token)
        prefix_seg = prefix_seg[None, :].repeat(batch_size, 1)
        prefix_pos0 = torch.ones(batch_size, max_gen_seq_length + 1).to(prefix_token)
        prefix_pos0[:, 0] = 0
        prefix_pos = torch.cat((prefix_token, prefix_pos0.to(prefix_token)), dim=-1).ne(self.config.pad_token_id)
        prefix_pos = torch.cumsum(prefix_pos, dim=-1) - 1

        self.prefix_state = dict(
            {
                "prefix_len": prefix_len,
                "prefix_token": prefix_token,
                "prefix_seg": prefix_seg,
                "prefix_mask": prefix_mask,
                "prefix_pos": prefix_pos,
                "pixel_values": pixel_values,
            }
        )
        decoder_seg = (torch.ones(batch_size * self.num_beams, 1) * self.config.target_type_id).to(prefix_token)
        decoder_seg[:, 0] = self.config.source_type_id
        decoder_mask_token = torch.ones(batch_size * self.num_beams, 1).to(prefix_token) * self.config.mask_token_id

        decoder_input_ids = torch.ones(batch_size, 1).to(prefix_token) * self.config.bos_token_id

        outputs = super().generate(
            decoder_input_ids,
            max_length=max_gen_seq_length,
            min_length=min_gen_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            decoder_start_token_id=decoder_start_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences.reshape(-1, num_return_sequences, outputs.sequences.size(-1))
        outputs.sequences = torch.zeros(sequences.size(0), num_return_sequences, max_gen_seq_length).to(
            device=sequences.device
        )
        outputs.sequences[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(sequences=outputs.sequences, sequences_scores=outputs.sequences_scores)


class VLPForClassification(GenericModel):
    def __init__(
        self,
        vlp_config_path,
        detectron2_config_path,
        freeze_vision_model: bool = True,
        freeze_base_model: bool = False,
        max_num_bbox: int = 100,
        num_class: int = 1,
    ):
        super().__init__()
        self.config = _VLPConfig.from_json_file(vlp_config_path)
        self.config.gradient_checkpointing = False
        self.freeze_vision_model = freeze_vision_model
        self.max_num_bbox = max_num_bbox
        self.num_class = num_class

        self.bert = PrefixPixelModel(self.config, add_pooling_layer=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_class)

        self.vision_model = _VLPRCNN(detectron2_config_path)
        self.vision_embedding = nn.Sequential(
            nn.Linear(self.config.region_feature_size, self.config.region_feature_size),
            nn.ReLU(),
            nn.Linear(self.config.region_feature_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob),
        )

        self.vision_position_embedding = nn.Sequential(
            nn.Linear(self.config.region_position_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob),
        )

        self.vision_type_embedding = self.bert.embeddings.token_type_embeddings
        self.vision_embedding_layer_norm = self.bert.embeddings.LayerNorm
        self.vision_embedding_dropout = self.bert.embeddings.dropout

        self.init_weights()

        if freeze_vision_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        if freeze_base_model:
            for p in self.bert.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_vision_model:
            self.vision_model.train(False)
        return self

    def from_pretrained(self, weight_path):
        """
        Load model's pretrained weight
        Args:
            weight_path: the path of pretrained weight of mbart
        """
        if not (is_remote_url(weight_path) or os.path.exists(weight_path)):
            return
        weight_path = hf_cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "vis_embed" in key:
                new_key = key.replace("vis_embed", "vision_embedding")
            if "vis_pe_embed" in key:
                new_key = key.replace("vis_pe_embed", "vision_position_embedding")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        self.load_state_dict(state_dict, False)
        logging.info(f"{type(self).__name__} model load weight from pretrain {weight_path}")

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(self.parameters()).device

    def _prepare_pixel_embedding_mask(self, pixel_values):
        outputs = self.vision_model.prepare_features(pixel_values)
        vis_feat, vis_pe = outputs.vis_feat, outputs.vis_pe
        _vis_group = []
        for f, p in zip(vis_feat, vis_pe):
            f = f[: self.max_num_bbox]
            p = p[: self.max_num_bbox]

            _f = F.pad(f, [0, 0, 0, self.max_num_bbox - f.size(0)], "constant", 0)
            _p = F.pad(p, [0, 0, 0, self.max_num_bbox - p.size(0)], "constant", 0)
            _m = torch.tensor([1] * p.size(0) + [0] * (self.max_num_bbox - p.size(0))).to(p.device)

            _vis_group.append([_f, _p, _m])

        vis_feat, vis_pe, vis_mask = zip(*_vis_group)
        vis_feat, vis_pe, vis_mask = (
            torch.stack(vis_feat, dim=0),
            torch.stack(vis_pe, dim=0),
            torch.stack(vis_mask, dim=0),
        )
        vis_feat = self.vision_embedding(vis_feat)
        vis_pe = self.vision_position_embedding(vis_pe).to(vis_feat)
        vis_seg = self.vision_type_embedding(
            self.config.source_type_id * torch.ones(vis_feat.size()[:2]).to(vis_feat.device).long()
        ).to(vis_feat)

        embeddings = self.vision_embedding_layer_norm(vis_feat + vis_pe + vis_seg)

        embeddings_output = self.vision_embedding_dropout(embeddings)
        embeddings_mask = vis_mask
        return GenericOutputs(
            embeddings_output=embeddings_output,
            embeddings_mask=embeddings_mask,
        )

    def _prepare_pixel_attention_mask(self, pixel_mask, text_mask):
        pixel_length = pixel_mask.size(-1)
        text_length = text_mask.size(-1)
        new_seq_len = pixel_length + text_length
        attn_mask = torch.zeros(pixel_mask.size(0), new_seq_len).to(text_mask)

        assert pixel_mask.dim() == 2
        attn_mask[:, :pixel_length].copy_(pixel_mask)

        return attn_mask

    def forward(
        self,
        tokens_ids=None,
        attn_mask=None,
        seg_ids=None,
        pos_ids=None,
        pixel_values=None,
    ):
        """
        Args:
            tokens_ids: tokens of encode text & decode
            attn_mask: attention mask of tokens
            seg_ids: token type ids
            pos_ids: position ids
            pixel_values: pixels of images
        Returns: forward logits
        """
        pixel_embedding_mask = self._prepare_pixel_embedding_mask(pixel_values)
        pixel_embedding = pixel_embedding_mask.embeddings_output
        pixel_mask = pixel_embedding_mask.embeddings_mask

        attn_mask = self._prepare_pixel_attention_mask(pixel_mask, attn_mask)
        outputs = self.bert(
            tokens_ids,
            seg_ids,
            attn_mask,
            pos_ids,
            pixel_embedding,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
