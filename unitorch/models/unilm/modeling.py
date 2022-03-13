# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from unitorch.modules.prefix_model import (
    PrefixConfig,
    PrefixTextModel,
    _reorder_buffer,
    _reorder_buffer_v2,
)
from unitorch.models import GenericModel, GenericOutputs


class UnilmForGeneration(GenericModel, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self, config_path):
        """
        Args:
            config_path: config file path to unilm model
        """
        super().__init__()
        self.config = PrefixConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = False
        self.bert = PrefixTextModel(self.config)
        self.cls = BertOnlyMLMHead(self.config)
        self.init_weights()

        self.hist_index = int(self.config.output_hidden_states) + int(self.config.output_attentions) + 2
        self.bert.embeddings.word_embeddings.weight = self.cls.predictions.decoder.weight

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(self.parameters()).device

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
            prefix_token, prefix_seg, prefix_pos, prefix_mask = (
                self.prefix_state["prefix_token"],
                self.prefix_state["prefix_seg"],
                self.prefix_state["prefix_pos"],
                self.prefix_state["prefix_mask"],
            )
            prefix_len = self.prefix_state["prefix_len"]
            outputs = self.bert(
                prefix_token[:, :prefix_len],
                prefix_seg[:, :prefix_len],
                prefix_mask[:, :prefix_len, :prefix_len],
                prefix_pos[:, :prefix_len],
            )
            token_pos = prefix_pos.repeat(1, self.num_beams).view(active_batch_size, prefix_pos.size(1))
            token_pos = token_pos[:, prefix_len:]
            token_mask = (
                prefix_mask.unsqueeze(1)
                .repeat(1, self.num_beams, 1, 1)
                .view(active_batch_size, prefix_mask.size(1), prefix_mask.size(1))
            )
            token_mask = token_mask[:, prefix_len:, :]
            history_states = outputs[self.hist_index]
            decoder_mask_token = torch.ones(active_batch_size, 1).to(decoder_input_ids) * self.config.mask_token_id
            decoder_seg_ids = torch.ones(active_batch_size, 2).to(decoder_input_ids) * self.config.target_type_id
        else:
            (token_pos, token_mask, decoder_mask_token, decoder_seg_ids, history_states,) = (
                past[0],
                past[1],
                past[2],
                past[3],
                past[4:],
            )
        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_mask_ids": decoder_mask_token,
            "decoder_attn_mask": token_mask,
            "decoder_seg_ids": decoder_seg_ids,
            "decoder_pos_ids": token_pos,
            "past_key_values": history_states,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        For beam search in huggingface generation mixin
        """
        (pos_ids, token_mask, decoder_mask_token, decoder_seg, history_states,) = (
            past[0],
            past[1],
            past[2],
            past[3],
            past[4:],
        )
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(_reorder_buffer(layer_past, beam_idx))
        newpast = [
            pos_ids,
            token_mask,
            decoder_mask_token,
            decoder_seg,
        ] + reordered_past
        return newpast

    @staticmethod
    def _reorder_cache_v2(past, batch_idx, beam_idx):
        """
        For faster inference by optimized beam search in generation mixin v2
        """
        (pos_ids, token_mask, decoder_mask_token, decoder_seg, history_states,) = (
            past[0],
            past[1],
            past[2],
            past[3],
            past[4:],
        )
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(_reorder_buffer_v2(layer_past, batch_idx, beam_idx))
        pos_ids = pos_ids[beam_idx]
        token_mask = token_mask[beam_idx]
        decoder_mask_token = decoder_mask_token[beam_idx]
        decoder_seg = decoder_seg[beam_idx]
        newpast = [
            pos_ids,
            token_mask,
            decoder_mask_token,
            decoder_seg,
        ] + reordered_past
        return newpast

    def forward(
        self,
        tokens_ids=None,
        attn_mask=None,
        seg_ids=None,
        pos_ids=None,
        decoder_input_ids=None,
        decoder_pos_ids=None,
        decoder_seg_ids=None,
        decoder_attn_mask=None,
        decoder_mask_ids=None,
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
            others: used in beam search
        Returns: forward logits
        """
        if self.training:
            outputs = self.bert(
                tokens_ids,
                seg_ids,
                attn_mask,
                pos_ids,
            )
            logits = self.cls(outputs[0])
            return logits
        decoder_token = torch.cat([decoder_input_ids, decoder_mask_ids], dim=1)
        decoder_len = decoder_token.size(1)
        decoder_token = decoder_token[:, -2:]
        decoder_mask = decoder_attn_mask[
            :,
            decoder_len - 2 : decoder_len,
            : self.prefix_state["prefix_len"] + decoder_len,
        ]
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
        ] + outputs[self.hist_index]
        return Seq2SeqLMOutput(logits=logits, past_key_values=state4cache)

    def generate(
        self,
        tokens_ids,
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
        """
        self.num_beams = num_beams
        prefix_token = tokens_ids
        prefix_mask1 = tokens_ids.ne(self.config.pad_token_id).long()
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
        prefix_pos0 = torch.ones(batch_size, max_gen_seq_length + 1).to(tokens_ids)
        prefix_pos0[:, 0] = 0
        prefix_pos = torch.cat((tokens_ids, prefix_pos0.to(tokens_ids)), dim=-1).ne(self.config.pad_token_id)
        prefix_pos = torch.cumsum(prefix_pos, dim=-1) - 1

        self.prefix_state = dict(
            {
                "prefix_len": prefix_len,
                "prefix_token": prefix_token,
                "prefix_seg": prefix_seg,
                "prefix_mask": prefix_mask,
                "prefix_pos": prefix_pos,
            }
        )
        decoder_seg = (torch.ones(batch_size * self.num_beams, 1) * self.config.target_type_id).to(prefix_token)
        decoder_seg[:, 0] = self.config.source_type_id
        decoder_mask_token = torch.ones(batch_size * self.num_beams, 1).to(prefix_token) * self.config.mask_token_id
        if decoder_start_token_id is not None:
            self.config.bos_token_id = decoder_start_token_id

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
