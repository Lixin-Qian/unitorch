# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import device, Tensor
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    XLMProphetNetConfig,
    XLMProphetNetModel,
)
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from unitorch.models import GenericModel, GenericOutputs


class XProphetNetForGeneration(GenericModel, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(
        self,
        config_path,
        freeze_word_embedding: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()

        self.config = XLMProphetNetConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.prophetnet = XLMProphetNetModel(self.config)
        self.padding_idx = self.config.pad_token_id

        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        if freeze_word_embedding:
            self.lm_head.weight.requires_grad = False
            self.prophetnet.word_embeddings.weight.requires_grad = False

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    def get_encoder(self):
        return self.prophetnet.encoder

    def get_decoder(self):
        return self.prophetnet.decoder

    @property
    def device(self) -> device:
        return next(self.parameters()).device

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        decoder_length = decoder_input_ids.size(1)
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        if decoder_length == 1:
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.num_beams, *encoder_hidden_states.size()[1:]
            )
            encoder_hidden_states = encoder_hidden_states[:, 0]
            encoder_outputs.last_hidden_state = encoder_hidden_states

        return {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

    @staticmethod
    def _reorder_cache_v2(past, batch_idx, beam_idx):
        if batch_idx is None:
            return _reorder_cache(past, beam_idx)
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            past_state1 = tuple(
                [past_state.index_select(0, beam_idx) for past_state in layer_past[:2]]
            )
            past_state2 = tuple(
                [past_state.index_select(0, batch_idx) for past_state in layer_past[2:]]
            )
            reordered_past += (past_state1 + past_state2,)

        return reordered_past

    def forward(
        self,
        tokens_ids_a=None,
        tokens_mask_a=None,
        tokens_ids_b=None,
        tokens_mask_b=None,
        decoder_input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        decoder_length=None,
        return_dict=None,
    ):
        if self.training:
            outputs = self.prophetnet(
                input_ids=tokens_ids_a,
                attention_mask=tokens_mask_a,
                decoder_input_ids=tokens_ids_b,
                decoder_attention_mask=tokens_mask_b,
            )
            batch_size, sequence_length = tokens_ids_b.shape[:2]
            predicting_streams = outputs[1].view(
                batch_size, self.config.ngram, sequence_length, -1
            )
            predict_logits = self.lm_head(predicting_streams)
            return predict_logits

        batch_size, sequence_length = decoder_input_ids.shape[:2]
        outputs = self.prophetnet(
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
        )
        predicting_streams = outputs[1].view(
            batch_size, self.config.ngram, sequence_length, -1
        )
        predict_logits = self.lm_head(predicting_streams)
        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None
        return Seq2SeqLMOutput(logits=logits, past_key_values=outputs.past_key_values)

    def generate(
        self,
        tokens_ids,
        num_beams=5,
        decoder_start_token_id=2,
        decoder_end_token_id=2,
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
        self.num_beams = num_beams
        outputs = super().generate(
            tokens_ids,
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

        sequences = outputs.sequences.reshape(
            -1, num_return_sequences, outputs.sequences.size(-1)
        )
        outputs.sequences = torch.zeros(
            sequences.size(0), num_return_sequences, max_gen_seq_length
        ).to(device=sequences.device)
        outputs.sequences[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences, sequences_scores=outputs.sequences_scores
        )
