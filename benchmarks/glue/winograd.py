# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.cuda.amp import autocast
from unitorch import hf_cached_path
from transformers import RobertaTokenizer, DebertaTokenizer
from unitorch.models import GenericModel, GenericOutputs, _truncate_seq_pair
from unitorch.models.roberta import RobertaForMaskLM
from unitorch.models.deberta import DebertaForMaskLM
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
    register_dataset,
)
from unitorch.cli.models import (
    ClassificationOutputs,
    LossOutputs,
    BaseInputs,
    BaseTargets,
    ClassificationTargets,
)


@register_model("benchmarks/model/classification/winograd")
class WinogradModel(GenericModel):
    def __init__(
        self,
        model_name,
        config_path,
        mask_token_id,
        margin_alpha=5.0,
        margin_beta=0.4,
        use_margin_loss=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if model_name == "roberta":
            self.model = RobertaForMaskLM(
                config_path,
                gradient_checkpointing=gradient_checkpointing,
            )

        if model_name == "deberta":
            self.model = DebertaForMaskLM(
                config_path,
                gradient_checkpointing=gradient_checkpointing,
            )

        self.mask_token_id = mask_token_id
        self.margin_alpha = margin_alpha
        self.margin_beta = margin_beta
        self.use_margin_loss = use_margin_loss

    @classmethod
    @add_default_section_for_init("benchmarks/model/classification/winograd")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("benchmarks/model/classification/winograd")
        config_path = config.getoption("config_path", None)
        config_path = hf_cached_path(config_path)

        model_name = config.getoption("model_name", None)
        mask_token_id = config.getoption("mask_token_id", 50264)
        margin_alpha = config.getoption("margin_alpha", 5.0)
        margin_beta = config.getoption("margin_beta", 0.4)
        use_margin_loss = config.getoption("use_margin_loss", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        inst = cls(
            model_name=model_name,
            config_path=config_path,
            mask_token_id=mask_token_id,
            margin_alpha=margin_alpha,
            margin_beta=margin_beta,
            use_margin_loss=use_margin_loss,
            gradient_checkpointing=gradient_checkpointing,
        )

        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = hf_cached_path(pretrained_weight_path)
        inst.model.from_pretrained(pretrained_weight_path)

        return inst

    def get_scores(self, tokens, attn_mask, seg_ids, pos_ids, mask):
        new_tokens = tokens.clone()
        new_tokens[mask.bool()] = self.mask_token_id
        logits = self.model(new_tokens, attn_mask, seg_ids, pos_ids)
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)
        scores = lprobs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        mask = mask.type_as(scores)
        scores = (scores * mask).sum(dim=-1) / mask.sum(dim=-1)
        return scores

    @autocast()
    def forward(
        self,
        query_tokens_ids,
        query_attn_mask,
        query_seg_ids,
        query_pos_ids,
        query_mlm_mask,
        cands_tokens_ids,
        cands_attn_mask,
        cands_seg_ids,
        cands_pos_ids,
        cands_mlm_mask,
    ):
        query_scores = self.get_scores(
            query_tokens_ids,
            query_attn_mask,
            query_seg_ids,
            query_pos_ids,
            query_mlm_mask,
        )
        cands_tokens_ids = cands_tokens_ids.reshape(-1, *cands_tokens_ids.size()[2:])
        cands_attn_mask = cands_attn_mask.reshape(-1, *cands_attn_mask.size()[2:])
        cands_seg_ids = cands_seg_ids.reshape(-1, *cands_seg_ids.size()[2:])
        cands_pos_ids = cands_pos_ids.reshape(-1, *cands_pos_ids.size()[2:])
        cands_mlm_mask = cands_mlm_mask.reshape(-1, *cands_mlm_mask.size()[2:])
        cands_scores = self.get_scores(
            cands_tokens_ids,
            cands_attn_mask,
            cands_seg_ids,
            cands_pos_ids,
            cands_mlm_mask,
        )
        cands_scores = cands_scores.reshape(
            query_scores.size(0), -1, *cands_scores.size()[1:]
        )
        if self.training:
            if self.use_margin_loss:
                loss = (
                    (
                        -query_scores.unsqueeze(dim=1)
                        + self.margin_alpha
                        * (
                            cands_scores
                            - query_scores.unsqueeze(dim=1)
                            + self.margin_beta
                        ).clamp(min=0)
                    )
                    .sum(dim=-1)
                    .mean()
                )
            else:
                query_scores = query_scores.unsqueeze(dim=1)
                loss = F.cross_entropy(
                    torch.cat([query_scores, cands_scores], dim=1),
                    torch.zeros(query_scores.size(0)).to(query_scores).long(),
                )
            return LossOutputs(loss=loss)

        outputs = (query_scores.unsqueeze(dim=1) >= cands_scores).all(dim=-1).int()
        return ClassificationOutputs(outputs=outputs)

    def from_checkpoint(
        self,
        ckpt_dir="./cache",
        weight_name="pytorch_model.bin",
        **kwargs,
    ):
        self.model.from_checkpoint(ckpt_dir, weight_name, **kwargs)

    def save_checkpoint(
        self,
        ckpt_dir="./cache",
        weight_name="pytorch_model.bin",
        **kwargs,
    ):
        self.model.save_checkpoint(ckpt_dir, weight_name, **kwargs)

    def from_pretrained(
        self,
        weight_path=None,
        replace_keys=dict(),
        **kwargs,
    ):
        self.model.from_pretrained(weight_path, replace_keys, **kwargs)


# processor
def find_token(sentence, start_pos):
    found_tok = None
    for tok in sentence:
        if tok.idx == start_pos:
            found_tok = tok
            break
    return found_tok


def find_span(sentence, search_text, start=0):
    search_text = search_text.lower()
    for tok in sentence[start:]:
        remainder = sentence[tok.i :].text.lower()
        if remainder.startswith(search_text):
            len_to_consume = len(search_text)
            start_idx = tok.idx
            for next_tok in sentence[tok.i :]:
                end_idx = next_tok.idx + len(next_tok.text)
                if end_idx - start_idx == len_to_consume:
                    span = sentence[tok.i : next_tok.i + 1]
                    return span
    return None


def extended_noun_chunks(sentence):
    noun_chunks = {(np.start, np.end) for np in sentence.noun_chunks}
    np_start, cur_np = 0, "NONE"
    for i, token in enumerate(sentence):
        np_type = token.pos_ if token.pos_ in {"NOUN", "PROPN"} else "NONE"
        if np_type != cur_np:
            if cur_np != "NONE":
                noun_chunks.add((np_start, i))
            if np_type != "NONE":
                np_start = i
            cur_np = np_type
    if cur_np != "NONE":
        noun_chunks.add((np_start, len(sentence)))
    return [sentence[s:e] for (s, e) in sorted(noun_chunks)]


def filter_noun_chunks(
    chunks,
    exclude_pronouns=False,
    exclude_query=None,
    exact_match=False,
):
    if exclude_pronouns:
        chunks = [
            np
            for np in chunks
            if (np.lemma_ != "-PRON-" and not all(tok.pos_ == "PRON" for tok in np))
        ]

    if exclude_query is not None:
        excl_txt = [exclude_query.lower()]
        filtered_chunks = []
        for chunk in chunks:
            lower_chunk = chunk.text.lower()
            found = False
            for excl in excl_txt:
                if (
                    not exact_match and (lower_chunk in excl or excl in lower_chunk)
                ) or lower_chunk == excl:
                    found = True
                    break
            if not found:
                filtered_chunks.append(chunk)
        chunks = filtered_chunks

    return chunks


class WinogradProcessor(object):
    def __init__(
        self,
        processor_name: str,
        vocab_path: str,
        merge_path: str,
        max_seq_length: int = 128,
        source_type_id: int = 0,
        target_type_id: int = 0,
    ):
        import spacy
        from sacremoses import MosesDetokenizer

        self.nlp = spacy.load("en_core_web_lg")
        self.detok = MosesDetokenizer(lang="en")

        if processor_name == "roberta":
            self.tokenizer = RobertaTokenizer(
                vocab_path,
                merge_path,
            )
        if processor_name == "deberta":
            self.tokenizer = DebertaTokenizer(
                vocab_path,
                merge_path,
            )
        self.max_seq_length = max_seq_length
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.position_start_id = self.pad_token_id + 1

    def _preprocess_wnli(self, sentence1, sentence2):
        pronouns = [
            "i",
            "me",
            "we",
            "us",
            "you",
            "he",
            "him",
            "she",
            "her",
            "they",
            "them",
            "it",
            "one",
            "ones",
            "his",
            "their",
        ]
        pronouns += [f"{ch}." for ch in pronouns] + [f"{ch}," for ch in pronouns]

        def replace_raw_text(s):
            rep = {
                "She saw that ": "",
                "her mother ": "mother ",
                "her mother's": "mother",
                "be ride of": "be rid of",
                "his father's": "father",
                "Her mother's": "mother",
                "He cursed it under his breath for standing in his way,": "",
                "Bernard, who had not told the government official that he was less than 21 when he filed for a homestead claim, did not consider that": "",
                "Lionel plans to use it on Geoffrey and ": "",
                "new house ": "house ",
                "now.": "now,",
                "Prehistoric humans": "humans",
                "Alice's daughter": "daughter",
                "old house": "house",
                "The teller": "The tellers",
                "the sound was": "the sound",
                "the piece": "the pieces",
            }
            for k, v in rep.items():
                s = s.replace(k, v)
            return s

        sentence2 = replace_raw_text(sentence2)
        words1 = sentence1.lower().split(" ")
        words2 = sentence2.lower().split(" ")

        def replace(w):
            rep = {
                "regulate": "regular",
                "were": "was",
                "behind": "through",
            }
            return [rep.get(t, t) for t in w]

        words1 = replace(words1)
        words2 = replace(words2)
        ret = dict()
        for i, t1 in enumerate(words1):
            if t1 not in pronouns:
                continue
            # prefix
            if i > 0:
                a = i - 1
                if (t1 == "them" or t1 == "them.") and words1[i - 1] == "of":
                    a -= 1
                if (t1 == "it" or t1 == "it.") and words1[i - 1] == "over":
                    a -= 1
                k = len(words2) - 1
                while k >= 0 and words1[a] != words2[k]:
                    k -= 1
                while k >= 0 and a >= 0 and words1[a] == words2[k]:
                    k -= 1
                    a -= 1
            else:
                a = -1
                k = -1
            k1 = k
            # suffix
            if i + 1 < len(words1):
                b = i + 1
                k = k + 1
                while k < len(words2) and words1[b] != words2[k]:
                    k += 1
                while k < len(words2) and b < len(words1) and words1[b] == words2[k]:
                    k += 1
                    b += 1
            else:
                b = i + 1
                k = len(words2)
            k2 = k
            if b - a > 2:
                ret[i] = (i, a, b, k1, k2)
        value = ""
        key = -1
        for k, v in ret.items():
            i, a, b, k1, k2 = v
            new_value = " ".join(words1[a + 1 : b])
            if len(new_value) > len(value):
                value = new_value
                key = k
        i, a, b, k1, k2 = ret[key]
        span_text = " ".join(words1[a + 1 : b])
        aa, bb, kk1, kk2 = a + 1, b - 1, k1 + 1, k2 - 1
        while words1[aa] == words2[kk1] and aa < bb and kk1 < kk2:
            aa += 1
            kk1 += 1
        while words1[bb] == words2[kk2]:
            bb -= 1
            kk2 -= 1
        span2_text = " ".join(words1[aa : bb + 1])
        span2_index = aa
        span1_text = " ".join(words2[kk1 : kk2 + 1])
        rules = ["copies of ", "the ", "of "]
        for rule in rules:
            if span1_text.startswith(rule):
                span1_text = span1_text[len(rule) :]
            if span2_text.startswith(rule):
                span2_text = span2_text[len(rule) :]
        if sentence1.lower().find(span1_text.lower()) == -1:
            if span1_text.endswith("."):
                span1_text = span1_text[:-1]
            if span1_text.endswith("'s"):
                span1_text = span1_text[:-2]
        if sentence1.lower().find(span1_text.lower()) != -1:
            span1_index = sentence1[: sentence1.lower().find(span1_text.lower())].count(
                " "
            )
        else:
            span1_index = -1
        find_index = len(" ".join(sentence1.split(" ")[:span2_index]))
        span2_index = sentence1[
            : sentence1.lower().find(span2_text.lower(), find_index)
        ].count(" ")
        return self._preprocess_wsc(
            sentence1, span1_index, span1_text, span2_index, span2_text
        )

    def _preprocess_wsc(
        self,
        text: str,
        span1_index: int,
        span1_text: str,
        span2_index: int,
        span2_text: str,
    ):
        query = span1_text
        if query is not None:
            if query.endswith(".") or query.endswith(","):
                query = query[:-1]

        def strip_pronoun(x):
            return x.rstrip('.,"')

        tokens = text.split(" ")
        pronoun_idx = span2_index
        pronoun = strip_pronoun(span2_text)
        if strip_pronoun(tokens[pronoun_idx]) != pronoun:
            # hack: sometimes the index is misaligned
            if strip_pronoun(tokens[pronoun_idx + 1]) == pronoun:
                pronoun_idx += 1
        before = tokens[:pronoun_idx]
        after = tokens[pronoun_idx + 1 :]
        leading_space = " " if pronoun_idx > 0 else ""
        trailing_space = " " if len(after) > 0 else ""

        before = self.detok.detokenize(before, return_str=True)
        pronoun = self.detok.detokenize([pronoun], return_str=True)
        after = self.detok.detokenize(after, return_str=True)

        if pronoun.endswith(".") or pronoun.endswith(","):
            after = pronoun[-1] + trailing_space + after
            pronoun = pronoun[:-1]

        if after.startswith(".") or after.startswith(","):
            trailing_space = ""

        sentence = self.nlp(before + leading_space + pronoun + trailing_space + after)
        start = len(before + leading_space)
        first_pronoun_tok = find_token(sentence, start_pos=start)
        pronoun_span = find_span(sentence, pronoun, start=first_pronoun_tok.i)
        assert pronoun_span.text == pronoun

        cand_spans = filter_noun_chunks(
            extended_noun_chunks(sentence),
            exclude_pronouns=True,
            exclude_query=query,
            exact_match=False,
        )

        query = f"{leading_space}{query}"
        cands = [f"{leading_space}{span}" for span in cand_spans]
        return GenericOutputs(
            prefix=before,
            suffix=f"{trailing_space}{after}",
            query=query,
            cands=cands,
        )

    def _tokenize(self, prefix, text, suffix):
        prefix_tokens = self.tokenizer.tokenize(prefix)
        text_tokens = self.tokenizer.tokenize(text)
        suffix_tokens = self.tokenizer.tokenize(suffix)
        _truncate_seq_pair(
            prefix_tokens, suffix_tokens, self.max_seq_length - len(text_tokens) - 2
        )
        prefix_tokens = [self.bos_token] + prefix_tokens
        suffix_tokens = suffix_tokens + [self.eos_token]
        tokens = prefix_tokens + text_tokens + suffix_tokens
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_mask = [1] * len(tokens_ids)
        segment_ids = [self.source_type_id] * len(tokens_ids)
        padding = [0] * (self.max_seq_length - len(tokens_ids))
        tokens_ids += len(padding) * [self.pad_token_id]
        tokens_mask += padding
        segment_ids += len(padding) * [self.target_type_id]
        mlm_mask = (
            [0] * len(prefix_tokens)
            + [1] * len(text_tokens)
            + [0] * len(suffix_tokens)
            + padding
        )
        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            seg_ids=torch.tensor(segment_ids, dtype=torch.long),
            attn_mask=torch.tensor(tokens_mask, dtype=torch.long),
            pos_ids=torch.tensor(
                list(
                    range(
                        self.position_start_id,
                        self.position_start_id + self.max_seq_length,
                    )
                ),
                dtype=torch.long,
            ),
            mlm_mask=torch.tensor(mlm_mask),
        )


class WnliDataset(object):
    def __init__(self, processor, split, max_num_cands=10):
        if split == "train":
            self.dataset = load_dataset("super_glue", "wsc", split=split)
            self.dataset = self.dataset.filter(lambda v: v["label"] == 1)
        else:
            self.dataset = load_dataset("glue", "wnli", split=split)

        self.split = split
        self.processor = processor
        self.max_num_cands = max_num_cands

    def __getitem__(self, idx):
        row = self.dataset[idx]

        if self.split == "train":
            outputs = self.processor._preprocess_wsc(
                row["text"],
                row["span1_index"],
                row["span1_text"],
                row["span2_index"],
                row["span2_text"],
            )
        else:
            outputs = self.processor._preprocess_wnli(
                row["sentence1"], row["sentence2"]
            )

        prefix = outputs.prefix
        suffix = outputs.suffix
        query = outputs.query
        cands = outputs.cands

        while len(cands) < self.max_num_cands:
            cands += cands
        query_outputs = self.processor._tokenize(prefix, query, suffix)
        cands_outputs = [
            self.processor._tokenize(prefix, cand, suffix)
            for cand in cands[: self.max_num_cands]
        ]
        (
            query_tokens_ids,
            query_attn_mask,
            query_seg_ids,
            query_pos_ids,
            query_mlm_mask,
        ) = (
            query_outputs.tokens_ids,
            query_outputs.attn_mask,
            query_outputs.seg_ids,
            query_outputs.pos_ids,
            query_outputs.mlm_mask,
        )

        cands_tokens_ids = torch.stack([output.tokens_ids for output in cands_outputs])
        cands_attn_mask = torch.stack([output.attn_mask for output in cands_outputs])
        cands_seg_ids = torch.stack([output.seg_ids for output in cands_outputs])
        cands_pos_ids = torch.stack([output.pos_ids for output in cands_outputs])
        cands_mlm_mask = torch.stack([output.mlm_mask for output in cands_outputs])

        inputs = BaseInputs(
            query_tokens_ids=query_tokens_ids,
            query_attn_mask=query_attn_mask,
            query_seg_ids=query_seg_ids,
            query_pos_ids=query_pos_ids,
            query_mlm_mask=query_mlm_mask,
            cands_tokens_ids=cands_tokens_ids,
            cands_attn_mask=cands_attn_mask,
            cands_seg_ids=cands_seg_ids,
            cands_pos_ids=cands_pos_ids,
            cands_mlm_mask=cands_mlm_mask,
        )

        if self.split == "validation":
            targets = ClassificationTargets(targets=torch.tensor(row["label"]))
        else:
            targets = BaseTargets()

        return inputs, targets

    def __len__(
        self,
    ):
        return len(self.dataset)


class WscDataset(object):
    def __init__(self, processor, split, max_num_cands=10):
        if split == "train":
            self.dataset = load_dataset("super_glue", "wsc", split=split)
            self.dataset = self.dataset.filter(lambda v: v["label"] == 1)
        else:
            self.dataset = load_dataset("super_glue", "wsc", split=split)
        self.split = split
        self.processor = processor
        self.max_num_cands = max_num_cands

    def __getitem__(self, idx):
        row = self.dataset[idx]

        outputs = self.processor._preprocess_wsc(
            row["text"],
            row["span1_index"],
            row["span1_text"],
            row["span2_index"],
            row["span2_text"],
        )

        prefix = outputs.prefix
        suffix = outputs.suffix
        query = outputs.query
        cands = outputs.cands

        while len(cands) < self.max_num_cands:
            cands += cands
        query_outputs = self.processor._tokenize(prefix, query, suffix)
        cands_outputs = [
            self.processor._tokenize(prefix, cand, suffix)
            for cand in cands[: self.max_num_cands]
        ]
        (
            query_tokens_ids,
            query_attn_mask,
            query_seg_ids,
            query_pos_ids,
            query_mlm_mask,
        ) = (
            query_outputs.tokens_ids,
            query_outputs.attn_mask,
            query_outputs.seg_ids,
            query_outputs.pos_ids,
            query_outputs.mlm_mask,
        )

        cands_tokens_ids = torch.stack([output.tokens_ids for output in cands_outputs])
        cands_attn_mask = torch.stack([output.attn_mask for output in cands_outputs])
        cands_seg_ids = torch.stack([output.seg_ids for output in cands_outputs])
        cands_pos_ids = torch.stack([output.pos_ids for output in cands_outputs])
        cands_mlm_mask = torch.stack([output.mlm_mask for output in cands_outputs])

        inputs = BaseInputs(
            query_tokens_ids=query_tokens_ids,
            query_attn_mask=query_attn_mask,
            query_seg_ids=query_seg_ids,
            query_pos_ids=query_pos_ids,
            query_mlm_mask=query_mlm_mask,
            cands_tokens_ids=cands_tokens_ids,
            cands_attn_mask=cands_attn_mask,
            cands_seg_ids=cands_seg_ids,
            cands_pos_ids=cands_pos_ids,
            cands_mlm_mask=cands_mlm_mask,
        )

        if self.split == "validation":
            targets = ClassificationTargets(targets=torch.tensor(row["label"]))
        else:
            targets = BaseTargets()

        return inputs, targets

    def __len__(
        self,
    ):
        return len(self.dataset)


@register_dataset("benchmarks/dataset/glue/wnli")
class WinogradDatasets(object):
    def __init__(
        self,
        processor_name,
        vocab_path,
        merge_path,
        max_seq_length=128,
        max_num_cands=10,
        test_split="test",
    ):
        self.__datasets__ = dict()
        for split in ["train", "dev", "test"]:
            processor = WinogradProcessor(
                processor_name,
                vocab_path,
                merge_path,
                max_seq_length,
            )
            new_split = "validation" if split == "dev" else split
            new_split = test_split if split == "test" else new_split
            self.__datasets__[split] = WnliDataset(processor, new_split, max_num_cands)

    @classmethod
    @add_default_section_for_init("benchmarks/dataset/glue/wnli")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("benchmarks/dataset/glue/wnli")
        vocab_path = config.getoption("vocab_path", None)
        merge_path = config.getoption("merge_path", None)
        vocab_path = hf_cached_path(vocab_path)
        merge_path = hf_cached_path(merge_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
        }

    def get(self, split: str = "train"):
        return self.__datasets__.get(split)


@register_dataset("benchmarks/dataset/superglue/wsc")
class WinogradDatasets(object):
    def __init__(
        self,
        processor_name,
        vocab_path,
        merge_path,
        max_seq_length=128,
        max_num_cands=10,
        test_split="test",
    ):
        self.__datasets__ = dict()
        for split in ["train", "dev", "test"]:
            processor = WinogradProcessor(
                processor_name,
                vocab_path,
                merge_path,
                max_seq_length,
            )
            new_split = "validation" if split == "dev" else split
            new_split = test_split if split == "test" else new_split
            self.__datasets__[split] = WscDataset(processor, new_split, max_num_cands)

    @classmethod
    @add_default_section_for_init("benchmarks/dataset/superglue/wsc")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("benchmarks/dataset/superglue/wsc")
        vocab_path = config.getoption("vocab_path", None)
        merge_path = config.getoption("merge_path", None)
        vocab_path = hf_cached_path(vocab_path)
        merge_path = hf_cached_path(merge_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
        }

    def get(self, split: str = "train"):
        return self.__datasets__.get(split)
