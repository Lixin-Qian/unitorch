# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch import set_seed
from unitorch.models.bart import BartForGeneration, BartProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.bart.modeling import BartForGeneration as CLI_BartForGeneration
from unitorch.cli.models.bart.processing import BartProcessor as CLI_BartProcessor
import pkg_resources


class BartTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)
        config_path = pkg_resources.resource_filename("unitorch", "examples/configs/core/generation/bart.ini")
        self.config = CoreConfigureParser(config_path)
        self.config_path = cached_path("https://huggingface.co/facebook/bart-base/resolve/main/config.json")
        self.vocab_path = cached_path("https://huggingface.co/facebook/bart-base/resolve/main/vocab.json")
        self.merge_path = cached_path("https://huggingface.co/facebook/bart-base/resolve/main/merges.txt")
        self.weight_path = cached_path("https://huggingface.co/facebook/bart-base/resolve/main/pytorch_model.bin")

    @parameterized.named_parameters(
        {
            "testcase_name": "init bart from core configure",
            "encode": "test text for bart model",
            "max_gen_seq_length": 5,
            "decode": "test text",
        }
    )
    def test_config_init(self, encode, decode, max_gen_seq_length=10):
        model = CLI_BartForGeneration.from_core_configure(self.config)
        process = CLI_BartProcessor.from_core_configure(self.config)
        model.from_pretrained(self.weight_path)
        model.eval()
        inputs = process._processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)
        results = process._processing_decode(outputs).sequences
        assert results[0] == decode

    @parameterized.named_parameters(
        {
            "testcase_name": "init bart from package",
            "encode": "test text for bart model",
            "max_gen_seq_length": 5,
            "decode": "test text",
        }
    )
    def test_package_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = BartForGeneration(self.config_path)
        process = BartProcessor(self.vocab_path, self.merge_path)
        model.from_pretrained(self.weight_path)
        model.eval()
        inputs = process.processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
