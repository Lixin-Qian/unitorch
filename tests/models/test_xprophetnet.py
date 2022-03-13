# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch import set_seed
from unitorch.models.xprophetnet import XProphetNetForGeneration, XProphetNetProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.xprophetnet.modeling import (
    XProphetNetForGeneration as CLI_XProphetNetForGeneration,
)
from unitorch.cli.models.xprophetnet.processing import (
    XProphetNetProcessor as CLI_XProphetNetProcessor,
)
import pkg_resources


class XProphetNetTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)
        config_path = pkg_resources.resource_filename("unitorch", "examples/configs/core/generation/xprophetnet.ini")
        self.config = CoreConfigureParser(config_path)
        self.config_path = cached_path(
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json"
        )
        self.vocab_path = cached_path(
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/prophetnet.tokenizer"
        )
        self.weight_path = cached_path(
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/pytorch_model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init xprophetnet from core configure",
            "encode": "test text for xprophetnet model",
            "max_gen_seq_length": 5,
            "decode": "",
        }
    )
    def test_config_init(self, encode, decode, max_gen_seq_length=10):
        model = CLI_XProphetNetForGeneration.from_core_configure(self.config)
        process = CLI_XProphetNetProcessor.from_core_configure(self.config)
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
            "testcase_name": "init xprophetnet from package",
            "encode": "test text for xprophetnet model",
            "max_gen_seq_length": 5,
            "decode": "",
        }
    )
    def test_package_init(self, encode, decode, max_gen_seq_length=10):
        model = XProphetNetForGeneration(self.config_path)
        process = XProphetNetProcessor(self.vocab_path)
        model.from_pretrained(self.weight_path)
        model.eval()
        inputs = process.processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
