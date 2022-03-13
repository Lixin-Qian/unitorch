# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch import set_seed
from unitorch.models.unilm import UnilmForGeneration, UnilmProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.unilm.modeling import (
    UnilmForGeneration as CLI_UnilmForGeneration,
)
from unitorch.cli.models.unilm.processing import UnilmProcessor as CLI_UnilmProcessor
import pkg_resources


class UnilmTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)
        config_path = pkg_resources.resource_filename("unitorch", "examples/configs/core/generation/unilm.ini")
        self.config = CoreConfigureParser(config_path)
        self.config_path = cached_path(
            "https://huggingface.co/fuliucansheng/unilm/resolve/main/unilm-base-uncased-config.json"
        )
        self.vocab_path = cached_path("https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-vocab.txt")
        self.weight_path = cached_path("https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin")

    @parameterized.named_parameters(
        {
            "testcase_name": "init unilm from core configure",
            "encode": "test text for unilm model",
            "max_gen_seq_length": 5,
            "decode": "....",
        }
    )
    def test_config_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = CLI_UnilmForGeneration.from_core_configure(self.config)
        process = CLI_UnilmProcessor.from_core_configure(self.config)
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
            "testcase_name": "init unilm from package",
            "encode": "test text for unilm model",
            "max_gen_seq_length": 5,
            "decode": "....",
        }
    )
    def test_package_init(self, encode, decode, max_gen_seq_length=10):
        model = UnilmForGeneration(self.config_path)
        process = UnilmProcessor(self.vocab_path)
        model.from_pretrained(self.weight_path)
        model.eval()
        inputs = process.processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
