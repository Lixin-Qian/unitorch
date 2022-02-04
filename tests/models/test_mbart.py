# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch import set_seed, hf_cached_path
from unitorch.models.mbart import MBartForGeneration, MBartProcessor
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.mbart.modeling import (
    MBartForGeneration as CLI_MBartForGeneration,
)
from unitorch.cli.models.mbart.processing import MBartProcessor as CLI_MBartProcessor
import pkg_resources


class MBartTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)
        config_path = pkg_resources.resource_filename(
            "unitorch", "examples/configs/core/generation/mbart.ini"
        )
        self.config = CoreConfigureParser(config_path)
        self.config_path = hf_cached_path(
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json"
        )
        self.vocab_path = hf_cached_path(
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentence.bpe.model"
        )
        self.weight_path = hf_cached_path(
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/pytorch_model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init mbart from core configure",
            "encode": "test text for mbart model",
            "max_gen_seq_length": 5,
            "decode": "",
        }
    )
    def test_config_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = CLI_MBartForGeneration.from_core_configure(self.config)
        process = CLI_MBartProcessor.from_core_configure(self.config)
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
            "testcase_name": "init mbart from package",
            "encode": "test text for mbart model",
            "max_gen_seq_length": 5,
            "decode": "",
        }
    )
    def test_package_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = MBartForGeneration(self.config_path)
        process = MBartProcessor(self.vocab_path)
        model.from_pretrained(self.weight_path)
        model.eval()

        inputs = process.processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
