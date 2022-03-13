# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch import set_seed
from unitorch.models.mass import MASSForGeneration, MASSProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.mass.modeling import MASSForGeneration as CLI_MASSForGeneration
from unitorch.cli.models.mass.processing import MASSProcessor as CLI_MASSProcessor
import pkg_resources


class MASSTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)
        config_path = pkg_resources.resource_filename("unitorch", "examples/configs/core/generation/mass.ini")
        self.config = CoreConfigureParser(config_path)
        self.config_path = cached_path(
            "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-config.json"
        )
        self.vocab_path = cached_path(
            "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-vocab.txt"
        )
        self.weight_path = cached_path(
            "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-pytorch-model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init mass from core configure",
            "encode": "test text for mass model",
            "max_gen_seq_length": 5,
            "decode": "of core model model",
        }
    )
    def test_config_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = CLI_MASSForGeneration.from_core_configure(self.config)
        process = CLI_MASSProcessor.from_core_configure(self.config)
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
            "testcase_name": "init mass from package",
            "encode": "test text for mass model",
            "max_gen_seq_length": 5,
            "decode": "of core model model",
        }
    )
    def test_package_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = MASSForGeneration(self.config_path, self.vocab_path)
        process = MASSProcessor(self.vocab_path)
        model.from_pretrained(self.weight_path)
        model.eval()

        inputs = process.processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
