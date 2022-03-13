# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch import set_seed
from unitorch.models.infoxlm import InfoXLMForGeneration, InfoXLMProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.infoxlm.modeling import (
    InfoXLMForGeneration as CLI_InfoXLMForGeneration,
)
from unitorch.cli.models.infoxlm.processing import (
    InfoXLMProcessor as CLI_InfoXLMProcessor,
)
import pkg_resources


class InfoXLMTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)
        config_path = pkg_resources.resource_filename("unitorch", "examples/configs/core/generation/infoxlm.ini")
        self.config = CoreConfigureParser(config_path)
        self.config_path = cached_path(
            "https://huggingface.co/fuliucansheng/unilm/resolve/main/infoxlm-roberta-config.json"
        )
        self.vocab_path = cached_path("https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model")
        self.weight_path = cached_path(
            "https://huggingface.co/fuliucansheng/unilm/resolve/main/default-infoxlm-pytorch-model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init infoxlm from core configure",
            "encode": "test text for infoxlm model",
            "max_gen_seq_length": 5,
            "decode": "info test infoxl",
        }
    )
    def test_config_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = CLI_InfoXLMForGeneration.from_core_configure(self.config)
        process = CLI_InfoXLMProcessor.from_core_configure(self.config)
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
            "testcase_name": "init infoxlm from package",
            "encode": "test text for infoxlm model",
            "max_gen_seq_length": 5,
            "decode": "info test infoxl",
        }
    )
    def test_package_init(
        self,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
        model = InfoXLMForGeneration(self.config_path)
        process = InfoXLMProcessor(
            self.vocab_path,
        )
        model.from_pretrained(self.weight_path)
        model.eval()
        inputs = process.processing_inference(encode).tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()
        outputs = model.generate(
            inputs,
            max_gen_seq_length=max_gen_seq_length,
            decoder_start_token_id=2,
        )
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
