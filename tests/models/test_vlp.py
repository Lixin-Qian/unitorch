# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from PIL import Image
from absl.testing import absltest, parameterized
<<<<<<< HEAD
from unitorch import set_seed, hf_cached_path
from unitorch.models.vlp import VLPForGeneration, VLPProcessor
=======
from unitorch import set_seed
from unitorch.models.vlp import VLPForGeneration, VLPProcessor
from unitorch.cli import cached_path
>>>>>>> f16553e (add test vlp)
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.vlp.modeling import VLPForGeneration as CLI_VLPForGeneration
from unitorch.cli.models.vlp.processing import VLPProcessor as CLI_VLPProcessor
import pkg_resources


class VLPTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)

<<<<<<< HEAD
        config_path = pkg_resources.resource_filename(
            "unitorch", "examples/configs/core/generation/vlp.ini"
        )
        self.config = CoreConfigureParser(config_path)

        self.detection_config_path = hf_cached_path(
            "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/FasterRCNN_X_101_64x4d_FPN_2x_config.yaml"
        )
        self.detection_weight_path = hf_cached_path(
            "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/FasterRCNN_X_101_64x4d_FPN_2x_pytorch_model.bin"
        )
        self.vlp_config_path = hf_cached_path(
            "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/config.json"
        )
        self.vlp_weight_path = hf_cached_path(
            "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/pytorch_model.bin"
        )
        self.vocab_path = hf_cached_path(
            "https://huggingface.co/microsoft/unilm-base-cased/resolve/main/vocab.txt"
        )
=======
        config_path = pkg_resources.resource_filename("unitorch", "examples/configs/core/generation/vlp.ini")
        self.config = CoreConfigureParser(config_path)

        self.detection_config_path = cached_path(
            "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/faster_rcnn_x101_64x4d_fpn_2x_config.yaml"
        )
        self.detection_weight_path = cached_path(
            "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/faster_rcnn_x101_64x4d_fpn_2x_model.bin"
        )
        self.vlp_config_path = cached_path(
            "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/config.json"
        )
        self.vlp_weight_path = cached_path(
            "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/pytorch_model.bin"
        )
        self.vocab_path = cached_path("https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/vocab.txt")
>>>>>>> f16553e (add test vlp)

    @parameterized.named_parameters(
        {
            "testcase_name": "init vlp from core configure",
            "image": "https://huggingface.co/fuliucansheng/detection/resolve/main/input.jpg",
            "encode": "test text for vlp model",
            "decode": "",
            "max_gen_seq_length": 10,
        }
    )
    def test_config_init(
        self,
        image,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
<<<<<<< HEAD
        image = hf_cached_path(image)
=======
        image = cached_path(image)
>>>>>>> f16553e (add test vlp)
        model = CLI_VLPForGeneration.from_core_configure(self.config)
        model.eval()
        process = CLI_VLPProcessor.from_core_configure(self.config)
        results = process._processing_inference(image, encode)
        images = results.pixel_values.unsqueeze(0)
        inputs = results.tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, images, inputs = model.cuda(), images.cuda(), inputs.cuda()
        outputs = model.generate(images, inputs, max_gen_seq_length=max_gen_seq_length)
        results = process._processing_decode(outputs).sequences
        assert results[0] == decode

    @parameterized.named_parameters(
        {
            "testcase_name": "init vlp from package",
            "image": "https://huggingface.co/fuliucansheng/detection/resolve/main/input.jpg",
            "encode": "test text for vlp model",
            "decode": "",
            "max_gen_seq_length": 10,
        }
    )
    def test_package_init(
        self,
        image,
        encode,
        decode,
        max_gen_seq_length=10,
    ):
<<<<<<< HEAD
        image = hf_cached_path(image)
=======
        image = cached_path(image)
>>>>>>> f16553e (add test vlp)
        model = VLPForGeneration(self.vlp_config_path, self.detection_config_path)
        model.vision_model.from_pretrained(self.detection_weight_path)
        model.from_pretrained(self.vlp_weight_path)
        model.eval()
        process = VLPProcessor(
            self.vocab_path,
            pixel_mean=[103.53, 116.28, 123.675],
            pixel_std=[1.0, 1.0, 1.0],
        )
        image = Image.open(image)
        results = process.processing_inference(image, encode)
        images = results.image.unsqueeze(0)
        inputs = results.tokens_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, images, inputs = model.cuda(), images.cuda(), inputs.cuda()
        outputs = model.generate(images, inputs, max_gen_seq_length=max_gen_seq_length)
        results = process.processing_decode(outputs.sequences)
        assert results[0] == decode
