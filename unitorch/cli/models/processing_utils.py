# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random
from PIL import Image, ImageFile
from unitorch.models.processing_utils import (
    GeneralizedProcessor as _GeneralizedProcessor,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    BaseInputs,
    BaseOutputs,
    BaseTargets,
    LossOutputs,
    EmbeddingOutputs,
    ClassificationOutputs,
    ClassificationTargets,
    GenerationOutputs,
    GenerationTargets,
    DetectionOutputs,
    DetectionTargets,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

ACT2FN = {
    "relu": nn.functional.relu,
    "tanh": torch.tanh,
    "sigmoid": lambda v: torch.sigmoid(v.float()),
    "softmax": lambda v: torch.softmax(v.float(), dim=-1),
}


def _process_returns(attrs, dtype="BaseInputs"):
    assert dtype in globals()
    cls = globals()[dtype]
    assert issubclass(cls, BaseInputs) or issubclass(cls, BaseOutputs) or issubclass(cls, BaseTargets)
    return cls(attrs)


class GeneralizedProcessor(_GeneralizedProcessor):
    def __init__(
        self,
        num_class: int = None,
        sep: str = ",",
        max_seq_length: int = 128,
        map_dict: Dict = dict(),
        act_fn: str = None,
        return_scores: bool = True,
    ):
        super().__init__(
            num_class=num_class,
            sep=sep,
            max_seq_length=max_seq_length,
            map_dict=map_dict,
        )
        self.act_fn = ACT2FN.get(act_fn, None)
        self.return_scores = return_scores

    @classmethod
    @add_default_section_for_init("core/process/general")
    def from_core_configure(cls, config, **kwargs):
        pass

    # preprocess functions
    @register_process("core/process/digit")
    def _processing_digit(
        self,
        digit: Union[int, float, str],
        dtype: str = "int",
        key: str = "num",
        returns: str = "BaseInputs",
    ):
        tensor = super().processing_digit(
            digit=digit,
            dtype=dtype,
        )
        return _process_returns({key: tensor}, dtype=returns)

    @register_process("core/process/label")
    def _processing_target(
        self,
        text: str,
        sample_weight: Union[float, str] = None,
        dtype: str = "int",
    ):
        """
        for classification
        """
        tensor = super().processing_target(
            text=text,
            dtype=dtype,
        )

        outputs = ClassificationTargets(targets=tensor)

        if sample_weight is not None:
            outputs.sample_weight = super().processing_digit(
                sample_weight,
                dtype="float",
            )

        return outputs

    @register_process("core/process/features")
    def _processing_features(
        self,
        features: Union[List, str],
        sep: str = None,
        dtype: str = "float",
        shape: tuple = None,
        key: str = "features",
        returns: str = "BaseInputs",
    ):
        """
        for features
        """
        tensor = super().processing_features(
            features=features,
            sep=sep,
            dtype=dtype,
            shape=shape,
        )
        return _process_returns({key: tensor}, dtype=returns)

    @register_process("core/process/sequence")
    def _processing_sequence(
        self,
        text: Union[List, str],
        sample_weight: Union[float, str] = None,
        sep: str = None,
        dtype: str = "int",
    ):
        """
        process functions for BIO/BIEO label
        """
        tensor = super().processing_sequence(
            text=text,
            dtype=dtype,
        )

        outputs = ClassificationTargets(targets=tensor)
        if sample_weight is not None:
            outputs.sample_weight = super().processing_digit(
                sample_weight,
                dtype="float",
            )

        return outputs

    @register_process("core/process/multi_label")
    def _processing_multi_target(
        self,
        text: Union[List, str],
        sample_weight: Union[float, str] = None,
        sep: str = None,
        dtype: str = "int",
    ):
        """
        process functions for bce label
        """
        if self.num_class == 1:
            tensor = super().processing_target(text, dtype=dtype)
            tensor = tensor.unsqueeze(-1)
        else:
            tensor = super().processing_multi_target(
                text=text,
                sep=sep,
            )

        outputs = ClassificationTargets(targets=tensor)
        if sample_weight is not None:
            outputs.sample_weight = super().processing_digit(
                sample_weight,
                dtype="float",
            )

        return outputs

    # post process functions
    @register_process("core/postprocess/binary_score")
    def _processing_binary_score(
        self,
        outputs: Union[ClassificationOutputs, BaseOutputs],
    ):
        if not hasattr(outputs, "outputs"):
            raise ValueError(f"core/postprocess/binary_score can't process the outputs")

        if self.act_fn is not None:
            outputs.outputs = self.act_fn(outputs.outputs)

        if outputs.outputs.dim() == 2:
            pscore = outputs.outputs[:, 1] if outputs.outputs.size(-1) > 1 else outputs.outputs[:, 0]
        else:
            pscore = outputs.outputs
        pscore = list(map(float, pscore))
        _infos = outputs.to_dict()
        _infos.pop("outputs")
        return BaseOutputs(**_infos, **{"pscore": pscore})

    @register_process("core/postprocess/classifier_score")
    def _processing_classifier_score(
        self,
        outputs: Union[ClassificationOutputs, BaseOutputs],
    ):
        if not hasattr(outputs, "outputs"):
            raise ValueError(f"core/postprocess/classifier_score can't process the outputs")

        assert outputs.outputs.dim() == 2

        if self.act_fn is not None:
            outputs.outputs = self.act_fn(outputs.outputs)

        _outputs = outputs.outputs.numpy()
        pscore = _outputs.max(-1)
        pclass = _outputs.argmax(-1)
        class_score = [
            {
                "class": int(c),
                "score": f"{s:.6f}",
                "scores": o.tolist() if self.return_scores else [],
            }
            for c, s, o in zip(pclass, pscore, _outputs)
        ]
        _infos = outputs.to_dict()
        _infos.pop("outputs")
        return BaseOutputs(**_infos, **{"class_score": class_score})

    @register_process("core/postprocess/embedding")
    def _processing_classifier_score(
        self,
        outputs: Union[EmbeddingOutputs, BaseOutputs],
    ):
        if not hasattr(outputs, "embedding"):
            raise ValueError(f"core/postprocess/classifier_score can't process the outputs")

        embedding = outputs.embedding
        if embedding.dim() > 2:
            embedding = embedding.reshape(embedding.size(0), -1)

        assert embedding.dim() == 2

        embedding = [" ".join(map(lambda x: f"{x:.6f}", emb)) for emb in embedding]
        _infos = outputs.to_dict()
        _infos.pop("embedding")
        return BaseOutputs(**_infos, **{"embedding": embedding})


class ImageProcessor(object):
    def __init__(
        self,
        image_type: str = None,
        image_size: tuple = (256, 256),
        https_image_request_url: str = None,
    ):
        self.image_type = image_type
        self.image_size = image_size
        self.https_image_request_url = https_image_request_url

    @classmethod
    @add_default_section_for_init("core/process/image")
    def from_core_configure(cls, config, **kwargs):
        pass

    def _request_url(self, url):
        while True:
            try:
                doc = requests.get(url, timeout=600)
                return doc
            except:
                time.sleep(random() * 2)

    @register_process("core/process/read_image")
    def _read_image(
        self,
        image,
        image_type=None,
    ):
        image_type = image_type if image_type is not None else self.image_type
        try:
            if image_type == "base64":
                image = io.BytesIO(base64.b64decode(image))
                return Image.open(image)

            if image_type == "hex":
                image = io.BytesIO(bytes.fromhex(image))
                return Image.open(image)

            if self.https_image_request_url is None:
                return Image.open(image)

            url = self.https_image_request_url.format(image)
            doc = self._request_url(url)
            if doc.status_code != 200 or doc.content == b"":
                raise ValueError(f"can't find the image {image}")

            return Image.open(io.BytesIO(doc.content)).convert("RGB")
        except:
            logging.info(f"core/process/read image use fake image for {image}")
            return Image.new("RGB", self.image_size, (255, 255, 255))
