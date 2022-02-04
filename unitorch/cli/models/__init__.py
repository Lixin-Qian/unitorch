# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli.models.modeling_utils import (
    BaseInputs,
    BaseOutputs,
    BaseTargets,
    ListInputs,
    LossOutputs,
    EmbeddingOutputs,
    ClassificationOutputs,
    ClassificationTargets,
    GenerationOutputs,
    GenerationTargets,
    DetectionOutputs,
    DetectionTargets,
    SegmentationOutputs,
    SegmentationTargets,
)
from unitorch.cli.models.modeling_utils import (
    general_model_decorator,
    generation_model_decorator,
    detection_model_decorator,
    segmentation_model_decorator,
)

# import model classes & process functions
import unitorch.cli.models.processing_utils
import unitorch.cli.models.bart
import unitorch.cli.models.bert
import unitorch.cli.models.clip
import unitorch.cli.models.deberta
import unitorch.cli.models.detectron2
import unitorch.cli.models.mass
import unitorch.cli.models.mbart
import unitorch.cli.models.prophetnet
import unitorch.cli.models.roberta
import unitorch.cli.models.xprophetnet
import unitorch.cli.models.unilm
import unitorch.cli.models.vlp
import unitorch.cli.models.infoxlm
import unitorch.cli.models.senet
import unitorch.cli.models.vit
import unitorch.cli.models.vit_mae
import unitorch.cli.models.swin
import unitorch.cli.models.detr
