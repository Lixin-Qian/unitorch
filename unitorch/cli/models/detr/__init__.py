# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

# pretrained infos
pretrained_detr_infos = {
    "default-detr": {
        "config": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/preprocessor_config.json",
    },
    "detr-resnet-50": {
        "config": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/pytorch_model.bin",
    },
}

try:
    import unitorch.cli.models.detr.modeling
    import unitorch.cli.models.detr.processing
except ImportError as error:
    logging.warning("detr cli model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
