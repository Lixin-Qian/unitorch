# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_vlp_infos = {
    # vlp
    "default-vlp": {
        "detectron2_config": "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/FasterRCNN_X_101_64x4d_FPN_2x_config.yaml",
        "config": "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/config.json",
        "vocab": "https://huggingface.co/microsoft/unilm-base-cased/resolve/main/vocab.txt",
    },
    "vlp-coco": {
        "detectron2_config": "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/FasterRCNN_X_101_64x4d_FPN_2x_config.yaml",
        "detectron2_weight": "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/FasterRCNN_X_101_64x4d_FPN_2x_pytorch_model.bin",
        "config": "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/config.json",
        "weight": "https://huggingface.co/fuliucansheng/detection/resolve/main/VLP/pytorch_model.bin",
        "vocab": "https://huggingface.co/microsoft/unilm-base-cased/resolve/main/vocab.txt",
    },
}

import sys
import logging

try:
    import unitorch.cli.models.vlp.modeling
    import unitorch.cli.models.vlp.processing
except ImportError as error:
    logging.warning("vlp cli model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
