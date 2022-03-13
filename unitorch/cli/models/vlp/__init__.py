# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_vlp_infos = {
    # vlp
    "default-vlp": {
        "detectron2_config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/faster_rcnn_x101_64x4d_fpn_2x_config.yaml",
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/config.json",
        "vocab": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/vocab.txt",
    },
    "vlp-coco": {
        "detectron2_config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/faster_rcnn_x101_64x4d_fpn_2x_config.yaml",
        "detectron2_weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/faster_rcnn_x101_64x4d_fpn_2x_model.bin",
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/config.json",
        "vocab": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/vocab.txt",
        "weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/vlp/pytorch_model.bin",
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
