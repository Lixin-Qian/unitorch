# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

# pretrained_generalized_rcnn_infos
pretrained_generalized_rcnn_infos = {
    "default-rcnn": {
        "config": "https://huggingface.co/fuliucansheng/detection/resolve/main/PascalVOC_Detection/FasterRCNN_R50_C4.yaml",
    },
    "pascal-voc-detection/faster-rcnn-r50-c4": {
        "config": "https://huggingface.co/fuliucansheng/detection/resolve/main/PascalVOC_Detection/FasterRCNN_R50_C4.yaml",
        "weight": "https://huggingface.co/fuliucansheng/detection/resolve/main/PascalVOC_Detection/pytorch_model.bin",
    },
}
try:
    import unitorch.cli.models.detectron2.generalized_rcnn
    import unitorch.cli.models.detectron2.processing
except ImportError as error:
    logging.warning("detectron2 cli model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
