# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

# pretrained_detectron2_infos
pretrained_detectron2_infos = {
    "default-rcnn": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/faster-rcnn/pascal-voc-detection/faster_rcnn_r50_c4_config.yaml",
    },
    "pascal-voc-detection/faster-rcnn-r50-c4": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/faster-rcnn/pascal-voc-detection/faster_rcnn_r50_c4_config.yaml",
        "weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/faster-rcnn/pascal-voc-detection/faster_rcnn_r50_c4_model.bin",
    },
    "default-yolo": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5s.yaml",
    },
    "yolov5/yolov5s": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5s.yaml",
        "weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5s.bin",
    },
    "yolov5/yolov5m": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5m.yaml",
        "weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5m.bin",
    },
    "yolov5/yolov5l": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5l.yaml",
        "weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5l.bin",
    },
    "yolov5/yolov5x": {
        "config": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5x.yaml",
        "weight": "https://huggingface.co/fuliucansheng/detectron2/resolve/main/yolo/v5/yolov5x.bin",
    },
}

try:
    import unitorch.cli.models.detectron2.generalized_rcnn
    import unitorch.cli.models.detectron2.yolo
    import unitorch.cli.models.detectron2.processing
except ImportError as error:
    logging.warning("detectron2 cli model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
