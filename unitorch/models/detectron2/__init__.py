# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

try:
    import unitorch.models.detectron2.backbone
    import unitorch.models.detectron2.meta_arch
    from unitorch.models.detectron2.generalized_rcnn import GeneralizedRCNN
    from unitorch.models.detectron2.processing import GeneralizedProcessor
    from unitorch.models.detectron2.yolo import YoloForDetection
except ImportError as error:
    logging.warning("detectron2 model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
