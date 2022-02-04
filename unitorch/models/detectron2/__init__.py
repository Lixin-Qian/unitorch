# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

try:
    from unitorch.models.detectron2.generalized_rcnn import GeneralizedRCNN
    from unitorch.models.detectron2.processing import GeneralizedRCNNProcessor
except ImportError as error:
    logging.warning("detectron2 model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
