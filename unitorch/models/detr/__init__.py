# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

try:
    from unitorch.models.detr.modeling import DetrForDetection, DetrForSegmentation
    from unitorch.models.detr.processing import DetrProcessor
except ImportError as error:
    logging.warning("detr model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
