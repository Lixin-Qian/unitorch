# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging

try:
    from unitorch.models.vlp.processing import VLPProcessor
    from unitorch.models.vlp.modeling import VLPForGeneration, VLPForClassification
except ImportError as error:
    logging.warning("vlp model can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
