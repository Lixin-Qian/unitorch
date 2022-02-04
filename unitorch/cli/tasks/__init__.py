# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import logging
import unitorch.cli.tasks.supervised_task

try:
    import unitorch.cli.tasks.deepspeed_task
except ImportError as error:
    logging.warning("deepspeed can't be imported.")
except:
    logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
