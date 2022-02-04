# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os

# env setting
os.environ["TRANSFORMERS_CACHE"] = os.getenv(
    "UNITORCH_CACHE", os.path.join(os.getenv("HOME"), ".cache/unitorch")
)

os.environ["HF_DATASETS_CACHE"] = os.getenv(
    "UNITORCH_CACHE", os.path.join(os.getenv("HOME"), ".cache/unitorch")
)

# logging & warning setting
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=logging.INFO,
)

# settings
import torch
import random
import numpy as np
import transformers
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# useful functions
from transformers import cached_path as hf_cached_path

# base imports
import unitorch.common
import unitorch.datasets
import unitorch.loss
import unitorch.score
import unitorch.models
import unitorch.optim
import unitorch.scheduler
import unitorch.utils
import unitorch.writer

# more classes
