# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

pretrained_senet_infos = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}


import unitorch.cli.models.senet.modeling
import unitorch.cli.models.senet.processing
