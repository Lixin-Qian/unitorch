# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_mass_infos = {
    # mass
    "default-mass": {
        "config": "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-config.json",
        "vocab": "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-vocab.txt",
    },
    "mass-base-uncased": {
        "config": "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-config.json",
        "vocab": "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-vocab.txt",
        "weight": "https://huggingface.co/fuliucansheng/mass/resolve/main/mass-base-uncased-pytorch-model.bin",
    },
}

import unitorch.cli.models.mass.modeling
import unitorch.cli.models.mass.processing
