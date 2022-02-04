# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_vit_mae_infos = {
    "default-vit-mae": {
        "config": "https://huggingface.co/facebook/vit-mae-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/vit-mae-base/resolve/main/preprocessor_config.json",
    },
    "vit-mae-base": {
        "config": "https://huggingface.co/facebook/vit-mae-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/vit-mae-base/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/vit-mae-base/resolve/main/pytorch_model.bin",
    },
    "vit-mae-large": {
        "config": "https://huggingface.co/facebook/vit-mae-large/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/vit-mae-large/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/vit-mae-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.vit_mae.modeling
import unitorch.cli.models.vit_mae.processing
