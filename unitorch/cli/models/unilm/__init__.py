# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_unilm_infos = {
    # unilm
    "default-unilm": {
        "config": "https://huggingface.co/fuliucansheng/unilm/resolve/main/unilm-base-uncased-config.json",
        "vocab": "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-vocab.txt",
    },
    "unilm-base-uncased": {
        "config": "https://huggingface.co/fuliucansheng/unilm/resolve/main/unilm-base-uncased-config.json",
        "vocab": "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-vocab.txt",
        "weight": "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin",
    },
    "unilm-base-cased": {
        "config": "https://huggingface.co/microsoft/unilm-base-cased/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/unilm-base-cased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/microsoft/unilm-base-cased/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.unilm.modeling
import unitorch.cli.models.unilm.processing
