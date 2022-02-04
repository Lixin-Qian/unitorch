# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_infoxlm_infos = {
    # infoxlm
    "default-infoxlm": {
        "config": "https://huggingface.co/fuliucansheng/unilm/resolve/main/infoxlm-roberta-config.json",
        "vocab": "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
    },
    "infoxlm-roberta": {
        "config": "https://huggingface.co/fuliucansheng/unilm/resolve/main/infoxlm-roberta-config.json",
        "vocab": "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
        "weight": "https://huggingface.co/fuliucansheng/unilm/resolve/main/default-infoxlm-pytorch-model.bin",
    },
}

import unitorch.cli.models.infoxlm.modeling
import unitorch.cli.models.infoxlm.processing
