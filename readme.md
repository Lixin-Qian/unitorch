<h1 align="Center"> <p> unitorch </p> </h1>

## Introduction

unitorch provides efficient implementation of popular unified NLU / NLG / CV / MM / RL models with PyTorch. It automatically optimizes training / inference speed based on pupular DeepLearning toolkits (transformers, fairseq, fastseq, etc) without accuracy loss. All these can be easily done (no need to change any code/model/data if using our command line tool, or simply add one-line code ` import unitorch` if using source code).

## Installation

```bash
pip3 install unitorch

# install from git repo
### enable no repeat ngram cuda kernel
python3 setup.py install --enable_ngram_cuda_extension

### install detection project
pip3 install git+https://github.com/fuliucansheng/unitorch#egg=project[detection] # with ngram cuda extension
pip3 install git+https://github.com/fuliucansheng/unitorch#egg=project[detection] --global-option="--enable_ngram_cuda_extension" # with ngram cuda extension

```

### Requirements

- python version >= 3.6
- fire
- scipy
- pyarrow
- sklearn
- configparser
- [datasets](https://github.com/huggingface/datasets)
- [fairseq](https://github.com/pytorch/fairseq)
- [torch](http://pytorch.org/) >= 1.7.0
- [transformers](https://github.com/huggingface/transformers)

## Usage

### Use source code

```python
# import unitorch at the beginning of your program
import unitorch

# use as general package
from unitorch.models.unilm import UnilmForGeneration
unilm_model = UnilmForGeneration("path/to/unilm/config.json")

# use auto/cli mode which need a config file for lib init
from unitorch import CoreConfigureParser
config = CoreConfigureParser("path/to/config.ini")

```

### Use unitorch cli
```bash
# only use config
unitorch-train path/to/config.ini \
	--core/model/bert@config_name_or_file bert-large-uncased

# run custom code using unitorch auto mode (like fairseq-cli)
unitorch-train path/to/code/directory \
	--core/model/bert@config_name_or_file bert-large-cased
```

### For Quick Examples In Generation Models
```
# one line for training
## set cache folder
export UNITORCH_CACHE="cache/unitorch"

## for single gpu
unitorch-train examples/configs/generation/mass.ini --train_file path/to/train.tsv --dev_file path/to/dev.tsv
## for ddp mode
python3 -m torch.distributed.launch --use_env --no_python --nproc_per_node 4 \
	unitorch-train examples/configs/generation/mass.ini \
	--train_file path/to/train.tsv --dev_file path/to/dev.tsv

# one line for inference (config file is in examples folder)
unitorch-infer examples/configs/generation/mass.ini --test_file path/to/test.tsv
```
