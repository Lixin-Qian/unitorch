
## Installation

#### Requirements

- python version >= 3.6
- [torch](http://pytorch.org/) >= 1.7.0
- fire
- scipy
- pyarrow
- sklearn
- [fairseq](https://github.com/pytorch/fairseq)
- [datasets](https://github.com/huggingface/datasets)
- configparser
- [transformers](https://github.com/huggingface/transformers)



##### Install with pip
```bash
pip install unitorch
pip install unitorch[detection]
```

##### Install from source code
```bash
pip install -e .
```

##### Install detection extension
```bash
pip install git+https://github.com/fuliucansheng/unitorch#egg=unitorch[detection]
```

##### Install detection and enable ngram extension
```bash
pip install git+https://github.com/fuliucansheng/unitorch#egg=unitorch[detection] --global-option="--enable_ngram_cuda_extension"
```

