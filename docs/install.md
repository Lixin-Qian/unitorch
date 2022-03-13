
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
pip3 install unitorch
pip3 install unitorch[detectron2]
```

##### Install from source code
```bash
pip3 install .
```

##### Install detection extension
```bash
pip3 install git+https://github.com/fuliucansheng/unitorch#egg=unitorch[detectron2]
```

##### Install detection and enable ngram extension
```bash
pip3 install git+https://github.com/fuliucansheng/unitorch#egg=unitorch[detectron2] --global-option="--enable_ngram_cuda_extension"
```

