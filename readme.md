<div align="Center"> 

![unitorch](unitorch.png)


[Documentation](https://fuliucansheng.github.io/unitorch) •
[Installation Instructions](https://fuliucansheng.github.io/unitorch/#/install) •
[Reporting Issues](https://github.com/fuliucansheng/unitorch/issues/new?assignees=&labels=&template=bug-report.yml)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unitorch)](https://pypi.org/project/unitorch/)
[![PyPI Version](https://badge.fury.io/py/unitorch.svg)](https://badge.fury.io/py/unitorch)
[![PyPI Downloads](https://pepy.tech/badge/unitorch)](https://pepy.tech/project/unitorch)
[![Github Downloads](https://img.shields.io/github/downloads/fuliucansheng/unitorch/total?color=blue&label=downloads&logo=github&logoColor=lightgrey)](https://img.shields.io/github/downloads/fuliucansheng/unitorch/total?color=blue&label=Downloads&logo=github&logoColor=lightgrey)

[![License](https://img.shields.io/github/license/fuliucansheng/unitorch?color=dfd)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/fuliucansheng/unitorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)

</div>

# Introduction
 
🔥 unitorch provides efficient implementation of popular unified NLU / NLG / CV / CTR / MM / RL models with PyTorch. It automatically optimizes training / inference speed based on pupular DeepLearning toolkits (transformers, fairseq, detectron2, fastseq, datasets, etc) without accuracy loss. All these can be easily done (no need to change any code/model/data if using our command line tool, or simply add one-line code ` import unitorch` if using source code).

------------------------------------

# What's New Model

* **ViTMAE** released with the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick.
* **Swin Transformer** released with the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
* **CLIP** released with the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
* **YOLOV5** released with the github [YOLOV5](https://github.com/ultralytics/yolov5) by Glenn Jocher.
* **Vision Transformer (ViT)** released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
* **INFOXLM** released with the paper [INFOXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training](https://arxiv.org/abs/2007.07834) by Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, Ming Zhou.
* **DeBERTa-V2** released with the paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
* **DeBERTa** released with the paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
* **DETR** released with the paper [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
* **MBart** released with the paper [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
* **XProphetNet** released with the paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
* **ProphetNet** released with the paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
* **BART** released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
* **VLP** released with the paper [Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/abs/1909.11059) by Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao.
* **RoBERTa** released together with the paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
* **Unilm** released together with the paper [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197) by Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon.
* **MASS** released together with the paper [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450) by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.
* **BERT** released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
* **SENet** released with the paper [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) by Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu.
* **Faster-RCNN** released with the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) by Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.

------------------------------------

# Features

* GPU-Based Block N-Gram Repeats
* Asynchronous Pipeline For Postprocess
* DeepSpeed Supports

# Installation

```bash
pip3 install unitorch
```

# Quick Examples

### Source Code
```python
import unitorch

# import unilm model
from unitorch.models.unilm import UnilmForGeneration
unilm_model = UnilmForGeneration("path/to/unilm/config.json")

# use the configuration class
from unitorch.cli import CoreConfigureParser
config = CoreConfigureParser("path/to/config.ini")
```

### Multi-GPU Training
```bash
python3 -m torch.distributed.launch --use_env --no_python --nproc_per_node 4 \
	unitorch-train examples/configs/generation/mass.ini \
	--train_file path/to/train.tsv --dev_file path/to/dev.tsv
```

### Single-GPU Inference
```bash
unitorch-infer examples/configs/generation/mass.ini --test_file path/to/test.tsv
```

> **Find more details in the Tutorials section of the [documentation](https://fuliucansheng.github.io/unitorch).**


# License

Code released under MIT license.
