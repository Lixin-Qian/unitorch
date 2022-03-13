
## Overview

> This is the structure of this library.

![Overview](/sources/images/overview.png)

#### Base Modules
Base modules are mainly designed to finish basic functions and they only focus on the implemented functionality itself.


#### Adapter Modules
Adapter modules are the adapters for base modules, to make them support different workflows. A base module may have several adapters to meet the different requirements from different workflows.


#### Command Line Interface
Command line defines a running workflow, and it will call the needed adapter modules according to the pipeline design.

#### Supported Models
* **Text Classification**
  - [x] Bert
  - [x] Roberta
  - [x] Deberta & DebertaV2

* **Text Generation**
  - [x] Bart
  - [x] MASS
  - [x] Unilm

* **Click Prediction**
  - [ ] DeepFM

* **Multi-Lingual Generation**
  - [x] InfoXLM
  - [x] MBart
  - [x] XProphetNet

* **Image Classification**
  - [x] SeNet
  - [x] VIT
  - [x] MAE VIT
  - [x] Swin

* **Object Detection**
  - [x] YoloV5
  - [x] FasterRCNN
  - [x] DETR

* **Image Segmentation**
  - [x] DETR

* **Image Caption**
  - [x] VLP

* **Multi-Modal Classification**
  - [x] CLIP