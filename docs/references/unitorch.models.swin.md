<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.swin`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.swin.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/modeling.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `SwinForImageClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/modeling.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SwinForImageClassification.__init__`

```python
__init__(config_path:str, num_class:Optional[int]=1)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to swin model 
 - <b>`num_class`</b>:  num class to classification 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/modeling.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SwinForImageClassification.forward`

```python
forward(pixel_values=None, output_attentions=None, output_hidden_states=None)
```



**Args:**
 
 - <b>`pixel_values`</b>:  pixels of image 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.swin.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/processing.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `SwinProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/processing.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SwinProcessor.__init__`

```python
__init__(vision_config_path:str)
```



**Args:**
 
 - <b>`vision_config_path`</b>:  vision config path to swin processor 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/swin/processing.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SwinProcessor.processing_image_classifictaion`

```python
processing_image_classifictaion(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 




---

