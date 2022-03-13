<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.vit`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.vit.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/modeling.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `ViTForImageClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/modeling.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `ViTForImageClassification.__init__`

```python
__init__(config_path:str, num_class:Optional[int]=1)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to vit model 
 - <b>`num_class`</b>:  num class to classification 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/modeling.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `ViTForImageClassification.forward`

```python
forward(pixel_values=None, output_attentions=None, output_hidden_states=None)
```



**Args:**
 
 - <b>`pixel_values`</b>:  pixels of image 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.vit.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/processing.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `ViTProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/processing.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `ViTProcessor.__init__`

```python
__init__(vision_config_path:str)
```



**Args:**
 
 - <b>`vision_config_path`</b>:  vision config path to vit processor 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vit/processing.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `ViTProcessor.processing_image_classifictaion`

```python
processing_image_classifictaion(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 




---

