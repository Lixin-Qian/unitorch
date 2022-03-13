<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.senet`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.senet.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/modeling.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `SeResNet`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/modeling.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SeResNet.__init__`

```python
__init__(arch:str, num_class:int)
```



**Args:**
 
 - <b>`arch`</b>:  model structure, one of ['resnet18', 'resnet50', 'resnet101', 'resnet152'] 
 - <b>`num_class`</b>:  num class to classification 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/modeling.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SeResNet.forward`

```python
forward(pixel_values:torch.Tensor)
```








---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.senet.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/processing.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `SeNetProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/processing.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SeNetProcessor.__init__`

```python
__init__(
    pixel_mean:List[float]=[0.4039, 0.4549, 0.4823],
    pixel_std:List[float]=[1.0, 1.0, 1.0],
    resize_shape:Optional[List[int]]=[224, 224],
    crop_shape:Optional[List[int]]=[224, 224]
)
```



**Args:**
 
 - <b>`pixel_mean`</b>:  pixel means to process image 
 - <b>`pixel_std`</b>:  pixel std to process image 
 - <b>`resize_shape`</b>:  shape to resize image 
 - <b>`crop_shape`</b>:  shape to crop image 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/senet/processing.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `SeNetProcessor.processing`

```python
processing(image:PIL.Image.Image)
```








---

