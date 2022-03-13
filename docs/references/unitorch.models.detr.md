<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detr`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detr.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DetrForDetection`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrForDetection.__init__`

```python
__init__(config_path:str, num_class:Optional[int]=None)
```






---

###### <kbd>property</kbd> DetrForDetection.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 

---

###### <kbd>property</kbd> DetrForDetection.dtype

`torch.dtype`: which dtype the parameters are (assuming that all the parameters are the same dtype). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/docs/unitorch/models/detr/modeling/detect#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrForDetection.detect`

```python
detect(
    images:Union[List[torch.Tensor], torch.Tensor],
    norm_bboxes:Optional[bool]=False
)
```



**Args:**
 
 - <b>`images`</b>:  list of image tensor 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrForDetection.forward`

```python
forward(
    images:Union[List[torch.Tensor], torch.Tensor],
    bboxes:Union[List[torch.Tensor], torch.Tensor],
    classes:Union[List[torch.Tensor], torch.Tensor]
)
```



**Args:**
 
 - <b>`images`</b>:  list of image tensor 
 - <b>`bboxes`</b>:  list of boxes tensor 
 - <b>`classes`</b>:  list of classes tensor 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DetrForSegmentation`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrForSegmentation.__init__`

```python
__init__(
    config_path:str,
    num_class:Optional[int]=None,
    enable_bbox_loss:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to detr model 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`enable_bbox_loss`</b>:  if enable bbox loss for segmentation 


---

###### <kbd>property</kbd> DetrForSegmentation.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 

---

###### <kbd>property</kbd> DetrForSegmentation.dtype

`torch.dtype`: which dtype the parameters are (assuming that all the parameters are the same dtype). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/modeling.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrForSegmentation.forward`

```python
forward(
    images:Union[List[torch.Tensor], torch.Tensor],
    masks:Union[List[torch.Tensor], torch.Tensor],
    bboxes:Union[List[torch.Tensor], torch.Tensor],
    classes:Union[List[torch.Tensor], torch.Tensor]
)
```



**Args:**
 
 - <b>`images`</b>:  list of image tensor 
 - <b>`masks`</b>:  list of mask tensor 
 - <b>`bboxes`</b>:  list of boxes tensor 
 - <b>`classes`</b>:  list of classes tensor 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/docs/unitorch/models/detr/modeling/segment#L404"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrForSegmentation.segment`

```python
segment(
    images:Union[List[torch.Tensor], torch.Tensor],
    norm_bboxes:Optional[bool]=False
)
```



**Args:**
 
 - <b>`images`</b>:  list of image tensor 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detr.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DetrProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrProcessor.__init__`

```python
__init__(
    vision_config_path:str,
    min_size_test:Optional[int]=800,
    max_size_test:Optional[int]=1333
)
```



**Args:**
 
 - <b>`vision_config_path`</b>:  vision config path to detr processor 
 - <b>`min_size_test`</b>:  resize shortest edge parameters 
 - <b>`max_size_test`</b>:  resize shortest edge parameters 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrProcessor.processing_detection`

```python
processing_detection(
    image:PIL.Image.Image,
    bboxes:List[List[float]],
    classes:List[int]
)
```



**Args:**
 
 - <b>`image`</b>:  input image 
 - <b>`bboxes`</b>:  bboxes to image 
 - <b>`classes`</b>:  class to each bbox 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrProcessor.processing_image`

```python
processing_image(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrProcessor.processing_segmentation`

```python
processing_segmentation(
    image:PIL.Image.Image,
    gt_image:PIL.Image.Image,
    num_class:Optional[int]=None
)
```



**Args:**
 
 - <b>`image`</b>:  input image 
 - <b>`gt_image`</b>:  ground truth image 
 - <b>`num_class`</b>:  num classes to classification 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detr/processing.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DetrProcessor.processing_transform`

```python
processing_transform(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 




---

