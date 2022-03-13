<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detectron2`




### **Global Variables**
---------------
- **backbone**

- **meta_arch**

- **generalized_rcnn**

- **processing**

- **yolo**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/generalized_rcnn.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detectron2.generalized_rcnn`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/generalized_rcnn.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `GeneralizedRCNN`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/generalized_rcnn.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedRCNN.__init__`

```python
__init__(detectron2_config_path:str)
```






---

###### <kbd>property</kbd> GeneralizedRCNN.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 

---

###### <kbd>property</kbd> GeneralizedRCNN.dtype

`torch.dtype`: which dtype the parameters are (assuming that all the parameters are the same dtype). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/docs/unitorch/models/detectron2/generalized_rcnn/detect#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedRCNN.detect`

```python
detect(
    images:Union[List[torch.Tensor], torch.Tensor],
    return_features:Optional[bool]=False,
    norm_bboxes:Optional[bool]=False
)
```



**Args:**
 
 - <b>`images`</b>:  list of image tensor 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/generalized_rcnn.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedRCNN.forward`

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


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detectron2.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/processing.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `GeneralizedProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/processing.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.__init__`

```python
__init__(
    pixel_mean:List[float],
    pixel_std:List[float],
    resize_shape:List[int]=[224, 224],
    min_size_test:Optional[int]=800,
    max_size_test:Optional[int]=1333
)
```



**Args:**
 
 - <b>`pixel_mean`</b>:  pixel means to process image 
 - <b>`pixel_std`</b>:  pixel std to process image 
 - <b>`resize_shape`</b>:  shape to resize image 
 - <b>`min_size_test`</b>:  resize shortest edge parameters 
 - <b>`max_size_test`</b>:  resize shortest edge parameters 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/processing.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_detection`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/processing.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_image`

```python
processing_image(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/processing.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_transform`

```python
processing_transform(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/yolo.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.detectron2.yolo`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/yolo.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `YoloForDetection`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/yolo.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `YoloForDetection.__init__`

```python
__init__(detectron2_config_path:str)
```



**Args:**
 
 - <b>`detectron2_config_path`</b>:  config file path to generalized yolo model 


---

###### <kbd>property</kbd> YoloForDetection.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 

---

###### <kbd>property</kbd> YoloForDetection.dtype

`torch.dtype`: which dtype the parameters are (assuming that all the parameters are the same dtype). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/docs/unitorch/models/detectron2/yolo/detect#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `YoloForDetection.detect`

```python
detect(
    images:Union[List[torch.Tensor], torch.Tensor],
    norm_bboxes:Optional[bool]=False
)
```



**Args:**
 
 - <b>`images`</b>:  list of image tensor 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/yolo.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `YoloForDetection.forward`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/detectron2/yolo.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `YoloForDetection.from_pretrained`

```python
from_pretrained(weight_path:str, replace_keys:Dict=OrderedDict(), **kwargs)
```

Load model's pretrained weight 

**Args:**
 
 - <b>`weight_path`</b>:  the path of pretrained weight of mbart 




---

