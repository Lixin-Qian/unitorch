<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.clip`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.clip.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CLIPForPretrain`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForPretrain.__init__`

```python
__init__(
    config_path:str,
    projection_dim:Optional[int]=512,
    freeze_base_model:Optional[bool]=True,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to clip model 
 - <b>`projection_dim`</b>:  dimension to image/text output embedding 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`freeze_base_model`</b>:  if to freeze base model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForPretrain.forward`

```python
forward(
    input_ids=None,
    pixel_values=None,
    attention_mask=None,
    position_ids=None,
    output_attentions=None,
    output_hidden_states=None
)
```



**Args:**
 
 - <b>`input_ids`</b>:  tokens of text 
 - <b>`pixel_values`</b>:  pixels of image 
 - <b>`attention_mask`</b>:  attention mask of tokens 
 - <b>`position_ids`</b>:  position ids 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CLIPForClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForClassification.__init__`

```python
__init__(
    config_path:str,
    projection_dim:int=512,
    num_class:int=1,
    freeze_base_model=True,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to clip model 
 - <b>`projection_dim`</b>:  dimension to image/text output embedding 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`freeze_base_model`</b>:  if to freeze base model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForClassification.forward`

```python
forward(
    input_ids=None,
    pixel_values=None,
    attention_mask=None,
    position_ids=None,
    output_attentions=None,
    output_hidden_states=None
)
```



**Args:**
 
 - <b>`input_ids`</b>:  tokens of text 
 - <b>`pixel_values`</b>:  pixels of image 
 - <b>`attention_mask`</b>:  attention mask of tokens 
 - <b>`position_ids`</b>:  position ids 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CLIPForTextClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForTextClassification.__init__`

```python
__init__(
    config_path:str,
    projection_dim:int=512,
    num_class:int=1,
    freeze_base_model=True,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to clip model 
 - <b>`projection_dim`</b>:  dimension to image/text output embedding 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`freeze_base_model`</b>:  if to freeze base model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForTextClassification.forward`

```python
forward(
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    output_attentions=None,
    output_hidden_states=None
)
```



**Args:**
 
 - <b>`input_ids`</b>:  tokens of text 
 - <b>`attention_mask`</b>:  attention mask of tokens 
 - <b>`position_ids`</b>:  position ids 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CLIPForImageClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForImageClassification.__init__`

```python
__init__(
    config_path:str,
    projection_dim:int=512,
    num_class:int=1,
    freeze_base_model=True,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to clip model 
 - <b>`projection_dim`</b>:  dimension to image/text output embedding 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`freeze_base_model`</b>:  if to freeze base model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/modeling.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPForImageClassification.forward`

```python
forward(pixel_values=None, output_attentions=None, output_hidden_states=None)
```



**Args:**
 
 - <b>`pixel_values`</b>:  pixels of image 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.clip.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/processing.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CLIPProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/processing.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPProcessor.__init__`

```python
__init__(
    vocab_path:str,
    merge_path:str,
    vision_config_path:str,
    max_seq_length:Optional[int]=128,
    position_start_id:Optional[int]=0
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in bart tokenizer 
 - <b>`merge_path`</b>:  merge file path in bart tokenizer 
 - <b>`vision_config_path`</b>:  vision config path to clip processor 
 - <b>`max_seq_length`</b>:  max sequence length encode text 
 - <b>`position_start_id`</b>:  start id of position 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/processing.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPProcessor.processing_classification`

```python
processing_classification(
    text:str,
    image:PIL.Image.Image,
    max_seq_length:Optional[int]=None
)
```



**Args:**
 
 - <b>`text`</b>:  input text 
 - <b>`image`</b>:  input image 
 - <b>`max_seq_length`</b>:  max sequence length to input text 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/processing.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPProcessor.processing_image_classifictaion`

```python
processing_image_classifictaion(image:PIL.Image.Image)
```



**Args:**
 
 - <b>`image`</b>:  input image 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/clip/processing.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CLIPProcessor.processing_text_classifictaion`

```python
processing_text_classifictaion(text:str, max_seq_length:Optional[int]=None)
```



**Args:**
 
 - <b>`text`</b>:  input text 
 - <b>`max_seq_length`</b>:  max sequence length to input text 




---

