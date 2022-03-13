<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.vlp`




### **Global Variables**
---------------
- **processing**

- **modeling**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.vlp.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `VLPForGeneration`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForGeneration.__init__`

```python
__init__(
    vlp_config_path:str,
    detectron2_config_path:str,
    freeze_vision_model:Optional[bool]=True,
    max_num_bbox:Optional[int]=100
)
```



**Args:**
 
 - <b>`vlp_config_path`</b>:  config file path to text part of vlp model 
 - <b>`detectron2_config_path`</b>:  config file path to image part of vlp model (faster-rcnn based on detectron2) 
 - <b>`freeze_vision_model`</b>:  if to freeze image part of model 
 - <b>`max_num_bbox`</b>:  max num bbox returns from faster-rcnn 


---

###### <kbd>property</kbd> VLPForGeneration.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L395"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForGeneration.forward`

```python
forward(
    tokens_ids=None,
    attn_mask=None,
    seg_ids=None,
    pos_ids=None,
    pixel_values=None,
    decoder_input_ids=None,
    decoder_pos_ids=None,
    decoder_seg_ids=None,
    decoder_attn_mask=None,
    decoder_mask_ids=None,
    decoder_pixel_mask=None,
    past_key_values=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None
)
```



**Args:**
 
 - <b>`tokens_ids`</b>:  tokens of encode text & decode 
 - <b>`attn_mask`</b>:  attention mask of tokens 
 - <b>`seg_ids`</b>:  token type ids 
 - <b>`pos_ids`</b>:  position ids 
 - <b>`pixel_values`</b>:  pixels of images 
 - <b>`others`</b>:  used in beam search Returns: forward logits 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForGeneration.from_pretrained`

```python
from_pretrained(weight_path)
```

Load model's pretrained weight 

**Args:**
 
 - <b>`weight_path`</b>:  the path of pretrained weight of mbart 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L467"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForGeneration.generate`

```python
generate(
    pixel_values,
    tokens_ids=None,
    num_beams=5,
    decoder_start_token_id=101,
    decoder_end_token_id=102,
    num_return_sequences=1,
    min_gen_seq_length=0,
    max_gen_seq_length=48,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    early_stopping=True,
    length_penalty=1.0,
    num_beam_groups=1,
    diversity_penalty=0.0,
    diverse_rate=0.0,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0
)
```



**Args:**
 
 - <b>`tokens_ids`</b>:  tokens of encode text 
 - <b>`pixel_values`</b>:  pixels of images 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForGeneration.prepare_inputs_for_generation`

```python
prepare_inputs_for_generation(decoder_input_ids, past=None, **kwargs)
```

Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForGeneration.train`

```python
train(mode=True)
```






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L574"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `VLPForClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L575"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForClassification.__init__`

```python
__init__(
    vlp_config_path,
    detectron2_config_path,
    freeze_vision_model:bool=True,
    freeze_base_model:bool=False,
    max_num_bbox:int=100,
    num_class:int=1
)
```






---

###### <kbd>property</kbd> VLPForClassification.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L720"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForClassification.forward`

```python
forward(
    tokens_ids=None,
    attn_mask=None,
    seg_ids=None,
    pos_ids=None,
    pixel_values=None
)
```



**Args:**
 
 - <b>`tokens_ids`</b>:  tokens of encode text & decode 
 - <b>`attn_mask`</b>:  attention mask of tokens 
 - <b>`seg_ids`</b>:  token type ids 
 - <b>`pos_ids`</b>:  position ids 
 - <b>`pixel_values`</b>:  pixels of images Returns: forward logits 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L633"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForClassification.from_pretrained`

```python
from_pretrained(weight_path)
```

Load model's pretrained weight 

**Args:**
 
 - <b>`weight_path`</b>:  the path of pretrained weight of mbart 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/modeling.py#L624"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPForClassification.train`

```python
train(mode=True)
```








---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.vlp.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `VLPProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPProcessor.__init__`

```python
__init__(
    vocab_path,
    max_seq_length:Optional[int]=128,
    max_gen_seq_length:Optional[int]=30,
    do_lower_case:Optional[bool]=False,
    do_basic_tokenize:Optional[bool]=False,
    special_tokens_ids:Optional[Dict]={},
    source_type_id:Optional[int]=0,
    target_type_id:Optional[int]=1,
    pixel_mean:Optional[List[float]]=[123.675, 116.28, 103.53],
    pixel_std:Optional[List[float]]=[1.0, 1.0, 1.0],
    resize_shape:Optional[List[int]]=[224, 224]
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in vlp tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length text 
 - <b>`max_gen_seq_length`</b>:  max sequence length decode text 
 - <b>`do_lower_case`</b>:  if do lower case to input text 
 - <b>`do_basic_tokenize`</b>:  if do basic tokenize to input text 
 - <b>`special_tokens_ids`</b>:  special tokens dict in vlp tokenizer 
 - <b>`source_type_id`</b>:  token type id to encode text 
 - <b>`target_type_id`</b>:  token type id to decode text 
 - <b>`pixel_mean`</b>:  pixel means to process image 
 - <b>`pixel_std`</b>:  pixel std to process image 
 - <b>`resize_shape`</b>:  shape to resize image 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPProcessor.processing_caption`

```python
processing_caption(
    image:PIL.Image.Image,
    text:str,
    max_gen_seq_length:Optional[int]=None
)
```



**Args:**
 
 - <b>`image`</b>:  input image 
 - <b>`text`</b>:  decode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length to decode text 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPProcessor.processing_classification`

```python
processing_classification(
    image:PIL.Image.Image,
    text:str,
    text_pair:Optional[str]=None,
    max_seq_length:Optional[int]=None
)
```





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPProcessor.processing_generation`

```python
processing_generation(
    image:PIL.Image.Image,
    text:str,
    text_pair:str,
    max_seq_length:Optional[int]=None,
    max_gen_seq_length:Optional[int]=None
)
```



**Args:**
 
 - <b>`image`</b>:  input image 
 - <b>`text`</b>:  encode text 
 - <b>`text_pair`</b>:  decode text 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length to decode text 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPProcessor.processing_image`

```python
processing_image(image:PIL.Image.Image)
```





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/vlp/processing.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `VLPProcessor.processing_inference`

```python
processing_inference(
    image:PIL.Image.Image,
    text:str,
    max_seq_length:Optional[int]=None
)
```



**Args:**
 
 - <b>`image`</b>:  input image 
 - <b>`text`</b>:  encode text 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 




---

