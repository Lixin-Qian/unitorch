<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.infoxlm`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.infoxlm.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/modeling.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `InfoXLMForGeneration`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/modeling.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `InfoXLMForGeneration.__init__`

```python
__init__(config_path:str, freeze_word_embedding:Optional[bool]=True)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to infoxlm model 
 - <b>`freeze_word_embedding`</b>:  if to freeze word embedding in infoxlm model 


---

###### <kbd>property</kbd> InfoXLMForGeneration.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 






---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.infoxlm.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/processing.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `InfoXLMProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/processing.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `InfoXLMProcessor.__init__`

```python
__init__(
    vocab_path:str,
    max_seq_length:Optional[int]=128,
    max_gen_seq_length:Optional[int]=30,
    special_tokens_ids:Optional[Dict]={},
    source_type_id:Optional[int]=0,
    target_type_id:Optional[int]=0
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in bert tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length text 
 - <b>`max_gen_seq_length`</b>:  max sequence length decode text 
 - <b>`special_tokens_ids`</b>:  special tokens dict in bert tokenizer 
 - <b>`source_type_id`</b>:  token type id to text_a 
 - <b>`target_type_id`</b>:  token type id to text_b 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/infoxlm/processing.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `InfoXLMProcessor.processing_generation`

```python
processing_generation(
    text:str,
    text_pair:str,
    max_seq_length:Optional[int]=None,
    max_gen_seq_length:Optional[int]=None
)
```



**Args:**
 
 - <b>`text`</b>:  encode text 
 - <b>`text_pair`</b>:  decode text 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length to decode text 




---

