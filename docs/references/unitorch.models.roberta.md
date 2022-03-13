<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.roberta`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.roberta.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `RobertaForClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `RobertaForClassification.__init__`

```python
__init__(
    config_path:str,
    num_class:Optional[int]=1,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to roberta model 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `RobertaForClassification.forward`

```python
forward(
    tokens_ids:torch.Tensor,
    attn_mask:Optional[torch.Tensor]=None,
    seg_ids:Optional[torch.Tensor]=None,
    pos_ids:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`tokens_ids`</b>:  tokens of text 
 - <b>`attn_mask`</b>:  attention mask of tokens 
 - <b>`seg_ids`</b>:  token type ids 
 - <b>`pos_ids`</b>:  position ids 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `RobertaForMaskLM`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `RobertaForMaskLM.__init__`

```python
__init__(config_path:str, gradient_checkpointing:Optional[bool]=False)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to roberta model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/modeling.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `RobertaForMaskLM.forward`

```python
forward(
    tokens_ids:torch.Tensor,
    attn_mask:Optional[torch.Tensor]=None,
    seg_ids:Optional[torch.Tensor]=None,
    pos_ids:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`tokens_ids`</b>:  tokens of text 
 - <b>`attn_mask`</b>:  attention mask of tokens 
 - <b>`seg_ids`</b>:  token type ids 
 - <b>`pos_ids`</b>:  position ids 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.roberta.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/processing.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `RobertaProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/processing.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `RobertaProcessor.__init__`

```python
__init__(
    vocab_path:str,
    merge_path:str,
    max_seq_length:int=128,
    source_type_id:int=0,
    target_type_id:int=0
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in roberta tokenizer 
 - <b>`merge_path`</b>:  merge file path in roberta tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length encode text 
 - <b>`source_type_id`</b>:  token type id to text_a 
 - <b>`target_type_id`</b>:  token type id to text_b 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/roberta/processing.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `RobertaProcessor.processing_classification`

```python
processing_classification(
    text:str,
    text_pair:str=None,
    max_seq_length:Optional[int]=None
)
```



**Args:**
 
 - <b>`text`</b>:  encode text 
 - <b>`text_pair`</b>:  decode text 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 




---

