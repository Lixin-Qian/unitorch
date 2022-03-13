<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.deberta`




### **Global Variables**
---------------
- **modeling**

- **processing**

- **modeling_v2**

- **processing_v2**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.deberta.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DebertaForClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaForClassification.__init__`

```python
__init__(
    config_path:str,
    num_class:Optional[int]=1,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to deberta model 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaForClassification.forward`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DebertaForMaskLM`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaForMaskLM.__init__`

```python
__init__(config_path:str, gradient_checkpointing:Optional[bool]=False)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to deberta model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaForMaskLM.forward`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling_v2.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.deberta.modeling_v2`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling_v2.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DebertaV2ForClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling_v2.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaV2ForClassification.__init__`

```python
__init__(
    config_path:str,
    num_class:Optional[int]=1,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to deberta model 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/modeling_v2.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaV2ForClassification.forward`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.deberta.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DebertaProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaProcessor.__init__`

```python
__init__(
    vocab_path:str,
    merge_path:str,
    max_seq_length:Optional[int]=128,
    source_type_id:Optional[int]=0,
    target_type_id:Optional[int]=0
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in deberta tokenizer 
 - <b>`merge_path`</b>:  merge file path in deberta tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length input text 
 - <b>`source_type_id`</b>:  token type id to text_a 
 - <b>`target_type_id`</b>:  token type id to text_b 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaProcessor.processing_classification`

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


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing_v2.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.deberta.processing_v2`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing_v2.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `DebertaV2Processor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing_v2.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaV2Processor.__init__`

```python
__init__(
    vocab_path:str,
    max_seq_length:Optional[int]=128,
    source_type_id:Optional[int]=0,
    target_type_id:Optional[int]=1
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in deberta v2 tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length input text 
 - <b>`source_type_id`</b>:  token type id to text_a 
 - <b>`target_type_id`</b>:  token type id to text_b 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/deberta/processing_v2.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `DebertaV2Processor.processing_classification`

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

