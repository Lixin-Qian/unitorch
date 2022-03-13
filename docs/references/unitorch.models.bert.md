<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.bert`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.bert.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/modeling.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `BertForClassification`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/modeling.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BertForClassification.__init__`

```python
__init__(
    config_path:str,
    num_class:Optional[int]=1,
    gradient_checkpointing:Optional[bool]=False
)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to bert model 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/modeling.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BertForClassification.forward`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.bert.processing`





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/processing.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `get_bert_tokenizer`

```python
get_bert_tokenizer(
    vocab_path,
    do_lower_case:bool=True,
    do_basic_tokenize:bool=True,
    special_tokens_ids:Dict={}
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in bert tokenizer 
 - <b>`do_lower_case`</b>:  if do lower case to input text 
 - <b>`do_basic_tokenize`</b>:  if do basic tokenize to input text 
 - <b>`special_tokens_ids`</b>:  special tokens dict in bert tokenizer Returns: return bert tokenizer 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/processing.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `BertProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/processing.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BertProcessor.__init__`

```python
__init__(
    vocab_path,
    max_seq_length:Optional[int]=128,
    special_tokens_ids:Optional[Dict]={},
    do_lower_case:Optional[bool]=True,
    do_basic_tokenize:Optional[bool]=True,
    do_whole_word_mask:Optional[bool]=True,
    masked_lm_prob:Optional[float]=0.15,
    max_predictions_per_seq:Optional[int]=20
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in bert tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length text 
 - <b>`special_tokens_ids`</b>:  special tokens dict in bert tokenizer 
 - <b>`do_lower_case`</b>:  if do lower case to input text 
 - <b>`do_basic_tokenize`</b>:  if do basic tokenize to input text 
 - <b>`do_whole_word_mask`</b>:  if mask whole word in mlm task 
 - <b>`masked_lm_prob`</b>:  mask prob in mlm task 
 - <b>`max_predictions_per_seq`</b>:  max tokens to predict in mlm task 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bert/processing.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BertProcessor.processing_pretrain`

```python
processing_pretrain(
    text:str,
    text_pair:str,
    nsp_label:int,
    max_seq_length:Optional[int]=None,
    masked_lm_prob:Optional[float]=None,
    do_whole_word_mask:Optional[bool]=None,
    max_predictions_per_seq:Optional[int]=None
)
```



**Args:**
 
 - <b>`text`</b>:  text_a to bert pretrain 
 - <b>`text_pair`</b>:  text_b to bert pretrain 
 - <b>`nsp_label`</b>:  nsp label to bert pretrain 
 - <b>`max_seq_length`</b>:  max sequence length text 
 - <b>`masked_lm_prob`</b>:  mask prob in mlm task 
 - <b>`do_whole_word_mask`</b>:  if mask whole word in mlm task 
 - <b>`max_predictions_per_seq`</b>:  max tokens to predict in mlm task 




---

