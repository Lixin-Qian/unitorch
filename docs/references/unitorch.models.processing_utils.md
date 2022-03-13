<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.processing_utils`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `HuggingfaceGenerationProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceGenerationProcessor.__init__`

```python
__init__(
    tokenizer:transformers.tokenization_utils.PreTrainedTokenizer,
    max_seq_length:Optional[int]=128,
    max_gen_seq_length:Optional[int]=48
)
```



**Args:**
 
 - <b>`tokenizer`</b>:  a huggingface tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length to decode text 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceGenerationProcessor.processing_decode`

```python
processing_decode(sequences:torch.Tensor, skip_special_tokens:bool=True)
```



**Args:**
 
 - <b>`sequences`</b>:  generation model output tensor 2-dim or 3-dim 
 - <b>`skip_special_tokens`</b>:  if skip special tokens 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceGenerationProcessor.processing_evaluation`

```python
processing_evaluation(text:str, max_gen_seq_length:Optional[int]=None)
```



**Args:**
 
 - <b>`text`</b>:  decode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length to decode text 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceGenerationProcessor.processing_generation`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceGenerationProcessor.processing_inference`

```python
processing_inference(text:str, max_seq_length:Optional[int]=None)
```



**Args:**
 
 - <b>`text`</b>:  encode text 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `HuggingfaceClassificationProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceClassificationProcessor.__init__`

```python
__init__(
    tokenizer:transformers.tokenization_utils.PreTrainedTokenizer,
    max_seq_length:Optional[int]=128,
    source_type_id:Optional[int]=0,
    target_type_id:Optional[int]=1,
    position_start_id:Optional[int]=0
)
```



**Args:**
 
 - <b>`tokenizer`</b>:  a huggingface tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length to encode text 
 - <b>`source_type_id`</b>:  token type id to text_a 
 - <b>`target_type_id`</b>:  token type id to text_b 
 - <b>`position_start_id`</b>:  start id of position 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `HuggingfaceClassificationProcessor.processing_classification`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `GeneralizedProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.__init__`

```python
__init__(
    num_class:Optional[int]=None,
    sep:Optional[str]=',',
    max_seq_length:Optional[int]=128,
    map_dict:Optional[Dict]={}
)
```



**Args:**
 
 - <b>`num_class`</b>:  num class to classification 
 - <b>`sep`</b>:  delimiter to input text 
 - <b>`max_seq_length`</b>:  max sequence length to label sequence 
 - <b>`map_dict`</b>:  label mapping to input text 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.parse_digit`

```python
parse_digit(digit:Union[int, float, str], dtype:Optional[str]='int')
```



**Args:**
 
 - <b>`digit`</b>:  input text/int/float to convert 
 - <b>`dtype`</b>:  target data type Returns: a int/float number 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_digit`

```python
processing_digit(digit:Union[int, float, str], dtype:Optional[str]='int')
```



**Args:**
 
 - <b>`digit`</b>:  input text/int/float to convert 
 - <b>`dtype`</b>:  target data type Returns: a tensor 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_features`

```python
processing_features(
    features:Union[List, str],
    sep:Optional[str]=None,
    dtype:Optional[str]='int',
    shape:Optional[tuple]=None
)
```



**Args:**
 
 - <b>`features`</b>:  input feature list or string to process 
 - <b>`sep`</b>:  delimiter to split features string 
 - <b>`dtype`</b>:  target data type 
 - <b>`shape`</b>:  reshape the process results Returns: a tensor after replacement with map_dict 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_multi_target`

```python
processing_multi_target(text:Union[List, str], sep:Optional[str]=None)
```



**Args:**
 
 - <b>`text`</b>:  input list or string to process 
 - <b>`sep`</b>:  delimiter to split text Returns: a tensor after replacement with map_dict 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_sequence`

```python
processing_sequence(
    text:Union[List, str],
    sep:Optional[str]=None,
    dtype:Optional[str]='int'
)
```



**Args:**
 
 - <b>`text`</b>:  input list or string to process 
 - <b>`sep`</b>:  delimiter to split text 
 - <b>`dtype`</b>:  target data type Returns: a tensor after replacement with map_dict 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/processing_utils.py#L336"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GeneralizedProcessor.processing_target`

```python
processing_target(text:Union[int, float, str], dtype:Optional[str]='int')
```



**Args:**
 
 - <b>`text`</b>:  input text to convert 
 - <b>`dtype`</b>:  target data type Returns: a tensor after replacement with map_dict 




---

