<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.unilm`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.unilm.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/modeling.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `UnilmForGeneration`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/modeling.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `UnilmForGeneration.__init__`

```python
__init__(config_path)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to unilm model 


---

###### <kbd>property</kbd> UnilmForGeneration.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/modeling.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `UnilmForGeneration.forward`

```python
forward(
    tokens_ids=None,
    attn_mask=None,
    seg_ids=None,
    pos_ids=None,
    decoder_input_ids=None,
    decoder_pos_ids=None,
    decoder_seg_ids=None,
    decoder_attn_mask=None,
    decoder_mask_ids=None,
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
 - <b>`others`</b>:  used in beam search Returns: forward logits 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/modeling.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `UnilmForGeneration.generate`

```python
generate(
    tokens_ids,
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

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/modeling.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `UnilmForGeneration.prepare_inputs_for_generation`

```python
prepare_inputs_for_generation(decoder_input_ids, past=None, **kwargs)
```

Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method. 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.unilm.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/processing.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `UnilmProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/processing.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `UnilmProcessor.__init__`

```python
__init__(
    vocab_path,
    max_seq_length:Optional[int]=128,
    max_gen_seq_length:Optional[int]=30,
    do_lower_case:Optional[bool]=True,
    do_basic_tokenize:Optional[bool]=True,
    special_tokens_ids:Optional[Dict]={},
    source_type_id:Optional[int]=0,
    target_type_id:Optional[int]=1
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in unilm tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length encode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length decode text 
 - <b>`do_lower_case`</b>:  if do lower case to input text 
 - <b>`do_basic_tokenize`</b>:  if do basic tokenize to input text 
 - <b>`special_tokens_ids`</b>:  special tokens dict in mass tokenizer 
 - <b>`source_type_id`</b>:  token type id to encode text 
 - <b>`target_type_id`</b>:  token type id to decode text 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/unilm/processing.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `UnilmProcessor.processing_generation`

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

