<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.mass`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.mass.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `MASSForGeneration`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.__init__`

```python
__init__(config_path:str, vocab_path:str)
```






---

###### <kbd>property</kbd> MASSForGeneration.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.forward`

```python
forward(
    tokens_ids_a=None,
    tokens_ids_b=None,
    decoder_input_ids=None,
    encoder_outputs=None,
    incremental_state=None,
    decoder_length=None,
    attention_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None
)
```



**Args:**
 
 - <b>`tokens_ids_a`</b>:  tokens of encode text 
 - <b>`tokens_ids_b`</b>:  tokens of decode text 
 - <b>`others`</b>:  used in beam search Returns: forward logits 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L566"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.generate`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.get_decoder`

```python
get_decoder()
```

Returns the model's decoder. 

**Returns:**
 
 - <b>``nn.Module``</b>:  A torch module decoder to process hidden states. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.get_encoder`

```python
get_encoder()
```

Returns the model's encoder. 

**Returns:**
 
 - <b>``nn.Module``</b>:  A torch module encoder to process hidden states. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.get_input_embeddings`

```python
get_input_embeddings()
```

Returns the model's input embeddings. 

**Returns:**
 
 - <b>``nn.Module``</b>:  A torch module mapping vocabulary to hidden states. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L486"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.prepare_inputs_for_generation`

```python
prepare_inputs_for_generation(
    decoder_input_ids,
    past=None,
    attention_mask=None,
    head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs
)
```

Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/modeling.py#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSForGeneration.set_input_embeddings`

```python
set_input_embeddings(value)
```

Set model's input embeddings. 

**Args:**
 
 - <b>`value`</b> (`nn.Module`):  A module mapping vocabulary to hidden states. 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.mass.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/processing.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `MASSProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/mass/processing.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MASSProcessor.__init__`

```python
__init__(
    vocab_path:str,
    max_seq_length:Optional[int]=128,
    max_gen_seq_length:Optional[int]=30,
    do_lower_case:Optional[bool]=True,
    do_basic_tokenize:Optional[bool]=True,
    special_tokens_ids:Optional[Dict]={}
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in mass tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length encode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length decode text 
 - <b>`do_lower_case`</b>:  if do lower case to input text 
 - <b>`do_basic_tokenize`</b>:  if do basic tokenize to input text 
 - <b>`special_tokens_ids`</b>:  special tokens dict in mass tokenizer 







---

