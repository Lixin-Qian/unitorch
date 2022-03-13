<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.bart`




### **Global Variables**
---------------
- **modeling**

- **processing**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.bart.modeling`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `BartForGeneration`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.__init__`

```python
__init__(config_path:str, gradient_checkpointing:Optional[bool]=False)
```



**Args:**
 
 - <b>`config_path`</b>:  config file path to bart model 
 - <b>`gradient_checkpointing`</b>:  if to enable gradient_checkpointing 


---

###### <kbd>property</kbd> BartForGeneration.device

`torch.device`: The device on which the module is (assuming that all the module parameters are on the same device). 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.forward`

```python
forward(
    tokens_ids_a:Optional[torch.Tensor]=None,
    tokens_mask_a:Optional[torch.Tensor]=None,
    tokens_ids_b:Optional[torch.Tensor]=None,
    tokens_mask_b:Optional[torch.Tensor]=None,
    decoder_input_ids:Optional[torch.Tensor]=None,
    attention_mask:Optional[torch.Tensor]=None,
    encoder_outputs=None,
    past_key_values=None,
    output_attentions:Optional[bool]=None,
    output_hidden_states:Optional[bool]=None,
    decoder_length:Optional[int]=None,
    return_dict:Optional[bool]=None
)
```



**Args:**
 
 - <b>`tokens_ids_a`</b>:  tokens of encode text 
 - <b>`tokens_mask_a`</b>:  token masks of encode text 
 - <b>`tokens_ids_b`</b>:  tokens of decode text 
 - <b>`tokens_mask_b`</b>:  token masks of decode text 
 - <b>`others`</b>:  used in beam search Returns: forward logits 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L344"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.generate`

```python
generate(
    tokens_ids,
    num_beams=5,
    decoder_start_token_id=2,
    decoder_end_token_id=2,
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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.get_decoder`

```python
get_decoder()
```

Returns the model's decoder. 

**Returns:**
 
 - <b>``nn.Module``</b>:  A torch module decoder to process hidden states. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.get_encoder`

```python
get_encoder()
```

Returns the model's encoder. 

**Returns:**
 
 - <b>``nn.Module``</b>:  A torch module encoder to process hidden states. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.get_input_embeddings`

```python
get_input_embeddings()
```

Returns the model's input embeddings. 

**Returns:**
 
 - <b>``nn.Module``</b>:  A torch module mapping vocabulary to hidden states. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.prepare_inputs_for_generation`

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

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/modeling.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartForGeneration.set_input_embeddings`

```python
set_input_embeddings(value)
```

Set model's input embeddings. 

**Args:**
 
 - <b>`value`</b> (`nn.Module`):  A module mapping vocabulary to hidden states. 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/processing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models.bart.processing`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/processing.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `BartProcessor`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/bart/processing.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BartProcessor.__init__`

```python
__init__(
    vocab_path:str,
    merge_path:str,
    special_tokens_ids:Optional[Dict]={},
    max_seq_length:Optional[int]=128,
    max_gen_seq_length:Optional[int]=48
)
```



**Args:**
 
 - <b>`vocab_path`</b>:  vocab file path in bart tokenizer 
 - <b>`merge_path`</b>:  merge file path in bart tokenizer 
 - <b>`special_tokens_ids`</b>:  special tokens dict in bart tokenizer 
 - <b>`max_seq_length`</b>:  max sequence length encode text 
 - <b>`max_gen_seq_length`</b>:  max sequence length decode text 







---

