<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.models`




### **Global Variables**
---------------
- **processing_utils**

- **bart**

- **bert**

- **deberta**

- **detectron2**

- **mass**

- **mbart**

- **prophetnet**

- **roberta**

- **unilm**

- **infoxlm**

- **vlp**

- **xprophetnet**

- **vit**

- **vit_mae**

- **swin**

- **detr**

- **clip**

- **senet**



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `GenericModel`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GenericModel.__init__`

```python
__init__()
```








---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GenericModel.from_checkpoint`

```python
from_checkpoint(ckpt_dir='./cache', weight_name='pytorch_model.bin', **kwargs)
```



**Args:**
 
 - <b>`ckpt_dir`</b>:  checkpoint folder 
 - <b>`weight_name`</b>:  checkpoint name 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GenericModel.from_pretrained`

```python
from_pretrained(weight_path=None, replace_keys:Dict=OrderedDict(), **kwargs)
```



**Args:**
 
 - <b>`weight_path`</b>:  pretrained weight path 
 - <b>`replace_keys`</b>:  keys replacement in weight dict 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GenericModel.init_weights`

```python
init_weights()
```

Initialize the weights 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GenericModel.save_checkpoint`

```python
save_checkpoint(ckpt_dir='./cache', weight_name='pytorch_model.bin', **kwargs)
```



**Args:**
 
 - <b>`ckpt_dir`</b>:  checkpoint folder 
 - <b>`weight_name`</b>:  checkpoint name 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `GenericOutputs`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/models/__init__.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `GenericOutputs.__init__`

```python
__init__(attrs:Dict=OrderedDict(), **kwargs)
```



**Args:**
 
 - <b>`attrs`</b>:  attrs dict to init outputs 







---

